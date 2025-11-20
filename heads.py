import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules import Mlp, ConditionModulationBlock

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class DPTOutputAdapter_fix(nn.Module):
    def __init__(self, num_channels=1, stride_level=1, patch_size=16, 
                 hooks=[0, 1, 2, 3], layer_dims=[96, 192, 384, 768], 
                 feature_dim=256, last_dim=32, dim_tokens_enc=None, **kwargs):
        super().__init__()
        self.hooks = hooks
        self.stride_level = stride_level
        self.patch_size = (patch_size, patch_size)
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)
        
        self.scratch = nn.Module()
        # Simplify scratch blocks creation for brevity
        self.scratch.layer_rn = nn.ModuleList([nn.Conv2d(d, feature_dim, 3, 1, 1, bias=False) for d in layer_dims])
        self.scratch.refinenet1 = self._make_fusion(feature_dim)
        self.scratch.refinenet2 = self._make_fusion(feature_dim)
        self.scratch.refinenet3 = self._make_fusion(feature_dim)
        self.scratch.refinenet4 = self._make_fusion(feature_dim)

        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, 1, 1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(feature_dim // 2, last_dim, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(last_dim, num_channels, 1, 1, 0)
        )
        
        self.dim_tokens_enc = dim_tokens_enc
        self.act_postprocess = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim_tokens_enc[0], layer_dims[0], 1, 1, 0),
                nn.ConvTranspose2d(layer_dims[0], layer_dims[0], 4, 4, 0)
            ),
             nn.Sequential(
                nn.Conv2d(dim_tokens_enc[1], layer_dims[1], 1, 1, 0),
                nn.ConvTranspose2d(layer_dims[1], layer_dims[1], 2, 2, 0)
            ),
             nn.Sequential(nn.Conv2d(dim_tokens_enc[2], layer_dims[2], 1, 1, 0)),
             nn.Sequential(
                 nn.Conv2d(dim_tokens_enc[3], layer_dims[3], 1, 1, 0),
                 nn.Conv2d(layer_dims[3], layer_dims[3], 3, 2, 1)
            ),
        ])

    def _make_fusion(self, features):
        # Simplified fusion block
        return FusionBlock(features)

    def forward(self, encoder_tokens, image_size, ret_feat=False):
        H, W = image_size
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        layers = [encoder_tokens[hook] for hook in self.hooks]
        # Adapt tokens (remove cls if needed, usually just reshape)
        layers = [rearrange(l, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W) for l in layers]
        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        path_4 = self.scratch.refinenet4(layers[3])
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])
        
        out = self.head(path_1)
        if ret_feat: return out, path_1
        return out

class FusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.resConfUnit1 = nn.Sequential(nn.ReLU(), nn.Conv2d(features, features, 3, 1, 1), nn.ReLU(), nn.Conv2d(features, features, 3, 1, 1))
        self.resConfUnit2 = nn.Sequential(nn.ReLU(), nn.Conv2d(features, features, 3, 1, 1), nn.ReLU(), nn.Conv2d(features, features, 3, 1, 1))
        self.out_conv = nn.Conv2d(features, features, 1, 1, 0)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            res = F.interpolate(res, size=output.shape[2:], mode="bilinear")
            output = output + res
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(output)

class PoseDecoder(nn.Module):
    def __init__(self, hidden_size=768, target_dim=7):
        super().__init__()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * 4), out_features=target_dim)
    def forward(self, x): return self.mlp(x)

class SMPLDecoder(nn.Module):
    def __init__(self, hidden_size=768, target_dim=1, num_layers=2):
        super().__init__()
        # Simplified flexible MLP
        layers = []
        in_dim = hidden_size
        for i in range(num_layers):
            out_dim = target_dim if i == num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1: layers.append(nn.GELU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
    def forward(self, x): return self.mlp(x)

class DPTPts3dPoseSMPL(nn.Module):
    def __init__(self, net, has_conf=False, has_rgb=False, has_pose=False, has_msk=False):
        super().__init__()
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.pose_mode = net.pose_mode
        self.has_pose = has_pose
        self.has_rgb = has_rgb
        
        pts_channels = 3 + has_conf
        ed, dd = net.enc_embed_dim, net.dec_embed_dim
        
        # DPT Heads
        self.dpt_self = DPTOutputAdapter_fix(num_channels=pts_channels, dim_tokens_enc=[ed, dd, dd, dd], last_dim=128)
        self.dpt_cross = DPTOutputAdapter_fix(num_channels=pts_channels, dim_tokens_enc=[ed, dd, dd, dd], last_dim=128)
        self.final_transform = nn.ModuleList([ConditionModulationBlock(dd, net.dec_num_heads, rope=net.rope) for _ in range(2)])

        if has_rgb:
            self.dpt_rgb = DPTOutputAdapter_fix(num_channels=3, dim_tokens_enc=[ed, dd, dd, dd], last_dim=128)
        if has_pose:
            self.pose_head = PoseDecoder(hidden_size=dd)
        
        # SMPL Heads
        backbone_dim = net.backbone_dim # MHMR dim (1024 for ViT-L)
        self.mlp_classif = nn.Sequential(nn.Linear(backbone_dim, backbone_dim), nn.ReLU(), nn.Linear(backbone_dim, 1))
        self.mlp_offset = nn.Sequential(nn.Linear(backbone_dim, backbone_dim), nn.ReLU(), nn.Linear(backbone_dim, 2))
        self.mlp_fuse = SMPLDecoder(hidden_size=ed+backbone_dim, target_dim=dd, num_layers=2)
        
        self.deccam = SMPLDecoder(hidden_size=dd, target_dim=3, num_layers=2)
        # npose = 6d * 53 joints (SMPL-X) = 318
        self.decpose = SMPLDecoder(hidden_size=dd+backbone_dim, target_dim=318, num_layers=2) # 6d rot
        self.decshape = SMPLDecoder(hidden_size=dd+backbone_dim, target_dim=10, num_layers=2)
        self.decexpression = SMPLDecoder(hidden_size=dd+backbone_dim, target_dim=10, num_layers=2)
        
        # Init buffers (will be loaded from ckpt)
        self.register_buffer('init_body_pose', torch.zeros(1, 318))
        self.register_buffer('init_betas', torch.zeros(1, 10))
        self.register_buffer('init_cam', torch.zeros(1, 3))
        self.register_buffer('init_expression', torch.zeros(1, 10))

    def detect_mhmr(self, x):
        return torch.sigmoid(self.mlp_classif(x))

    def forward(self, x, img_info, **kwargs):
        # This forward is called inside the loop, x is a list of layer outputs
        # Logic extracted from original dpt_head.py
        # ... (Simplified logic for inference)
        return {}
