import torch
import torch.nn as nn
from functools import partial
from modules import Block, DecoderBlock, PatchEmbed, RoPE2D
from heads import DPTPts3dPoseSMPL
from utils import pad_image, nms, apply_threshold, unpad_uv, unpad_image, postprocess_output
from einops import rearrange

class Dinov2Backbone(nn.Module):
    def __init__(self, name='dinov2_vitl14'):
        super().__init__()
        self.encoder = torch.hub.load('facebookresearch/dinov2', name)
        self.embed_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_size

    def forward(self, x):
        return self.encoder.get_intermediate_layers(x)[0]

class LocalMemory(nn.Module):
    def __init__(self, size, k_dim, v_dim, num_heads, rope=None):
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.mem = nn.Parameter(torch.randn(1, size, 2 * v_dim) * 0.2)
        self.read_blocks = nn.ModuleList([
            DecoderBlock(2 * v_dim, num_heads, rope=rope) for _ in range(2)
        ])
        self.write_blocks = nn.ModuleList([
            DecoderBlock(2 * v_dim, num_heads, rope=rope) for _ in range(2)
        ])

    def inquire(self, query, mem):
        x = self.proj_q(query)
        masked = torch.zeros_like(x) 
        x = torch.cat([x, masked], dim=-1)
        for blk in self.read_blocks:
            x, _, _ = blk(x, mem, None, None)
        return x[..., -self.v_dim :]
    
    def update_mem(self, mem, feat_k, feat_v):
        feat_k = self.proj_q(feat_k)
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _, _ = blk(mem, feat, None, None)
        return mem

class ARCroco3DStereo(nn.Module):
    def __init__(self, 
                 img_size=512, patch_size=16, 
                 enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,
                 dec_embed_dim=768, dec_depth=12, dec_num_heads=12,
                 mhmr_img_res=896):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.dec_depth = dec_depth
        self.dec_num_heads = dec_num_heads
        self.mhmr_img_res = mhmr_img_res
        self.depth_mode = ('exp', -float('inf'), float('inf'))
        self.conf_mode = ('exp', 1, float('inf'))
        self.pose_mode = ('exp', -float('inf'), float('inf'))

        # Components
        self.rope = RoPE2D(freq=100.0, max_seq_len=5000)  # TPU-friendly: pre-allocate position encodings
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)
        self.patch_embed_ray_map = PatchEmbed(img_size, patch_size, 6, enc_embed_dim)
        
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, rope=self.rope) for _ in range(enc_depth)
        ])
        self.enc_blocks_ray_map = nn.ModuleList([
            Block(enc_embed_dim, 16, 4, rope=self.rope) for _ in range(2)
        ])
        self.enc_norm = nn.LayerNorm(enc_embed_dim)
        self.enc_norm_ray_map = nn.LayerNorm(enc_embed_dim)
        
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim)
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, rope=self.rope) for _ in range(dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(dec_embed_dim)
        
        # State Encoder/Decoder
        self.state_size = 768
        self.register_tokens = nn.Embedding(self.state_size, enc_embed_dim)
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim)
        self.dec_blocks_state = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, rope=self.rope) for _ in range(dec_depth)
        ])
        self.dec_norm_state = nn.LayerNorm(dec_embed_dim)
        
        # Special Tokens
        self.masked_img_token = nn.Parameter(torch.randn(1, enc_embed_dim))
        self.masked_ray_map_token = nn.Parameter(torch.randn(1, enc_embed_dim))
        self.masked_smpl_token = nn.Parameter(torch.randn(1, enc_embed_dim))
        self.pose_token = nn.Parameter(torch.randn(1, 1, dec_embed_dim))
        
        # Memory
        self.pose_retriever = LocalMemory(256, enc_embed_dim, dec_embed_dim, dec_num_heads, rope=None)
        
        # MHMR Parts
        self.backbone = Dinov2Backbone('dinov2_vitl14')
        self.backbone_dim = self.backbone.embed_dim
        self.bb_patch_size = self.backbone.patch_size
        self.bb_token_res = mhmr_img_res // self.bb_patch_size
        self.mhmr_masked_img_token = nn.Parameter(torch.randn(1, self.backbone_dim))
        self.mhmr_masked_smpl_token = nn.Parameter(torch.randn(1, self.backbone_dim))
        
        # Head
        self.downstream_head = DPTPts3dPoseSMPL(
            self, has_conf=True, has_rgb=True, has_pose=True, has_msk=True
        )

    def _encode_image(self, image):
        x, pos = self.patch_embed(image)
        for blk in self.enc_blocks:
            x = blk(x, pos)
        return [self.enc_norm(x)], pos

    def _init_state(self, device, batch_size):
        # Simple state init
        state_feat = self.register_tokens(torch.arange(self.state_size, device=device))
        state_feat = state_feat.unsqueeze(0).expand(batch_size, -1, -1)
        # 2D positional encoding for state
        width = int(self.state_size**0.5)
        width = width + 1 if width % 2 == 1 else width
        state_pos = torch.tensor([[i // width, i % width] for i in range(self.state_size)], device=device)
        state_pos = state_pos.unsqueeze(0).expand(batch_size, -1, -1)
        return self.decoder_embed_state(state_feat), state_pos

    def forward_recurrent(self, views):
        # Simplified inference loop
        device = views[0]['img'].device
        batch_size = views[0]['img'].shape[0]
        
        # Init State
        state_feat, state_pos = self._init_state(device, batch_size)
        mem = self.pose_retriever.mem.expand(batch_size, -1, -1)
        
        results = []
        
        for i, view in enumerate(views):
            # 1. Encode Image
            img_tensor = view['img']
            current_H, current_W = img_tensor.shape[-2:]
            
            feat, pos = self._encode_image(img_tensor)
            feat, pos = feat[-1], pos

            # 2. Encode MHMR (Dinov2)
            img_mhmr = view.get('img_mhmr') 
            mhmr_feat = self.backbone(img_mhmr) 

            # 3. Detection
            scores = self.downstream_head.detect_mhmr(mhmr_feat)
            
            # 4. Recurrent Step
            pose_feat = self.pose_retriever.inquire(feat.mean(1, keepdim=True), mem)
            
            # Construct Pose Position Encoding (-1) to fix shape mismatch
            pose_pos = torch.full(
                (batch_size, 1, 2), 
                fill_value=-1, 
                dtype=pos.dtype, 
                device=device
            )
            
            # Concatenate features and positions
            f_img_input = self.decoder_embed(feat)
            f_img_input = torch.cat([pose_feat, f_img_input], dim=1) 
            
            pos_input = torch.cat([pose_pos, pos], dim=1)
            
            f_state = state_feat
            
            all_layers = [f_img_input]
            
            for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
                f_state, _, _ = blk_state(f_state, f_img_input, state_pos, pos_input)
                f_img_input, _, _ = blk_img(f_img_input, f_state, pos_input, state_pos)
                all_layers.append(f_img_input)
                
            # 5. Head Prediction
            L = len(all_layers)
            # Pass Raw Encoder Feat (Index 0) and Decoder Features (Indices 1-3)
            # Decoder features must slice [:, 1:] to remove pose token
            head_input = [
                feat, 
                all_layers[L//2][:, 1:], 
                all_layers[L*3//4][:, 1:], 
                all_layers[-1][:, 1:]
            ]
            
            # Run Head (Output: [B, 4, H, W])
            res_tensor = self.downstream_head.dpt_self(head_input, image_size=(current_H, current_W))
            
            # Postprocess (Tensor -> Dict, Permute dims)
            res_dict = postprocess_output(res_tensor)
            
            results.append(res_dict)
            
            # Update Memory
            mem = self.pose_retriever.update_mem(mem, feat.mean(1, keepdim=True), all_layers[-1][:, 0:1])

        return results
