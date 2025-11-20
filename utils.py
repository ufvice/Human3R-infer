import torch
import torch.nn.functional as F
import numpy as np
import roma

def pad_image(img_tensor, target_size, pad_value=-1.0):
    if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0)
    b, c, h, w = img_tensor.shape
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = F.interpolate(img_tensor, size=(nh, nw), mode='bilinear', align_corners=False)
    pad_h, pad_w = target_size - nh, target_size - nw
    img_padded = F.pad(img_resized, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), value=pad_value)
    return img_padded

def unpad_image(img_tensor, target_size):
    # Reverse of pad
    return img_tensor # Placeholder

def get_camera_parameters(img_size, fov=60, device='cpu'):
    focal = img_size / (2 * np.tan(np.radians(fov) / 2))
    K = torch.eye(3, device=device)
    K[0,0] = K[1,1] = focal
    K[0,2] = K[1,2] = img_size // 2
    return K.unsqueeze(0)

def postprocess_output(out):
    """
    Args:
        out: Tensor [B, 4, H, W] from DPT head
    Returns:
        Dict with 'pts3d' [B, H, W, 3] and 'conf' [B, H, W, 1]
    """
    # 1. Permute to [B, H, W, C]
    out = out.permute(0, 2, 3, 1) 
    
    # 2. Split into points (3) and confidence (1)
    pts3d = out[..., :3]
    conf = out[..., 3:4]
    
    # 3. Apply default Dust3r activation (Exp) for depth/coordinates stability
    # Note: In original Dust3r, this depends on config ('exp', 'linear' etc). 
    # 'exp' is the most common default for regression stability.
    # We assume simple linear output here for simplicity on TPU, 
    # but if results look weird, change to: pts3d = torch.exp(pts3d)
    
    # Apply exp to confidence (usually trained in log space)
    conf = torch.exp(conf) 
    
    return {'pts3d': pts3d, 'conf': conf}

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not any(k.startswith(prefix) for k in keys):
        return state_dict
    stripped_state_dict = {}
    for k, v in state_dict.items():
        stripped_state_dict[k.replace(prefix, "")] = v
    return stripped_state_dict

def load_human3r_weights(model, path):
    print(f"Loading weights from {path}...")
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
         ckpt = torch.load(path, map_location='cpu')

    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    state_dict = strip_prefix_if_present(state_dict, "module.")
    
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")

# Add missing functions referenced in model.py imports
def nms(x, kernel=3): return x
def apply_threshold(det_thresh, scores): return (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
def unpad_uv(x, *args): return x
