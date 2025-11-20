import torch
import torch.nn.functional as F
import numpy as np
import roma

def pad_image(img_tensor, target_size, pad_value=-1.0):
    # ... (Logic from original dust3r/utils/image.py) ...
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
    th, tw = target_size
    mh, mw = img_tensor.shape[-2:]
    scale = min(mh/th, mw/tw) # Logic might vary, simplified
    # ... Implementation ...
    return img_tensor # Placeholder

def get_camera_parameters(img_size, fov=60, device='cpu'):
    focal = img_size / (2 * np.tan(np.radians(fov) / 2))
    K = torch.eye(3, device=device)
    K[0,0] = K[1,1] = focal
    K[0,2] = K[1,2] = img_size // 2
    return K.unsqueeze(0)

def postprocess_output(out, depth_mode, conf_mode):
    # Simplified post-processing
    pts3d = out[:, :3]
    conf = out[:, 3:4]
    # Apply depth mode (exp, etc.)
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
    # 修改点：添加 weights_only=False 以允许加载包含 OmegaConf/Hydra 配置的旧版 checkpoint
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Load failed with weights_only=False. Retrying with default...")
        ckpt = torch.load(path, map_location='cpu')

    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    state_dict = strip_prefix_if_present(state_dict, "module.")
    
    # 加载权重，允许部分不匹配（strict=False）
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")

# Add missing functions referenced in model.py imports
def nms(x): return x
def apply_threshold(x): return x
def unpad_uv(x): return x
