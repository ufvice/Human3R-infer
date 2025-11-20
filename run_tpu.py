import torch
import torch_xla.core.xla_model as xm
import torch_xla
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose
from model import ARCroco3DStereo
from utils import pad_image, load_human3r_weights

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output.pt")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Init TPU
    device = torch_xla.device()
    print(f"Using device: {device}")

    # 2. Load Model
    print("Initializing model...")
    model = ARCroco3DStereo(img_size=512, mhmr_img_res=896)
    load_human3r_weights(model, args.model_path)
    model.to(device)
    model.eval()

    # 3. Prep Data
    print(f"Loading {args.input_image}...")
    img = Image.open(args.input_image).convert("RGB")
    
    # 修改点：确保长宽都是 16 的倍数
    W, H = img.size
    scale = 512 / max(W, H)
    new_W = int(W * scale)
    new_H = int(H * scale)
    # Round to nearest multiple of 16
    new_W = round(new_W / 16) * 16
    new_H = round(new_H / 16) * 16
    print(f"Resizing image to {new_W}x{new_H}")
    
    img = img.resize((new_W, new_H))
    
    transform = Compose([ToTensor(), Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Pad for MHMR
    img_mhmr = pad_image(img_tensor, 896).to(device)
    
    views = [{
        'img': img_tensor,
        'img_mhmr': img_mhmr,
        'true_shape': torch.tensor([[H, W]]).to(device) # original size
    }]

    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        # Note: Autocast removed for TPU
        results = model.forward_recurrent(views)
    
    # TPU Optimization: Explicit synchronization point
    # This tells XLA to compile the entire inference as one graph
    xm.mark_step()
    
    # 5. Save
    print(f"Saving to {args.output_path}...")
    # Move to CPU after XLA graph execution
    cpu_results = [ {k: v.cpu() for k, v in res.items() if isinstance(v, torch.Tensor)} for res in results]
    torch.save(cpu_results, args.output_path)
    print("Done.")

if __name__ == "__main__":
    main()
