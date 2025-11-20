import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_result(pt_path):
    print(f"Loading {pt_path}...")
    # 加载结果 (list of dicts)
    results = torch.load(pt_path, map_location='cpu')
    
    # 取第一张图的结果
    res = results[0]
    
    # 1. 获取 3D 点云 [H, W, 3]
    pts3d = res['pts3d'].float().numpy().squeeze()
    # 2. 获取 置信度 [H, W]
    conf = res['conf'].float().numpy().squeeze()
    
    # 提取深度 (Z轴)
    # 注意：Human3R/Dust3r 的坐标系通常 Z 是深度
    depth = pts3d[..., 2]
    
    # 简单的数值检查
    print(f"Depth Range: Min={depth.min():.4f}, Max={depth.max():.4f}")
    print(f"Conf Range: Min={conf.min():.4f}, Max={conf.max():.4f}")
    
    if np.all(depth == 0):
        print("⚠️ 警告: 深度全为 0，模型推理可能失败！")
        return

    # 绘图
    plt.figure(figsize=(15, 5))
    
    # 绘制深度图
    plt.subplot(1, 3, 1)
    plt.title("Predicted Depth (Z)")
    # 使用 robust 的范围，过滤掉极值以便看清细节
    vmin, vmax = np.percentile(depth, [2, 98])
    plt.imshow(depth, cmap='magma', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.axis('off')

    # 绘制置信度
    plt.subplot(1, 3, 2)
    plt.title("Confidence")
    plt.imshow(conf, cmap='viridis')
    plt.colorbar()
    plt.axis('off')
    
    # 绘制 3D 点云的 X 坐标 (检查几何结构)
    plt.subplot(1, 3, 3)
    plt.title("Structure (X-coord)")
    plt.imshow(pts3d[..., 0], cmap='coolwarm')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("verify_output.png")
    print("结果已保存为 verify_output.png，请查看图片。")

if __name__ == "__main__":
    visualize_result("result.pt")