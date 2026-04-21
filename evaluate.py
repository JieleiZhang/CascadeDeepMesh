import os
import glob
import torch
import trimesh
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 核心计算函数
# ==========================================
def calc_cd_and_fscore(pc_pred, pc_gt, threshold=0.01):
    """计算单对点云的 CD Loss 和 F-score"""
    if pc_pred.dim() == 2:
        pc_pred = pc_pred.unsqueeze(0)
        pc_gt = pc_gt.unsqueeze(0)

    dist_matrix = torch.cdist(pc_pred, pc_gt, p=2.0)
    min_dist_pred_to_gt, _ = torch.min(dist_matrix, dim=2)
    min_dist_gt_to_pred, _ = torch.min(dist_matrix, dim=1)

    cd_loss = torch.mean(min_dist_pred_to_gt ** 2, dim=1) + torch.mean(min_dist_gt_to_pred ** 2, dim=1)
    
    precision = torch.mean((min_dist_pred_to_gt < threshold).float(), dim=1)
    recall = torch.mean((min_dist_gt_to_pred < threshold).float(), dim=1)
    f_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return cd_loss.item(), f_score.item(), precision.item(), recall.item()

def sample_points_from_mesh(mesh_path, num_points=10000):
    """从 obj 网格表面均匀采样点云"""
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        # 如果模型是由多个部件组成，确保它被当作单一网格处理
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
            
        # 表面均匀采样
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return torch.tensor(points, dtype=torch.float32)
    except Exception as e:
        print(f"\n[Error] 加载或采样失败: {mesh_path} | {e}")
        return None

# ==========================================
# 2. 评估流水线
# ==========================================
def evaluate_dataset(gt_dir, gen_dir, num_samples=8192, threshold=0.01, device='cuda'):
    # 获取所有生成的 obj 文件
    gen_files = glob.glob(os.path.join(gen_dir, "*.obj"))
    print(f"找到 {len(gen_files)} 个生成的数据进行评估...")

    total_cd = 0.0
    total_fscore = 0.0
    valid_count = 0

    # 使用 tqdm 显示进度条
    for gen_path in tqdm(gen_files, desc="Evaluating"):
        gen_filename = os.path.basename(gen_path)
        
        # 解析对应关系
        # 输入格式: val_preview_001_04379243_66255a0a235927ea1b81a92ddeaca85c_dec05_gen_1.obj
        # 目标 GT 格式: val_preview_001_04379243_66255a0a235927ea1b81a92ddeaca85c_dec05.obj
        base_name = gen_filename.split("_gen_")[0]
        gt_filename = f"{base_name}.obj"
        gt_path = os.path.join(gt_dir, gt_filename)

        if not os.path.exists(gt_path):
            print(f"\n[Warning] 找不到对应的 GT 文件: {gt_path}，跳过该样本。")
            continue

        # 加载并采样
        # 注意：为了公平比较，采样数量建议在 8192 或 10000
        pc_pred = sample_points_from_mesh(gen_path, num_points=num_samples)
        pc_gt = sample_points_from_mesh(gt_path, num_points=num_samples)

        if pc_pred is None or pc_gt is None:
            continue

        # 移动到 GPU 提升 cdist 计算速度
        pc_pred = pc_pred.to(device)
        pc_gt = pc_gt.to(device)

        # 计算指标
        cd, f1, p, r = calc_cd_and_fscore(pc_pred, pc_gt, threshold=threshold)
        
        total_cd += cd
        total_fscore += f1
        valid_count += 1

    if valid_count == 0:
        print("没有成功评估任何数据。请检查路径和文件格式。")
        return

    # 输出平均指标
    avg_cd = total_cd / valid_count
    avg_fscore = total_fscore / valid_count

    print("\n" + "="*40)
    print(" 最终评估报告 (Final Evaluation Report) ")
    print("="*40)
    print(f"评估样本总数: {valid_count}")
    print(f"采样点数 (Num Points): {num_samples}")
    print(f"F-score 阈值: {threshold}")
    print("-" * 40)
    print(f"Mean Chamfer Distance (CD): {avg_cd:.6f}")
    print(f"Mean F-score:             {avg_fscore:.4f}")
    print("="*40)

if __name__ == "__main__":
    # 配置路径
    GT_DATA_DIR = "data/preview_meshes/val"
    GEN_DATA_DIR = "/root/iFlame_local/outputs_val_finetune_for_evaluate"
    
    # 根据你的显卡情况选择计算设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluate_dataset(
        gt_dir=GT_DATA_DIR, 
        gen_dir=GEN_DATA_DIR, 
        num_samples=10000,   # 可以调整为 10000
        threshold=0.01,     # 如果网格未归一化到单位立方体[-1, 1]，此阈值需要根据实际尺寸调整
        device=DEVICE
    )