import os
import sys
import torch
import numpy as np
import trimesh
import shutil
import tempfile
from models.transformer import *

# ⚠️ 请确保这里的路径能正确导入你的 Tokenizer
from utils.tokenizer import CascadedTokenizer 

import numpy as np
import trimesh
import torch

def load_custom_point_cloud(file_path, num_points=4096, device='cuda', expected_channels=6):
    """
    优化后的点云加载函数
    num_points: 建议与训练时的点数保持一致（你之前的日志显示是 4096）
    expected_channels: 关键参数！如果模型训练时用的是 XYZ+Normal，这里必须是 6
    """
    print(f"📥 正在加载外部点云: {file_path}")
    
    # 1. 读取数据
    if file_path.endswith('.npy'):
        pc = np.load(file_path)
        if len(pc.shape) == 3: # 兼容 [1, N, C]
            pc = pc.squeeze(0)
    else:
        try:
            loaded_data = trimesh.load(file_path)
            if isinstance(loaded_data, trimesh.Scene):
                geom = next(iter(loaded_data.geometry.values()))
            else:
                geom = loaded_data

            if isinstance(geom, trimesh.PointCloud):
                pc_coords = np.array(geom.vertices)
                if hasattr(geom, 'vertex_normals') and geom.vertex_normals is not None and len(geom.vertex_normals) > 0:
                    normals = np.array(geom.vertex_normals)
                else:
                    print("⚠️ 警告：点云缺乏法向量，填入随机小噪点。")
                    normals = np.random.normal(0, 1e-3, pc_coords.shape)
                pc = np.concatenate([pc_coords, normals], axis=-1)

            elif isinstance(geom, trimesh.Trimesh):
                # 采样点数增加一点保险量
                pc_coords, face_indices = trimesh.sample.sample_surface(geom, num_points)
                normals = geom.face_normals[face_indices]
                pc = np.concatenate([pc_coords, normals], axis=-1)
            else:
                raise ValueError(f"无法处理的类型: {type(geom)}")
        except Exception as e:
            print(f"❌ 读取文件失败: {e}，改用全零占位。")
            pc = np.zeros((num_points, expected_channels))

    # ==========================================
    # 🔍 关键修复 1：通道对齐
    # ==========================================
    # 如果模型需要 6 通道但我们只有 3 通道，补齐它
    if pc.shape[1] == 3 and expected_channels == 6:
        print("⚠️ 模型需要 6 通道但输入只有 3 通道，正在补全零法向量...")
        dummy_normals = np.zeros_like(pc)
        pc = np.concatenate([pc, dummy_normals], axis=-1)
    # 如果有多余通道，截断它
    elif pc.shape[1] > expected_channels:
        pc = pc[:, :expected_channels]

    # 2. 采样到固定点数
    if len(pc) > num_points:
        indices = np.random.choice(len(pc), num_points, replace=False)
        pc = pc[indices]
    elif len(pc) < num_points:
        indices = np.random.choice(len(pc), num_points, replace=True)
        pc = pc[indices]

    # ==========================================
    # 🔍 关键修复 2：严格的归一化 (保持与训练一致)
    # ==========================================
    coords = pc[:, :3]
    # 中心化
    center = (coords.min(axis=0) + coords.max(axis=0)) / 2.0
    coords -= center
    # 缩放：将最大边长缩放到 1.0 (即范围约为 [-0.5, 0.5])
    scale = (coords.max(axis=0) - coords.min(axis=0)).max()
    if scale > 0:
        coords /= scale
    
    pc[:, :3] = coords
    
    # 法向量单位化
    if pc.shape[1] == 6:
        norms = np.linalg.norm(pc[:, 3:], axis=1, keepdims=True)
        pc[:, 3:] = pc[:, 3:] / (norms + 1e-8)

    # ==========================================
    # 🔍 关键修复 3：防止双重 unsqueeze
    # ==========================================
    pc_tensor = torch.tensor(pc, dtype=torch.float32).to(device)
    
    # 确保返回的形状是 [1, num_points, channels]
    if pc_tensor.dim() == 2:
        pc_tensor = pc_tensor.unsqueeze(0)
    
    print(f"📊 引导点云准备完毕: {pc_tensor.shape}, 范围: [{pc_tensor.min():.2f}, {pc_tensor.max():.2f}]")
    return pc_tensor

def main():
    # 1. 环境与基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CascadedTokenizer()
    
    # 路径配置
    ckpt_path = "checkpoints/iFlame_best_1.pth"
    my_pc_path = "data/preview_meshes/val/val_preview_021_03211117_a6ce00ec813199804e2b01a3e0fd3197_dec05.obj"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 2. 初始化模型
    print(f"🤖 正在初始化 iFlame 架构...")
    # 请确保参数与训练时一致 (num_categories, embed_dim, num_heads)
    model = Causal_iFlame(num_categories=tokenizer.vocab_size, embed_dim=512, num_heads=16)

    # 3. 加载权重 (带清洗逻辑)
    print(f"📦 正在从 {ckpt_path} 加载权重...")
    if not os.path.exists(ckpt_path):
        print(f"❌ 错误：找不到权重文件 {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # 自动移除 DataParallel 产生的 module. 或 _orig_mod. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('_orig_mod.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device).eval()
    print("✅ 权重加载成功！")

    # 4. 准备引导点云
    try:
        # 使用我们之前优化过的加载函数，确保返回 [1, 4096, 6] 的 Tensor
        test_pc = load_custom_point_cloud(my_pc_path, num_points=4096, device=device, expected_channels=6)
    except Exception as e:
        print(f"❌ 读取点云失败: {e}，将使用随机噪声。")
        test_pc = torch.randn(1, 4096, 6).to(device)

    # 5. 启动自回归生成
    print("🧠 正在生成 3D 序列 (状态机强制语法约束)...")
    # 构建初始输入: [SOS]
    input_ids = torch.tensor([[tokenizer.SOS_TOKEN]], dtype=torch.long).to(device)

    with torch.no_grad():
        generated_sequence = model.generate_sequence(
            initial_input=input_ids,
            pc=test_pc,
            max_seq_len=2816,      # 建议设为 2048 或 2304 以提速，2816 是绝对安全上限
            device=device,
            end_symbol=tokenizer.EOS_TOKEN,
            temperature=0.6,       # 针对 3D 坐标建议 0.4 - 0.7 之间
            top_k=30,              # 收紧采样范围，减少飞点
            top_p=0.95         
        )[0] # 取出 batch 中的第一个结果

    # 6. 保存原始 Token 序列 (用于备份分析)
    token_save_path = os.path.join(output_dir, "generated_tokens.npy")
    np.save(token_save_path, generated_sequence)
    print(f"💾 原始 Token 已保存至: {token_save_path}")

    # 7. 精准级联解码逻辑
    print("🔨 正在执行精准级联解码...")
    # 移除 Padding
    real_sequence = generated_sequence[generated_sequence != tokenizer.PAD_TOKEN]

    try:
        # 分段逻辑：Patch -> Block -> Offset
        sep_indices = np.where(real_sequence == tokenizer.SEP_TOKEN)[0]
        if len(sep_indices) < 1:
            print("❌ 错误：生成序列不完整，未找到分隔符 SEP。")
            return

        # 切分 Patch 部分
        bridge1_idx = sep_indices[0]
        part_p = real_sequence[1:bridge1_idx] 
        
        # 在剩余部分找 Block
        rest = real_sequence[bridge1_idx + 1 :]
        block_mask = (rest >= tokenizer.BLOCK_BASE) & (rest < tokenizer.OFFSET_BASE)
        block_indices = np.where(block_mask)[0]
        
        if len(block_indices) == 0:
            print("❌ 错误：序列中未找到有效的 Block 数据。")
            return
            
        last_block_idx = block_indices[-1]
        part_b = rest[:last_block_idx + 1]
        part_o = rest[last_block_idx + 1:] # 剩下的即为 Offset 区

        print(f"✂️ 序列切分成功 -> Patch: {len(part_p)}, Block: {len(part_b)}, Offset: {len(part_o)}")

        # 构造 tokenizer.decode_mesh 期望的输入格式
        def wrap_tokens(tokens):
            return np.concatenate([[tokenizer.SOS_TOKEN], tokens, [tokenizer.EOS_TOKEN]])

        p_seq = wrap_tokens(part_p)
        b_seq = wrap_tokens(part_b)
        o_seq = wrap_tokens(part_o)

        # 解码为 trimesh 对象
        mesh_recon, _ = tokenizer.decode_mesh(p_seq, b_seq, o_seq)

        # 8. 安全保存结果 (防止 WebDAV 异常)
        # 自动生成不重复的文件名
        base_name = "generated_iflame_result"
        idx = 1
        while os.path.exists(os.path.join(output_dir, f"{base_name}_{idx}.obj")):
            idx += 1
        
        final_mesh_name = f"{base_name}_{idx}.obj"
        final_pc_name = f"{base_name}_{idx}_input_pc.ply"
        
        mesh_path = os.path.join(output_dir, final_mesh_name)
        pc_path = os.path.join(output_dir, final_pc_name)

        # --- A. 保存生成的 Mesh ---
        if isinstance(mesh_recon, trimesh.Trimesh):
            mesh_recon.export(mesh_path)
        else:
            # 备选：手动构造 mesh
            faces = np.arange(len(mesh_recon)).reshape(-1, 3)
            trimesh.Trimesh(vertices=mesh_recon, faces=faces).export(mesh_path)
        
        # --- B. 安全保存引导点云 (修复 CUDA 报错) ---
        # 关键：先 .cpu().numpy()
        pc_np = test_pc.squeeze(0).detach().cpu().numpy()
        # 创建本地临时文件作为中转
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            trimesh.PointCloud(pc_np[:, :3]).export(tmp_path)
            shutil.copy2(tmp_path, pc_path) # 稳定写入目标目录
            print(f"✅ 引导点云已保存: {pc_path}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

        print(f"🎉 任务完美完成！3D 模型见: {mesh_path}")

    except Exception as e:
        print(f"❌ 解码或保存阶段发生崩溃: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()