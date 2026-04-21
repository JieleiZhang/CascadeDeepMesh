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

def load_custom_point_cloud(file_path, num_points=4096, device='cuda', expected_channels=6):
    """优化后的点云加载函数"""
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
                pc_coords, face_indices = trimesh.sample.sample_surface(geom, num_points)
                normals = geom.face_normals[face_indices]
                pc = np.concatenate([pc_coords, normals], axis=-1)
            else:
                raise ValueError(f"无法处理的类型: {type(geom)}")
        except Exception as e:
            print(f"❌ 读取文件失败: {e}，改用全零占位。")
            pc = np.zeros((num_points, expected_channels))

    # 🔍 关键修复 1：通道对齐
    if pc.shape[1] == 3 and expected_channels == 6:
        dummy_normals = np.zeros_like(pc)
        pc = np.concatenate([pc, dummy_normals], axis=-1)
    elif pc.shape[1] > expected_channels:
        pc = pc[:, :expected_channels]

    # 2. 采样到固定点数
    if len(pc) > num_points:
        indices = np.random.choice(len(pc), num_points, replace=False)
        pc = pc[indices]
    elif len(pc) < num_points:
        indices = np.random.choice(len(pc), num_points, replace=True)
        pc = pc[indices]

    # 🔍 关键修复 2：严格的归一化 (保持与训练一致)
    coords = pc[:, :3]
    center = (coords.min(axis=0) + coords.max(axis=0)) / 2.0
    coords -= center
    scale = (coords.max(axis=0) - coords.min(axis=0)).max()
    if scale > 0:
        coords /= scale
    
    pc[:, :3] = coords
    
    # 法向量单位化
    if pc.shape[1] == 6:
        norms = np.linalg.norm(pc[:, 3:], axis=1, keepdims=True)
        pc[:, 3:] = pc[:, 3:] / (norms + 1e-8)

    # 🔍 关键修复 3：防止双重 unsqueeze
    pc_tensor = torch.tensor(pc, dtype=torch.float32).to(device)
    if pc_tensor.dim() == 2:
        pc_tensor = pc_tensor.unsqueeze(0)
    
    return pc_tensor

def main():
    # ==========================================
    # 🎛️ 控制台打印开关 (设为 True 可查看详细对比与诊断信息)
    # ==========================================
    VERBOSE = False 

    # 1. 环境与基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CascadedTokenizer()
    
    ckpt_path = "checkpoints/iFlame_400_finetune_epoch_50.pth"
    my_pc_path = "data/preview_meshes/train/train_preview_001_04379243_e94dcd39a8e438851b17743c18fb63dc_dec05.obj"
    output_dir = "outputs_train"
    os.makedirs(output_dir, exist_ok=True)

    # 2. 初始化模型
    print("🤖 初始化 iFlame 架构并加载权重...")
    model = Causal_iFlame(num_categories=tokenizer.vocab_size, embed_dim=768, num_heads=16)

    if not os.path.exists(ckpt_path):
        print(f"❌ 错误：找不到权重文件 {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # 清洗 module 前缀
    new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device).eval()

    # 4. 准备引导点云
    try:
        test_pc = load_custom_point_cloud(my_pc_path, num_points=4096, device=device, expected_channels=6)
    except Exception as e:
        print(f"❌ 读取点云失败: {e}，使用随机噪声。")
        test_pc = torch.randn(1, 4096, 6).to(device)

    # 5. 准备初始输入 Token
    prompt_obj_path = my_pc_path 
    prompt_ratio = 0 
    
    def normalize_vertices(vertices):
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        return vertices

    full_seq = None
    prefix_length = 0

    try:
        prompt_mesh = trimesh.load(prompt_obj_path, force='mesh', process=False)
        prompt_mesh.vertices = normalize_vertices(prompt_mesh.vertices.copy())
        
        raw_tokens = tokenizer.encode_mesh(prompt_mesh)
        split_data = tokenizer.split_tokens(raw_tokens)
        
        part_p, part_b, part_o = split_data['stage1'][:-1], split_data['stage2'][1:-1], split_data['stage3'][1:]
        sep = np.array([tokenizer.SEP_TOKEN], dtype=np.int64)
        
        full_seq = np.concatenate([part_p, sep, part_b, sep, part_o])
        total_len = len(full_seq)

        prefix_length = max(1, min(total_len, int(total_len * prompt_ratio)))
        initial_tokens = full_seq[:prefix_length].tolist()
        
        if initial_tokens[0] != tokenizer.SOS_TOKEN:
            initial_tokens = [tokenizer.SOS_TOKEN] + initial_tokens
            prefix_length = len(initial_tokens) 
            
        input_ids = torch.tensor([initial_tokens], dtype=torch.long).to(device)

    except Exception as e:
        print(f"⚠️ 提取初始 Token 失败，详细原因: {e}")
        input_ids = torch.tensor([[tokenizer.SOS_TOKEN]], dtype=torch.long).to(device)

    # 6. 自回归生成
    print("🚀 正在生成 3D 序列...")
    with torch.no_grad():
        generated_sequence = model.generate_sequence(
            initial_input=input_ids,
            pc=test_pc,
            max_seq_len=2816,      
            device=device,
            end_symbol=tokenizer.EOS_TOKEN,
            temperature=0.4,       
            top_k=15,              
            top_p=0.9         
        )[0]
        
    # ==========================================
    # 7. 🔍 输出与原始 Token 的对比分析 (仅在 VERBOSE开启 时显示)
    # ==========================================
    if VERBOSE:
        print("\n" + "="*40 + "\n 📈 Token 级生成质量对比\n" + "="*40)
        if full_seq is not None and prefix_length > 0:
            valid_generated = generated_sequence.cpu().numpy() if isinstance(generated_sequence, torch.Tensor) else np.array(generated_sequence)
            valid_generated = valid_generated[valid_generated != tokenizer.PAD_TOKEN]
            
            generated_tail = valid_generated[prefix_length:]
            original_tail = full_seq[prefix_length:]
            
            print(f"📏 真实长度: {len(original_tail)} | 续写长度: {len(generated_tail)} | 差异: {abs(len(generated_tail) - len(original_tail))}")
            
            min_len = min(len(generated_tail), len(original_tail))
            if min_len > 0:
                exact_matches = np.sum(generated_tail[:min_len] == original_tail[:min_len])
                print(f"🎯 逐位匹配: {exact_matches}/{min_len} ({(exact_matches/min_len)*100:.2f}%)")
                print(f"📏 L1 距离: {np.mean(np.abs(generated_tail[:min_len] - original_tail[:min_len])):.2f}")
                print(f"👻 新颖 Token 数量: {len(set(generated_tail) - set(original_tail))}")
        else:
            print("⚠️ 未提取到原始序列，跳过对比。")
        print("="*40 + "\n")

    # 保存原始 Token
    token_save_path = os.path.join(output_dir, "generated_tokens.npy")
    np.save(token_save_path, generated_sequence)

    # 8. 精准级联解码逻辑
    real_sequence = generated_sequence[generated_sequence != tokenizer.PAD_TOKEN]

    try:
        part_p_list, part_b_list, part_o_list = [], [], []
        current_stage = None

        for token in real_sequence:
            token_val = int(token)
            if token_val in [getattr(tokenizer, 'SOS_TOKEN', -1), getattr(tokenizer, 'EOS_TOKEN', -1)]:
                continue

            if token_val == tokenizer.SEP_TOKEN:
                if current_stage == 'P': part_p_list.append(token_val)
                elif current_stage == 'B': part_b_list.append(token_val)
                elif current_stage == 'O': part_o_list.append(token_val)
                continue

            if (tokenizer.PATCH_BASE <= token_val < tokenizer.BLOCK_BASE) or \
               (tokenizer.SPECIAL_PATCH_BASE <= token_val < tokenizer.VOCAB_END):
                part_p_list.append(token_val)
                current_stage = 'P'
            elif tokenizer.BLOCK_BASE <= token_val < tokenizer.OFFSET_BASE:
                part_b_list.append(token_val)
                current_stage = 'B'
            elif tokenizer.OFFSET_BASE <= token_val < tokenizer.SPECIAL_PATCH_BASE:
                part_o_list.append(token_val)
                current_stage = 'O'

        part_p = np.array(part_p_list, dtype=real_sequence.dtype)
        part_b = np.array(part_b_list, dtype=real_sequence.dtype)
        part_o = np.array(part_o_list, dtype=real_sequence.dtype)

        if len(part_p) == 0: print("⚠️ 警告：未找到有效的 Patch 数据。")
        if len(part_b) == 0: 
            print("❌ 错误：未找到有效的 Block 数据。")
            return

        # 解码
        mesh_recon, recon_tokens = tokenizer.decode_mesh(part_p, part_b, part_o)
        recon_tokens = np.array(recon_tokens)

        # ==========================================
        # 🔍 Token 类别分布与越界诊断 (仅在 VERBOSE开启 时显示)
        # ==========================================
        if VERBOSE:
            print("\n" + "="*50 + "\n 🧮 Token 类别分布诊断 \n" + "="*50)
            raw_np, gen_np = np.array(raw_tokens), recon_tokens 

            def categorize_tokens(tokens):
                return {
                    "Patch (宏观)": np.sum((tokens >= 0) & (tokens < tokenizer.BLOCK_BASE)),
                    "Block (中观)": np.sum((tokens >= tokenizer.BLOCK_BASE) & (tokens < tokenizer.OFFSET_BASE)),
                    "Offset (细节)": np.sum((tokens >= tokenizer.OFFSET_BASE) & (tokens < tokenizer.SOS_TOKEN)),
                    "[SEP] (分隔符)": np.sum(tokens == tokenizer.SEP_TOKEN),
                }

            raw_counts, gen_counts = categorize_tokens(raw_np), categorize_tokens(gen_np)
            
            print(f"{'类别':<15} | {'GT':<8} | {'Gen':<8} | {'Diff'}")
            print("-" * 45)
            for cat in raw_counts.keys():
                diff = gen_counts[cat] - raw_counts[cat]
                flag = "⚠️" if diff != 0 else "✅"
                print(f"{cat:<15} | {raw_counts[cat]:<8} | {gen_counts[cat]:<8} | {diff:<4} {flag}")

            max_valid_id = max(tokenizer.SOS_TOKEN, tokenizer.EOS_TOKEN, tokenizer.SEP_TOKEN, tokenizer.PAD_TOKEN)
            if np.sum(gen_np > max_valid_id) > 0:
                print(f"\n🚨 警告: 包含 {np.sum(gen_np > max_valid_id)} 个非法越界 Token！")
            print("="*50 + "\n")

        # 9. 保存结果
        base_name = "generated_iflame_result"
        idx = 1
        while os.path.exists(os.path.join(output_dir, f"{base_name}_{idx}.obj")): idx += 1
        
        mesh_path, pc_path = os.path.join(output_dir, f"{base_name}_{idx}.obj"), os.path.join(output_dir, f"{base_name}_{idx}_input_pc.ply")

        if isinstance(mesh_recon, trimesh.Trimesh):
            mesh_recon.export(mesh_path)
        else:
            trimesh.Trimesh(vertices=mesh_recon, faces=np.arange(len(mesh_recon)).reshape(-1, 3)).export(mesh_path)
        
        # 安全保存点云
        pc_np = test_pc.squeeze(0).detach().cpu().numpy()
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            trimesh.PointCloud(pc_np[:, :3]).export(tmp.name)
            shutil.copy2(tmp.name, pc_path) 
            os.remove(tmp.name)

        print(f"🎉 任务完成！重构 Token: {len(recon_tokens)}。3D 模型见: {mesh_path}")

    except Exception as e:
        print(f"❌ 解码阶段发生崩溃: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()