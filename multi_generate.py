import os
import sys
import glob
import torch
import torch.multiprocessing as mp
import numpy as np
import trimesh
import shutil
import tempfile
import math
from models.transformer import *

# ⚠️ 请确保这里的路径能正确导入你的 Tokenizer
from utils.tokenizer import CascadedTokenizer 

def load_custom_point_cloud(file_path, num_points=4096, device='cuda', expected_channels=6):
    """优化后的点云加载函数 (保持原样)"""
    if file_path.endswith('.npy'):
        pc = np.load(file_path)
        if len(pc.shape) == 3:
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

    if pc.shape[1] == 3 and expected_channels == 6:
        dummy_normals = np.zeros_like(pc)
        pc = np.concatenate([pc, dummy_normals], axis=-1)
    elif pc.shape[1] > expected_channels:
        pc = pc[:, :expected_channels]

    if len(pc) > num_points:
        indices = np.random.choice(len(pc), num_points, replace=False)
        pc = pc[indices]
    elif len(pc) < num_points:
        indices = np.random.choice(len(pc), num_points, replace=True)
        pc = pc[indices]

    coords = pc[:, :3]
    center = (coords.min(axis=0) + coords.max(axis=0)) / 2.0
    coords -= center
    scale = (coords.max(axis=0) - coords.min(axis=0)).max()
    if scale > 0:
        coords /= scale
    pc[:, :3] = coords
    
    if pc.shape[1] == 6:
        norms = np.linalg.norm(pc[:, 3:], axis=1, keepdims=True)
        pc[:, 3:] = pc[:, 3:] / (norms + 1e-8)

    pc_tensor = torch.tensor(pc, dtype=torch.float32).to(device)
    if pc_tensor.dim() == 2:
        pc_tensor = pc_tensor.unsqueeze(0)
    
    return pc_tensor

def normalize_vertices(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    return vertices

def worker_fn(worker_id, gpu_id, assigned_files, num_per_obj, ckpt_path, output_dir, prompt_ratio=0):
    """
    单个 Worker 的执行逻辑：在指定的 GPU 上加载模型，并处理分配给它的文件列表
    """
    if not assigned_files:
        print(f"[Worker {worker_id}] 没有分配到任务，退出。")
        return

    device = torch.device(f"cuda:{gpu_id}")
    print(f"🤖 [Worker {worker_id}] 启动，分配到 GPU {gpu_id}，负责处理 {len(assigned_files)} 个文件。")
    
    tokenizer = CascadedTokenizer()
    model = Causal_iFlame(num_categories=tokenizer.vocab_size, embed_dim=768, num_heads=16)

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.to(device).eval()
    except Exception as e:
        print(f"❌ [Worker {worker_id}] 加载权重失败: {e}")
        return

    for file_idx, file_path in enumerate(assigned_files):
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n▶️ [Worker {worker_id}] 开始处理 ({file_idx+1}/{len(assigned_files)}): {base_filename}")
        
        # 1. 准备引导点云 (每个文件提取一次)
        try:
            test_pc = load_custom_point_cloud(file_path, num_points=4096, device=device, expected_channels=6)
        except Exception as e:
            print(f"❌ [Worker {worker_id}] 读取点云失败: {e}，跳过此文件。")
            continue

        # 2. 准备初始输入 Token (每个文件提取一次)
        try:
            prompt_mesh = trimesh.load(file_path, force='mesh', process=False)
            prompt_mesh.vertices = normalize_vertices(prompt_mesh.vertices.copy())
            
            raw_tokens = tokenizer.encode_mesh(prompt_mesh)
            split_data = tokenizer.split_tokens(raw_tokens)
            
            part_p, part_b, part_o = split_data['stage1'][:-1], split_data['stage2'][1:-1], split_data['stage3'][1:]
            sep = np.array([tokenizer.SEP_TOKEN], dtype=np.int64)
            full_seq = np.concatenate([part_p, sep, part_b, sep, part_o])
            total_len = len(full_seq)

            prefix_length = max(1, min(total_len, int(total_len * prompt_ratio)))
            initial_tokens = full_seq[:prefix_length].tolist()
            
            if len(initial_tokens) == 0 or initial_tokens[0] != tokenizer.SOS_TOKEN:
                initial_tokens = [tokenizer.SOS_TOKEN] + initial_tokens
                prefix_length = len(initial_tokens) 
                
            input_ids = torch.tensor([initial_tokens], dtype=torch.long).to(device)
        except Exception as e:
            print(f"⚠️ [Worker {worker_id}] 提取初始 Token 失败 ({e})，使用 SOS_TOKEN。")
            input_ids = torch.tensor([[tokenizer.SOS_TOKEN]], dtype=torch.long).to(device)

        # 3. 对当前文件循环生成 N 次
        for gen_idx in range(1, num_per_obj + 1):
            print(f"  ⏳ [Worker {worker_id}] 正在为 {base_filename} 生成第 {gen_idx}/{num_per_obj} 个结果...")
            
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

            # 4. 解码并保存
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

                    part_p  = np.array(part_p_list, dtype=np.int64)
                    part_b = np.array(part_b_list, dtype=np.int64)
                    part_o = np.array(part_o_list, dtype=np.int64)
        
                if len(part_b) == 0: 
                    print(f"  ❌ [Worker {worker_id}] 变体 {gen_idx} 未找到有效 Block，跳过。")
                    continue

                mesh_recon, _ = tokenizer.decode_mesh(part_p, part_b, part_o)
                
                # 命名格式: 原文件名_gen_索引.obj
                mesh_path = os.path.join(output_dir, f"{base_filename}_gen_{gen_idx}.obj")
                pc_path = os.path.join(output_dir, f"{base_filename}_input_pc.ply") # 点云只存一份就行，或者覆盖也无妨

                if isinstance(mesh_recon, trimesh.Trimesh):
                    mesh_recon.export(mesh_path)
                else:
                    trimesh.Trimesh(vertices=mesh_recon, faces=np.arange(len(mesh_recon)).reshape(-1, 3)).export(mesh_path)
                
                # 存下参考点云 (方便对比)
                if gen_idx == 1:
                    pc_np = test_pc.squeeze(0).detach().cpu().numpy()
                    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                        trimesh.PointCloud(pc_np[:, :3]).export(tmp.name)
                        shutil.copy2(tmp.name, pc_path) 
                        os.remove(tmp.name)

                print(f"  🎉 [Worker {worker_id}] {base_filename} 的第 {gen_idx} 个模型已保存至: {mesh_path}")

            except Exception as e:
                print(f"  ❌ [Worker {worker_id}] 解码阶段发生崩溃: {e}")


                
def main():
    # ==========================================
    # 🎛️ 任务配置区
    # ==========================================
    ckpt_path = "checkpoints/iFlame_best_400_without_hourglass_finetune.pth"
    input_folder = "data/preview_meshes/val"   # 文件夹路径
    output_dir = "outputs_val_finetune_for_evaluate"                 # 输出路径
    
    top_x = 200    # 批量读取前 x 个 obj
    n_per_obj = 1 # 每个 obj 生成 n 个结果

    # Worker 配置：4 个进程，分配到 2 张显卡
    worker_configs = [
        {"worker_id": 0, "gpu_id": 0},
        {"worker_id": 1, "gpu_id": 0},
        {"worker_id": 2, "gpu_id": 1},
        {"worker_id": 3, "gpu_id": 1},
    ]

    os.makedirs(output_dir, exist_ok=True)

    # 1. 查找目标文件夹下所有的 obj 文件
    all_obj_files = sorted(glob.glob(os.path.join(input_folder, "*.obj")))
    if not all_obj_files:
        print(f"❌ 错误：在 {input_folder} 中没有找到任何 .obj 文件。")
        return
    
    # 2. 截取前 x 个文件
    files_to_process = all_obj_files[:top_x]
    total_files = len(files_to_process)
    print(f"📂 找到 {len(all_obj_files)} 个文件，准备处理前 {total_files} 个。每个文件生成 {n_per_obj} 次。")

    # 3. 将任务均匀分配给所有的 Worker
    num_workers = len(worker_configs)
    chunk_size = math.ceil(total_files / num_workers)
    
    processes = []
    for i, config in enumerate(worker_configs):
        # 划分文件切片
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_files)
        assigned_files = files_to_process[start_idx:end_idx]
        
        # 启动进程
        p = mp.Process(
            target=worker_fn, 
            args=(
                config["worker_id"], 
                config["gpu_id"], 
                assigned_files, 
                n_per_obj, 
                ckpt_path, 
                output_dir,
                0 # prompt_ratio
            )
        )
        processes.append(p)
        p.start()

    # 4. 等待所有 Worker 完成任务
    for p in processes:
        p.join()
        
    print("\n✅ 所有批量生成任务已完成！")

if __name__ == "__main__":
    # PyTorch 在多显卡多进程下必须使用 spawn 模式，否则 CUDA 初始化会报错
    mp.set_start_method('spawn', force=True)
    main()