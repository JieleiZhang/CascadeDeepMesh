import numpy as np
import pickle
import torch
import trimesh
from pathlib import Path
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt  # [新增：用于绘制长度分布直方图]

# -----------------------------------------------------------------------------
# 导入你的 Tokenizer
# -----------------------------------------------------------------------------
try:
    from utils.tokenizer import CascadedTokenizer
except ImportError:
    print("❌ 错误: 无法导入 CascadedTokenizer。")
    class CascadedTokenizer: # Dummy placeholder
        SEP_TOKEN, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN = 0, 1, 2, 3
        def encode_mesh(self, m): return []
        def split_tokens(self, t): return {'stage1':[],'stage2':[],'stage3':[]}

# -----------------------------------------------------------------------------
# 基础几何预处理函数
# -----------------------------------------------------------------------------
def normalize_vertices(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    max_bound = (bounds[1] - bounds[0]).max()
    if max_bound > 0:
        vertices = vertices / max_bound
    return vertices

# -----------------------------------------------------------------------------
# 主数据处理逻辑
# -----------------------------------------------------------------------------
def run_preprocessing(input_pkl, output_dir, preview_base_dir=None, 
                      maxlen=8192, num_pc_points=1024, save_previews=False):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_previews:
        if preview_base_dir is None:
            preview_base_dir = output_dir / "previews"
        else:
            preview_base_dir = Path(preview_base_dir)
        print(f"📂 [可视化] 预览文件将保存至: {preview_base_dir}")

    tokenizer = CascadedTokenizer()

    if not os.path.exists(input_pkl):
        print(f"❌ 错误: 找不到原始 PKL 数据集文件 {input_pkl}")
        return

    print(f"⏳ 正在加载原始 PKL 数据集: {input_pkl} ...")
    with open(input_pkl, 'rb') as f:
        raw_data = pickle.load(f)

    for split in ['train', 'val']:
        print(f"\n🛠️  正在预处理 [{split.upper()}] 集合 (共 {len(raw_data[f'name_{split}'])} 个)...")
        
        if save_previews:
            split_preview_dir = preview_base_dir / split
            split_mesh_dir = split_preview_dir / "meshes"
            split_pc_dir = split_preview_dir / "pointclouds"
            split_mesh_dir.mkdir(parents=True, exist_ok=True)
            split_pc_dir.mkdir(parents=True, exist_ok=True)

        vertices_list = raw_data[f'vertices_{split}']
        faces_list = raw_data[f'faces_{split}']
        names_list = raw_data[f'name_{split}']
        num_meshes = len(names_list)

        all_pc = []
        all_input_ids = []
        all_labels = []
        
        # [新增：用于记录所有真实序列长度的容器]
        raw_seq_lengths = [] 

        for i in tqdm(range(num_meshes), desc="Processing Mesh"):
            verts = normalize_vertices(vertices_list[i])
            faces = faces_list[i]
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh_name = str(names_list[i]).replace('/', '_').replace(' ', '_')

            # --- 步骤 1：采样点云 (包含坐标与法向量) ---
            try:
                # trimesh 会返回采样点的坐标 (pc) 和它们所在面的索引 (face_indices)
                pc, face_indices = trimesh.sample.sample_surface(mesh, num_pc_points)
                
                # 根据面索引，直接从 mesh 中提取对应的表面法向量
                normals = mesh.face_normals[face_indices]
                # 强制法向量单位化
                normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
                # 处理可能的 NaN (比如退化面的法向量为 0导致除 0 错)
                normals = np.nan_to_num(normals)
                
                # 将 3维坐标 和 3维法向量 拼接在一起，形成 [N, 6] 的数组
                pc_with_normals = np.concatenate([pc, normals], axis=-1)
                
            except Exception as e:
                # 异常处理：如果采样失败，必须生成 [N, 6] 的全零数组以保持维度一致
                # print(f"⚠️ 警告: 点云采样失败，使用全零填充: {e}") 
                pc_with_normals = np.zeros((num_pc_points, 6))
                
            # 存入列表，此时数据形状为 (N, 6)
            all_pc.append(pc_with_normals.astype(np.float32))


            # --- 可视化保存 ---
            if save_previews:
                try:
                    preview_mesh_path = split_mesh_dir / f"{mesh_name}.obj"
                    if not preview_mesh_path.exists():
                        mesh.export(preview_mesh_path)
                    
                    preview_pc_path = split_pc_dir / f"{mesh_name}_pc.ply"
                    if not preview_pc_path.exists():
                        trimesh.PointCloud(vertices=pc_with_normals[:, :3]).export(preview_pc_path)
                except Exception as e:
                    print(f"⚠️  警告: 导出预览文件失败 [{split}/{mesh_name}]: {e}")

            # --- 步骤 2：级联编码 ---
            try:
                raw_tokens = tokenizer.encode_mesh(mesh)
                split_data = tokenizer.split_tokens(raw_tokens)
                
                p_seq, b_seq, o_seq = split_data['stage1'], split_data['stage2'], split_data['stage3']
                
                if len(p_seq) > 0 and len(b_seq) > 0:
                    part_p = p_seq[:-1] 
                    part_b = b_seq[1:-1] 
                    part_o = o_seq[1:] 
                    sep = np.array([tokenizer.SEP_TOKEN], dtype=np.int64)
                    full_seq = np.concatenate([part_p, sep, part_b, sep, part_o])
                else:
                    raise ValueError("Empty sequences")
            except Exception:
                full_seq = np.array([tokenizer.SOS_TOKEN, tokenizer.SEP_TOKEN, tokenizer.SEP_TOKEN, tokenizer.EOS_TOKEN], dtype=np.int64)

            # [新增：记录截断/填充前的真实长度]
            raw_seq_lengths.append(len(full_seq))

            # --- 步骤 3：截断与填充 ---
            if len(full_seq) > maxlen:
                input_ids = full_seq[:maxlen]
            else:
                input_ids = full_seq

            labels = input_ids.copy()
            pad_len = maxlen - len(input_ids)
            if pad_len > 0:
                input_ids = np.pad(input_ids, (0, pad_len), constant_values=tokenizer.PAD_TOKEN)
                labels = np.pad(labels, (0, pad_len), constant_values=-100) 
            
            all_input_ids.append(input_ids.astype(np.int64))
            all_labels.append(labels.astype(np.int64))

        # -------------------------------------------------------------------------
        # [新增：Token 长度分布统计与可视化绘制]
        # -------------------------------------------------------------------------
        lengths_arr = np.array(raw_seq_lengths)
        print(f"\n📊 === [{split.upper()}] 集合 Token 序列长度分析 ===")
        print(f"   最大长度: {lengths_arr.max()}")
        print(f"   最小长度: {lengths_arr.min()}")
        print(f"   平均长度: {lengths_arr.mean():.2f}")
        print(f"   中位数:   {np.median(lengths_arr):.0f}")
        print(f"   90% 的数据长度 <= {np.percentile(lengths_arr, 90):.0f}")
        print(f"   95% 的数据长度 <= {np.percentile(lengths_arr, 95):.0f}")
        print(f"   99% 的数据长度 <= {np.percentile(lengths_arr, 99):.0f}")
        print(f"   当前设置的 MAX_CONTEXT_LEN: {maxlen}")
        
        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.hist(lengths_arr, bins=50, color='#4C72B0', edgecolor='black', alpha=0.8)
        plt.title(f'Token Sequence Length Distribution ({split.upper()} Set)', fontsize=14)
        plt.xlabel('Sequence Length (Number of Tokens)', fontsize=12)
        plt.ylabel('Number of Meshes', fontsize=12)
        
        # 画一条红线标出当前的 maxlen，直观展示截断情况
        plt.axvline(maxlen, color='#C44E52', linestyle='dashed', linewidth=2, 
                    label=f'Current Max Len ({maxlen})')
        
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plot_path = output_dir / f"{split}_token_length_distribution.png"
        plt.savefig(plot_path, dpi=300)
        plt.close() # 关闭图表释放内存
        print(f"📈 长度分布直方图已保存至: {plot_path}\n")
        # -------------------------------------------------------------------------

        # --- 步骤 4：保存张量 ---
        print(f"💾 正在将数据转换为大型张量数组并保存到 {output_dir} ...")
        np.save(output_dir / f"{split}_pc.npy", np.stack(all_pc, axis=0))
        np.save(output_dir / f"{split}_input_ids.npy", np.stack(all_input_ids, axis=0))
        np.save(output_dir / f"{split}_labels.npy", np.stack(all_labels, axis=0))
        
        with open(output_dir / f"{split}_names.pkl", 'wb') as f_names:
            pickle.dump(names_list, f_names)

        print(f"✅ [{split.upper()}] 处理完成。Token序列形状: {np.stack(all_input_ids, axis=0).shape}")

if __name__ == "__main__":
    INPUT_PKL = 'data/processed_data_face_400.pkl' 
    OUTPUT_DIR = 'data/preprocessed_tensors_face_400' 
    
    ENABLE_PREVIEW = False 
    PREVIEW_DIR = 'data/preprocessed_previews_face_400' 

    MAX_CONTEXT_LEN = 4608 
    NUM_PC_SAMPLING = 4096

    print("🚀 === Cascaded DeepMesh 離線数据处理脚本启动 ===")
    
    try:
        run_preprocessing(
            INPUT_PKL, 
            OUTPUT_DIR, 
            preview_base_dir=PREVIEW_DIR, 
            maxlen=MAX_CONTEXT_LEN, 
            num_pc_points=NUM_PC_SAMPLING,
            save_previews=ENABLE_PREVIEW    
        )
        print("\n🎉 === 数据处理成功完成！ ===")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ === 数据处理失败: {e} ===")