import numpy as np
import trimesh
from pathlib import Path
import os

# -----------------------------------------------------------------------------
# 导入你的 Tokenizer (请确保当前运行路径能找到 utils 文件夹)
# -----------------------------------------------------------------------------
try:
    from utils.tokenizer import CascadedTokenizer
except ImportError:
    print("❌ 错误: 无法导入 CascadedTokenizer。请确保路径正确且 utils/tokenizer.py 存在。")
    class CascadedTokenizer: # Dummy placeholder 仅供代码结构测试
        SEP_TOKEN, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN = 0, 1, 2, 3
        def encode_mesh(self, m): return []
        def split_tokens(self, t): return {'stage1': [4,5], 'stage2': [6,7], 'stage3': [8,9]}

# -----------------------------------------------------------------------------
# 基础几何预处理函数
# -----------------------------------------------------------------------------
def normalize_vertices(vertices):
    """标准归一化: 将 Mesh 坐标中心化并缩放到 [-0.5, 0.5] 区间"""
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    max_bound = (bounds[1] - bounds[0]).max()
    if max_bound > 0:
        vertices = vertices / max_bound
    return vertices

# -----------------------------------------------------------------------------
# 核心逻辑：处理单个 OBJ 并生成样本数据集
# -----------------------------------------------------------------------------
def generate_sample_dataset(obj_filepath, output_dir, maxlen=8192, num_pc_points=1024):
    obj_path = Path(obj_filepath)
    output_dir = Path(output_dir)
    
    if not obj_path.exists():
        print(f"❌ 错误: 找不到指定的 OBJ 文件 -> {obj_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = CascadedTokenizer()
    
    print(f"⏳ 正在读取网格文件: {obj_path.name} ...")
    
    # 1. 加载 OBJ 文件
    # force='mesh' 确保加载为单一网格，process=False 保持原始拓扑
    raw_mesh = trimesh.load(obj_path, force='mesh', process=False)
    
    # 2. 几何标准化
    verts = normalize_vertices(raw_mesh.vertices)
    mesh = trimesh.Trimesh(vertices=verts, faces=raw_mesh.faces, process=False)
    
    # 3. 采样点云
    print(f"🎯 正在采样 {num_pc_points} 个点云...")
    try:
        pc, _ = trimesh.sample.sample_surface(mesh, num_pc_points)
    except Exception as e:
        print(f"⚠️ 采样失败，使用全零点云替代: {e}")
        pc = np.zeros((num_pc_points, 3))
    pc = pc.astype(np.float32)

    # 4. Token 编码
    print(f"🧩 正在进行 Token 级联编码...")
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
            raise ValueError("编码生成的序列为空")
    except Exception as e:
        print(f"⚠️ 编码失败，使用 Dummy 序列替代: {e}")
        full_seq = np.array([tokenizer.SOS_TOKEN, tokenizer.SEP_TOKEN, tokenizer.SEP_TOKEN, tokenizer.EOS_TOKEN], dtype=np.int64)

    # 5. 截断与填充 (Padding)
    print(f"📏 正在截断/填充至最大长度 {maxlen} (当前实际长度: {len(full_seq)})...")
    if len(full_seq) > maxlen:
        input_ids = full_seq[:maxlen]
    else:
        input_ids = full_seq

    labels = input_ids.copy()
    pad_len = maxlen - len(input_ids)
    
    if pad_len > 0:
        input_ids = np.pad(input_ids, (0, pad_len), constant_values=tokenizer.PAD_TOKEN)
        labels = np.pad(labels, (0, pad_len), constant_values=-100) 
        
    input_ids = input_ids.astype(np.int64)
    labels = labels.astype(np.int64)

    # 6. 扩展维度以模拟 Batch 数据 (将形状变为 [1, ...])
    # 这一步极其重要，它让单条数据变成了 Dataset/DataLoader 能够处理的形状
    final_pc = np.expand_dims(pc, axis=0)               # Shape: (1, 1024, 3)
    final_input_ids = np.expand_dims(input_ids, axis=0) # Shape: (1, 8192)
    final_labels = np.expand_dims(labels, axis=0)       # Shape: (1, 8192)

    # 7. 保存为 NPY 数组
    print(f"\n💾 正在将样本数据集保存至: {output_dir}")
    np.save(output_dir / "sample_pc.npy", final_pc)
    np.save(output_dir / "sample_input_ids.npy", final_input_ids)
    np.save(output_dir / "sample_labels.npy", final_labels)
    
    # 顺便存一个预览用的归一化网格
    mesh.export(output_dir / "sample_normalized_preview.obj")

    print("\n✅ === 样本数据集生成完毕！ ===")
    print(f"   点云特征张量形状:     {final_pc.shape}")
    print(f"   Input IDs 张量形状: {final_input_ids.shape}")
    print(f"   Labels 张量形状:    {final_labels.shape}")


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # 👇 请在这里填入你用来测试的 OBJ 文件路径
    # -----------------------------------------------------------------
    MY_TEST_OBJ = "data/preprocessed_previews/train/meshes/03593526_6bb307429e516e2f129f1cb6c6fa9dd5_dec21.obj" 
    
    # 样本数据集的存放位置
    SAMPLE_OUTPUT_DIR = "data/sample_dataset" 
    
    # 保持和模型一致的配置
    MAX_CONTEXT_LEN = 8192 
    NUM_PC_SAMPLING = 1024

    # 运行生成
    generate_sample_dataset(
        obj_filepath=MY_TEST_OBJ,
        output_dir=SAMPLE_OUTPUT_DIR,
        maxlen=MAX_CONTEXT_LEN,
        num_pc_points=NUM_PC_SAMPLING
    )