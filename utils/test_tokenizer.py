import os
import glob
import numpy as np
import trimesh
from tqdm import tqdm
import traceback

# ⚠️ 确保正确导入你的 Tokenizer
from utils.tokenizer import CascadedTokenizer 

def process_and_test_mesh(obj_path, tokenizer):
    """
    处理单个 Mesh 并比对 Token。
    返回: (is_success: bool, message: str)
    """
    try:
        # [1] 加载与预处理 (复用你的逻辑)
        mesh_orig = trimesh.load(obj_path, force='mesh', process=False)
        if isinstance(mesh_orig, trimesh.Scene):
            mesh_orig = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=g.vertices, faces=g.faces) 
                for g in mesh_orig.geometry.values()
            ])
            
        mesh_orig.merge_vertices()
        mesh_orig.update_faces(mesh_orig.nondegenerate_faces())
        mesh_orig.update_faces(mesh_orig.unique_faces())
        mesh_orig.process()
        
        mesh_orig.vertices -= mesh_orig.centroid
        max_dist = np.max(np.linalg.norm(mesh_orig.vertices, axis=1))
        if max_dist > 1e-6:
            mesh_orig.apply_scale(0.95 / max_dist)

        # [2] 编码
        raw_tokens = tokenizer.encode_mesh(mesh_orig)

        # [3] 拆解为三个阶段
        data_dict = tokenizer.split_tokens(raw_tokens)
        s1 = data_dict['stage1']
        s2 = data_dict['stage2']
        s3 = data_dict['stage3']

        # [4] 解码
        mesh_recon, recon_tokens = tokenizer.decode_mesh(s1, s2, s3)

        # [5] 严格比对
        raw_np = np.array(raw_tokens)
        recon_np = np.array(recon_tokens)

        # 检查长度
        if len(raw_np) != len(recon_np):
            return False, f"长度不匹配 (原始: {len(raw_np)}, 重构: {len(recon_np)})"

        # 检查内容
        if np.array_equal(raw_np, recon_np):
            return True, "完全一致"
        else:
            # 找出不一致的索引位置
            diff_indices = np.where(raw_np != recon_np)[0]
            sample_diff = diff_indices[:5].tolist() # 只展示前 5 个差异点防止刷屏
            return False, f"内容不一致，共 {len(diff_indices)} 处差异。前几处位置: {sample_diff}"

    except Exception as e:
        # 捕获崩溃并返回简短报错信息
        error_msg = str(e)
        return False, f"处理崩溃: {error_msg}"

if __name__ == "__main__":
    
    # === 配置区域 ===
    tokenizer = CascadedTokenizer()
    # 替换为你实际存放 obj 的根目录
    dataset_dir = "data/preview_meshes/train" 
    # 限制测试数量，设为 None 则测试整个文件夹
    MAX_FILES = 500 
    # ================

    print(f"🔍 正在扫描目录: {dataset_dir}")
    # 查找所有 obj 文件
    all_obj_files = glob.glob(os.path.join(dataset_dir, "**/*.obj"), recursive=True)
    
    if len(all_obj_files) == 0:
        print("❌ 未找到任何 .obj 文件，请检查路径！")
        exit()

    if MAX_FILES is not None:
        all_obj_files = all_obj_files[:MAX_FILES]
        
    print(f"🚀 开始批量测试 {len(all_obj_files)} 个 Mesh...")

    success_count = 0
    fail_records = [] # 记录失败的文件和原因

    # 使用 tqdm 包装循环以显示进度条
    for obj_path in tqdm(all_obj_files, desc="测试进度", unit="mesh"):
        is_success, msg = process_and_test_mesh(obj_path, tokenizer)
        
        if is_success:
            success_count += 1
        else:
            filename = os.path.basename(obj_path)
            fail_records.append(f"{filename} -> {msg}")

    # === 输出统计报告 ===
    print("\n" + "="*40)
    print(" 📊 Token 一致性测试报告 ")
    print("="*40)
    print(f"总计测试: {len(all_obj_files)}")
    print(f"✅ 成功匹配: {success_count}")
    print(f"❌ 失败数量: {len(fail_records)}")
    
    success_rate = (success_count / len(all_obj_files)) * 100
    print(f"🎯 准确率:   {success_rate:.2f}%")
    print("="*40)

    # 打印具体的失败原因，方便排查
    if fail_records:
        print("\n⚠️ 失败详情 (前 20 条):")
        for record in fail_records[:20]:
            print(f" - {record}")
        
        if len(fail_records) > 20:
            print(f"   ... 及其他 {len(fail_records) - 20} 个错误。")