import sys
import os
import numpy as np
import torch
import trimesh

# 获取当前脚本所在目录 (data/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (CascadeDeepMesh/)
project_root = os.path.dirname(current_dir)
# 将根目录加入 Python 搜索路径
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.serialization import serialize, deserialize

class CascadedTokenizer:
    def __init__(self):
        self.PATCH_SIZE = 4
        self.BLOCK_SIZE = 8
        self.OFFSET_SIZE = 16
        
        # --- 基础区间 ---
        self.PATCH_BASE = 0
        self.BLOCK_BASE = self.PATCH_SIZE**3          # 64
        self.OFFSET_BASE = self.BLOCK_BASE + self.BLOCK_SIZE**3  # 64+512=576
        
        # --- 特殊跳跃区间 (Start Patch) ---
        self.SPECIAL_PATCH_BASE = self.BLOCK_SIZE**3 + self.OFFSET_SIZE**3 + self.PATCH_SIZE**3 
        # 4672
        
        # --- 结束区间 ---
        self.VOCAB_END = self.SPECIAL_PATCH_BASE + self.PATCH_SIZE**3 # 4736
        
        # --- 定义 GPT 用的特殊符号 ---
        self.SOS_TOKEN = self.VOCAB_END + 0
        self.EOS_TOKEN = self.VOCAB_END + 1
        self.SEP_TOKEN = self.VOCAB_END + 2
        self.PAD_TOKEN = self.VOCAB_END + 3
        
        self.vocab_size = self.PAD_TOKEN + 1

    def split_tokens(self, raw_tokens):
        patch_seq = []
        block_seq = []
        offset_seq = []

        last_token_type = None # 'P', 'B', 'O'

        for token in raw_tokens:
            token = int(token)
            
            # --- Patch ---
            if (self.PATCH_BASE <= token < self.BLOCK_BASE) or \
               (self.SPECIAL_PATCH_BASE <= token < self.VOCAB_END):
                patch_seq.append(token)
                
                # [关键修复] 当从 Patch 切换时...
                
                # 1. 如果上一个是 Offset，说明"上一个 Block"彻底结束了，
                #    必须给 Offset 序列也加个分隔符！
                #    (之前漏了这一行，导致跨 Patch 的 Offset 粘在了一起)
                if last_token_type == 'O':
                    offset_seq.append(self.SEP_TOKEN)

                # 2. 切分 Block 序列 (原有的逻辑)
                if last_token_type in ['B', 'O']:
                     block_seq.append(self.SEP_TOKEN)
                     
                last_token_type = 'P'

            # --- Block ---
            elif self.BLOCK_BASE <= token < self.OFFSET_BASE:
                block_seq.append(token)
                
                # 当从 Offset 切换回 Block 时，切分 Offset 序列
                if last_token_type in ['B', 'O']:
                    offset_seq.append(self.SEP_TOKEN)
                last_token_type = 'B'

            # --- Offset ---
            elif self.OFFSET_BASE <= token < self.SPECIAL_PATCH_BASE:
                offset_seq.append(token)
                last_token_type = 'O'

        # 添加 SOS/EOS
        patch_seq = [self.SOS_TOKEN] + patch_seq + [self.EOS_TOKEN]
        block_seq = [self.SOS_TOKEN] + block_seq + [self.EOS_TOKEN]
        offset_seq = [self.SOS_TOKEN] + offset_seq + [self.EOS_TOKEN]

        return {
            "stage1": np.array(patch_seq, dtype=np.int64),
            "stage2": np.array(block_seq, dtype=np.int64),
            "stage3": np.array(offset_seq, dtype=np.int64)
        }

    def encode_mesh(self, mesh_obj):
        """输入: trimesh 对象 -> 输出: 原版 DeepMesh 序列"""
        raw_tokens = serialize(mesh_obj)
        return raw_tokens
    
    def _split_sequence(self, sequence, sep_token):
        """[辅助函数] 将 flat list 按 sep_token 切分"""
        groups = []
        current_group = []
        for token in sequence:
            if token == sep_token:
                groups.append(current_group)
                current_group = []
            elif token in [self.SOS_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN]:
                continue 
            else:
                current_group.append(token)
        
        if current_group:
            groups.append(current_group)
        elif len(groups) == 0 and len(sequence) > 0: 
            groups.append([t for t in sequence if t < self.SOS_TOKEN])
            
        return groups

    def decode_mesh(self, patch_seq, block_seq, offset_seq):
        """
        [推理核心函数] 
        返回: (trimesh对象, 重组后的序列numpy数组)
        """
        # 1. 预处理 Patch (注意: 使用 SOS_TOKEN 作为阈值，保留 Restart Patch)
        patches = [t for t in patch_seq if t < self.SOS_TOKEN]

        # 2. 预处理 Block
        blocks_grouped = self._split_sequence(block_seq, self.SEP_TOKEN)

        # 3. 预处理 Offset
        offsets_grouped = self._split_sequence(offset_seq, self.SEP_TOKEN)

        # 4. 重新组装
        flat_sequence = []
        global_block_idx = 0 

        for i, p_token in enumerate(patches):
            if p_token >= self.SOS_TOKEN: continue
            flat_sequence.append(p_token)
            
            if i < len(blocks_grouped):
                current_blocks = blocks_grouped[i]
            else:
                current_blocks = [] 

            for b_token in current_blocks:
                if b_token >= self.SOS_TOKEN: continue
                flat_sequence.append(b_token)
                
                if global_block_idx < len(offsets_grouped):
                    current_offsets = offsets_grouped[global_block_idx]
                else:
                    current_offsets = [] 
                
                for o_token in current_offsets:
                    if o_token >= self.SOS_TOKEN: continue
                    flat_sequence.append(o_token)
                
                global_block_idx += 1

        print(f"重组后序列长度: {len(flat_sequence)}")
        
        # 记录重组后的 Token 序列以便对比
        recon_tokens_array = np.array(flat_sequence, dtype=np.int64)

        try:
            # step A: 获取顶点场
            vertices_array = deserialize(recon_tokens_array.copy())
            
            # step B: 构造 faces 索引
            num_vertices = len(vertices_array)
            if num_vertices % 3 != 0:
                vertices_array = vertices_array[:num_vertices - (num_vertices % 3)]
                num_vertices = len(vertices_array)
            
            num_faces = num_vertices // 3
            faces_indices = np.arange(num_vertices).reshape(num_faces, 3)
            
            # step C: 封装对象
            mesh = trimesh.Trimesh(vertices=vertices_array, faces=faces_indices, process=False)
            
            # === [核心修复] ===
            
            # 1. 强力合并顶点 (Aggressive Merge)
            # digits=4 表示降低精度要求，强制把靠得很近的点捏在一起
            # 这样能有效修复量化带来的"裂缝"
            # 1. 手动对顶点坐标进行舍入 (保留4位小数)
            # 这相当于 digits=4 的吸附效果，能让微小错位的点坐标变得完全一致
            mesh.vertices = np.round(mesh.vertices, 4)

            # 2. 然后调用合并 (此时不需要 digits 参数)
            # 因为坐标已经完全一致了，merge_vertices 默认的精确匹配就能把它们焊死
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            
            # 2. 清理重复面
            mesh.update_faces(mesh.unique_faces())
            
            # 3. 清理退化面 (面积为0或针尖面)
            mesh.update_faces(mesh.nondegenerate_faces())
            
            # 4. 清理孤立顶点
            mesh.remove_unreferenced_vertices()

            # 5. 修复法线 (让可视化更正常)
            mesh.fix_normals()
            
            return mesh, recon_tokens_array
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Mesh 解码失败: {e}")
            return None, None

# --- 测试代码 ---
# ====================== 【测试主函数：新增 删除token 测试】 ======================
if __name__ == "__main__":
    
    tokenizer = CascadedTokenizer()
    
    obj_path = "data/preprocessed_previews_2000/train/meshes/02747177_112120e83ef41de3571c83071e95ca03_dec21.obj"
    
    # [1] 加载 Mesh
    if not os.path.exists(obj_path):
        print(f"❌ 找不到 {obj_path}，使用球体")
        mesh_orig = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    else:
        print(f"📂 加载 {obj_path}")
        mesh_orig = trimesh.load(obj_path, force='mesh', process=False)
        if isinstance(mesh_orig, trimesh.Scene):
            mesh_orig = trimesh.util.concatenate([trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in mesh_orig.geometry.values()])
        mesh_orig.merge_vertices()
        # 修复：兼容旧版 trimesh
        mesh_orig.update_faces(mesh_orig.nondegenerate_faces())
        mesh_orig.update_faces(mesh_orig.unique_faces())
        mesh_orig.process()
        mesh_orig.vertices -= mesh_orig.centroid
        max_dist = np.max(np.linalg.norm(mesh_orig.vertices, axis=1))
        if max_dist > 1e-6:
            mesh_orig.apply_scale(0.95 / max_dist)

    print(f"[原始Mesh] {len(mesh_orig.vertices)} 顶点, {len(mesh_orig.faces)} 面")

    # [2] 编码
    raw_tokens = tokenizer.encode_mesh(mesh_orig)
    print(f"[原始序列长度] {len(raw_tokens)}")

    # [3] 拆解为三个阶段
    data_dict = tokenizer.split_tokens(raw_tokens)
    s1 = data_dict['stage1']
    s2 = data_dict['stage2']
    s3 = data_dict['stage3']

    print(f"[Stage1] {s1.shape}")
    print(f"[Stage2] {s2.shape}")
    print(f"[Stage3] {s3.shape}")

    mesh_recon, recon_tokens = tokenizer.decode_mesh(s1, s2, s3)
    if mesh_recon:
        mesh_recon.export("debug_orig.obj")
        print(f"✅ 原版导出: debug_orig.obj")

    # =========================================================================
    # 🧪 测试1：删除 Stage1（Patch）中间 1 个 token
    # =========================================================================
    print("\n===== [2/4] 测试：删除 Stage1 一个Token =====")
    s1_corrupt = np.delete(s1, len(s1)//2)  # 删掉中间一个
    m1, t1 = tokenizer.decode_mesh(s1_corrupt, s2, s3)
    if m1:
        m1.export("debug_corrupt_stage1.obj")
        print(f"✅ 损坏版(删Stage1)导出: debug_corrupt_stage1.obj")

    # =========================================================================
    # 🧪 测试2：删除 Stage2（Block）中间 1 个 token
    # =========================================================================
    print("\n===== [3/4] 测试：删除 Stage2 一个Token =====")
    s2_corrupt = np.delete(s2, len(s2)//2)
    m2, t2 = tokenizer.decode_mesh(s1, s2_corrupt, s3)
    if m2:
        m2.export("debug_corrupt_stage2.obj")
        print(f"✅ 损坏版(删Stage2)导出: debug_corrupt_stage2.obj")

    # =========================================================================
    # 🧪 测试3：删除 Stage3（Offset）中间 1 个 token
    # =========================================================================
    print("\n===== [4/4] 测试：删除 Stage3 一个Token =====")
    s3_corrupt = np.delete(s3, len(s3)//2)
    m3, t3 = tokenizer.decode_mesh(s1, s2, s3_corrupt)
    if m3:
        m3.export("debug_corrupt_stage3.obj")
        print(f"✅ 损坏版(删Stage3)导出: debug_corrupt_stage3.obj")

    # =========================================================================
    # 最终对比
    # =========================================================================
    print("\n" + "="*60)
    print("✅ 全部测试完成！导出文件：")
    print(" 1. debug_orig.obj                - 原版")
    print(" 2. debug_corrupt_stage1.obj      - 删除1个Patch Token")
    print(" 3. debug_corrupt_stage2.obj      - 删除1个Block Token")
    print(" 4. debug_corrupt_stage3.obj      - 删除1个Offset Token")
    print("="*60)