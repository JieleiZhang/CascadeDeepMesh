import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import networkx as nx
import json
import trimesh
from utils.tokenizer import CascadedTokenizer



def get_shifted_sequence(sequence):
    non_special = np.flatnonzero(np.isin(sequence, [0, 1, 2], invert=True))
    if non_special.shape[0] > 0:
        idx = non_special[0]
        val = sequence[idx]
        sequence[non_special] -= (val - 3)
    return sequence


def read_faces(text):
    all_lines = text.splitlines()
    all_face_lines = [x for x in all_lines if x.startswith('f ')]
    all_faces = [[int(y.split('/')[0]) - 1 for y in x.strip().split(' ')[1:]] for x in all_face_lines]
    return all_faces


def read_vertices(text):
    all_lines = text.splitlines()
    all_vertex_lines = [x for x in all_lines if x.startswith('v ')]
    all_vertices = np.array([[float(y) for y in x.strip().split(' ')[1:]] for x in all_vertex_lines])
    assert all_vertices.shape[1] == 3, 'vertices should have 3 coordinates'
    return all_vertices


def quantize_coordinates(coords, num_tokens=256):
    if torch.is_tensor(coords):
        coords = torch.clip((coords + 0.5), 0, 1) * num_tokens  # type: ignore
        coords_quantized = coords.round().long()
    else:
        coords = np.clip((coords + 0.5), 0, 1) * num_tokens  # type: ignore
        coords_quantized = coords.round().astype(int)
    return coords_quantized


def face_to_cycles(face):
    """Find cycles in face."""
    g = nx.Graph()
    for v in range(len(face) - 1):
        g.add_edge(face[v], face[v + 1])
    g.add_edge(face[-1], face[0])
    return list(nx.cycle_basis(g))


def sort_vertices_and_faces(vertices_, faces_, num_tokens=256):
    vertices = np.clip((vertices_ + 0.5), 0, 1) * num_tokens  # type: ignore
    vertices_quantized_ = vertices.round().astype(int)

    vertices_quantized_ = vertices_quantized_[:, [2, 0, 1]]
    vertices_quantized, unique_inverse = np.unique(vertices_quantized_, axis=0, return_inverse=True)

    sort_inds = np.lexsort(vertices_quantized.T)

    vertices_quantized = vertices_quantized[sort_inds]
    vertices_quantized = np.stack([vertices_quantized[:, 2], vertices_quantized[:, 1], vertices_quantized[:, 0]], axis=-1)

    # Re-index faces and tris to re-ordered vertices.
    faces = [np.argsort(sort_inds)[unique_inverse[f]] for f in faces_]
    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(c)
                # Cyclically permute faces just that first index is the smallest.
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices_quantized.shape[0]
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1)
    vertices_quantized = vertices_quantized[vert_connected]
    # Re-index faces and tris to re-ordered vertices.
    vert_indices = (
            np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
    faces = [vert_indices[f].tolist() for f in faces]
    vertices = vertices_quantized / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    return vertices, faces

def scale_vertices(vertices, x_lims=(0.75, 1.25), y_lims=(0.75, 1.25), z_lims=(0.75, 1.25)):
    # scale x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    vertices = np.stack([vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1)
    return vertices


def shift_vertices(vertices, x_lims=(-0.1, 0.1), y_lims=(-0.1, 0.1), z_lims=(-0.075, 0.075)):
    # shift x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    # y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    # z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    x = max(min(x, 0.5 - vertices[:, 0].max()), -0.5 - vertices[:, 0].min())
    # y = max(min(y, 0.5 - vertices[:, 1].max()), -0.5 - vertices[:, 1].min())
    # z = max(min(z, 0.5 - vertices[:, 2].max()), -0.5 - vertices[:, 2].min())
    vertices = np.stack([vertices[:, 0] + x, vertices[:, 1], vertices[:, 2]], axis=-1)
    return vertices


def normalize_vertices(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    return vertices


def create_feature_stack(vertices, faces, num_tokens):
    vertices, faces = sort_vertices_and_faces(vertices, faces, num_tokens)
    # need more features: positions, angles, area, cross_product
    triangles = vertices[faces, :]
    triangles= create_feature_stack_from_triangles(triangles)
    return triangles, vertices, faces


def create_feature_stack_from_triangles(triangles):
    # t_areas = area(triangles) * 1e3
    # t_angles = angle(triangles) / float(np.pi)
    # t_normals = unit_vector(normal(triangles))
    return triangles.reshape(-1, 9)#, t_normals.reshape(-1, 3), t_areas.reshape(-1, 1), t_angles.reshape(-1, 3)

class TrianglesDataset(Dataset):

    def __init__(self, dataset_path='processed_data.pkl', split='train', scale_augment=True, shift_augment=True,
                 overfit=False, num_tokens=256,category_prefix=None,allmaxlen=0):
    
        self.dataset_path = Path(dataset_path)
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment
        self.num_tokens = num_tokens
        self.overfit = overfit

        self.cached_vertices = []
        self.cached_faces = []
        self.names = []
        self.split=split
        self.category_prefix=category_prefix

        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)

            if self.category_prefix:
                filtered_names = []
                filtered_vertices = []
                filtered_faces = []
                for idx, name in enumerate(data[f'name_{self.split}']):
                    if name.startswith(self.category_prefix):
                        filtered_names.append(name)
                        filtered_vertices.append(data[f'vertices_{self.split}'][idx])
                        filtered_faces.append(data[f'faces_{self.split}'][idx])
                self.names = filtered_names
                self.cached_vertices = filtered_vertices
                self.cached_faces = filtered_faces
            else:
                # Load all categories if no specific prefix is provided
                self.names = data[f'name_{self.split}']
                self.cached_vertices = data[f'vertices_{self.split}']
                self.cached_faces = data[f'faces_{self.split}']

            # Handling overfitting scenario
            if overfit:
                multiplier = 1 if self.split == 'val' else 500
                self.names = data['name_train'][:1] * multiplier
                self.cached_vertices = data['vertices_train'][:1] * multiplier
                self.cached_faces = data['faces_train'][:1] * multiplier
            
        print(f"{len(self.cached_vertices)} meshes loaded. 767faces")


        self.maxlen=767+1+1
        self.allmaxlen=0#allmaxlen-self.maxlen
    def __len__(self):
        return len(self.names)

    def __getitem__(self, mesh_idx):

        vertices = self.cached_vertices[mesh_idx]
        faces = self.cached_faces[mesh_idx]
        vertices = normalize_vertices(vertices)
        # 每次 __getitem__ 时实时进行数据增强：
        if self.scale_augment:
            vertices = scale_vertices(vertices)
        vertices = normalize_vertices(vertices)
        if self.shift_augment:
            vertices = shift_vertices(vertices)
        

        # ---------------------------------------------------
        # 修改 1：精准捕获清洗后的顶点和面片
        # ---------------------------------------------------
        # 原代码：triangle_feature, *_ = create_feature_stack(...)
        # 新代码：
        triangle_feature, cleaned_vertices, cleaned_faces = create_feature_stack(vertices, faces, self.num_tokens)

        # ===================================================
        # 🔥 新增模块：使用清洗后的几何数据动态生成 1024 个点云
        # ===================================================
        # process=False 防止 trimesh 再次自作主张修改我们的顶点顺序
        mesh_for_pc = trimesh.Trimesh(vertices=cleaned_vertices, faces=cleaned_faces, process=False)
        
        # 在连续的表面上按面积均匀采样 1024 个点
        point_cloud, _ = trimesh.sample.sample_surface(mesh_for_pc, 1024)
        
        # 转换为 PyTorch 张量，指定 float32 极其重要，防止后续网络数据类型报错
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        # ===================================================

        mesh_data = quantize_coordinates(torch.tensor(triangle_feature),self.num_tokens)#[:9]
        current_length = mesh_data.shape[0]
    
        # Calculate how many rows to pad
        padding_needed = self.maxlen - current_length-2
        padding = np.ones((1, mesh_data.shape[1])) *(self.num_tokens +2) # pad with zeros, shape (padding_needed, 9)
        padding1 = np.ones((1, mesh_data.shape[1])) *(self.num_tokens +3)
        mesh_data = np.vstack([padding,mesh_data, padding1])  # Stack original data with padding
        if padding_needed > 0:
            # Pad with rows of zeros (or any other value you want)
            padding = np.ones((padding_needed, mesh_data.shape[1])) *(self.num_tokens +1) # pad with zeros, shape (padding_needed, 9)
            mesh_data = np.vstack([mesh_data, padding])  # Stack original data with padding
        if self.allmaxlen:
            padding = np.ones(( self.allmaxlen, mesh_data.shape[1])) *(1) # pad with zeros, shape (padding_needed, 9)
            mesh_data = np.vstack([mesh_data, padding])  # Stack original data with padding
        # ---------------------------------------------------
        # 修改 2：返回成对的数据
        # ---------------------------------------------------
        # 原代码：return mesh_data.reshape(-1).astype("int"), 0
        # 新代码：将 0 替换为 point_cloud
        return mesh_data.reshape(-1).astype("int"), point_cloud


class NpyTensorDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # ⏳ 真正的魔法在这里：使用 np.load 并开启 mmap_mode (Memory Mapping)
        # 这不会把文件全部读入内存，而是把文件映射到虚拟内存中，根据 GPU 需要现场行切片读取。
        # 速度极快，且 RAM 占用极低（多卡 DDP 时必备）
        print(f"⏳ 正在加载预处理完成的张量数组 {self.data_dir} [{split.upper()}]...")
        
        self.input_ids = np.load(self.data_dir / f"{split}_input_ids.npy", mmap_mode='r')
        self.labels = np.load(self.data_dir / f"{split}_labels.npy", mmap_mode='r')
        self.point_clouds = np.load(self.data_dir / f"{split}_pc.npy", mmap_mode='r')

        self.length = self.input_ids.shape[0]
        print(f"✅ 加载完成。共 {self.length} 个 Mesh。Token维度: {self.input_ids.shape[1]}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # idx -> 现场行切片读取，几乎零 CPU 瓶颈
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "point_cloud": torch.tensor(self.point_clouds[idx], dtype=torch.float32)
        }


# -----------------------------------------------------------------------------
# 导入你的 Tokenizer
# -----------------------------------------------------------------------------
try:
    from utils.tokenizer import CascadedTokenizer
except ImportError:
    pass

# -----------------------------------------------------------------------------
# 带 Bound 控制的归一化函数
# -----------------------------------------------------------------------------
def normalize_mesh_with_bound(vertices, bound=0.95):
    """
    将顶点移动到原点，并根据 bound 缩放。
    如果 bound=1.0，最长边将被缩放至 1.0 (即范围在 -0.5 到 0.5 之间)
    """
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    vertices = vertices - center
    
    scale = np.max(vmax - vmin)
    if scale > 0:
        # scale 会把最大跨度变成 1，然后乘以 bound
        vertices = vertices * (bound / scale)
        
    return vertices

# -----------------------------------------------------------------------------
# 动态加载 & 增强 Dataset
# -----------------------------------------------------------------------------
def normalize_vertices(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    return vertices

def scale_vertices(vertices, x_lims=(0.75, 1.25), y_lims=(0.75, 1.25), z_lims=(0.75, 1.25)):
    # scale x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    vertices = np.stack([vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1)
    return vertices

def shift_vertices(vertices, x_lims=(-0.1, 0.1), y_lims=(-0.1, 0.1), z_lims=(-0.075, 0.075)):
    # shift x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    # 限制位移，防止越界 [-0.5, 0.5]
    x = max(min(x, 0.5 - vertices[:, 0].max()), -0.5 - vertices[:, 0].min())
    y = max(min(y, 0.5 - vertices[:, 1].max()), -0.5 - vertices[:, 1].min())
    z = max(min(z, 0.5 - vertices[:, 2].max()), -0.5 - vertices[:, 2].min())
    vertices = np.stack([vertices[:, 0] + x, vertices[:, 1] + y, vertices[:, 2] + z], axis=-1)
    return vertices

# ==========================================
# 🚀 更新后的 DataLoader
# ==========================================
class DynamicAugmentDataset(Dataset):
    def __init__(self, input_pkl, split='train', maxlen=4608, num_pc_points=4096, scale_augment=True, shift_augment=False):
        self.split = split
        self.training = (split == 'train')
        self.maxlen = maxlen
        self.num_pc_points = num_pc_points
        self.tokenizer = CascadedTokenizer()
        
        # 新增增强控制参数
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment

        print(f"⏳ 正在加载原始 PKL 数据集 [{split.upper()}] 到内存...")
        with open(input_pkl, 'rb') as f:
            raw_data = pickle.load(f)
            
        self.vertices_list = raw_data[f'vertices_{split}']
        self.faces_list = raw_data[f'faces_{split}']
        self.names_list = raw_data[f'name_{split}']
        
        self.length = len(self.names_list)
        print(f"✅ [{split.upper()}] 加载完成。共 {self.length} 个 Mesh。")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        iter_cnt = 0
        
        while True:
            iter_cnt += 1
            try:
                # 1. 获取原始数据
                v = self.vertices_list[idx].copy()
                f = self.faces_list[idx]
                
                # =========================================================
                # 2. 🔄 核心增强逻辑 (对齐 TrianglesDataset)
                # =========================================================
                # 无论是否训练，首先进行基准归一化
                v = normalize_vertices(v)
                
                # 训练时且在容错尝试的前两轮内，应用数据增强
                if self.training and iter_cnt <= 2:
                    if self.scale_augment:
                        v = scale_vertices(v)
                    
                    # 缩放后再次归一化，保证特征尺度一致
                    v = normalize_vertices(v)
                    

                else:
                    # 如果不是训练模式，或者由于严重错误退化到了 iter_cnt > 2，则仅保持归一化
                    pass 
                # =========================================================

                # 3. ☁️ 点云采样与法线计算
                mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
                pc, face_indices = trimesh.sample.sample_surface(mesh, self.num_pc_points)
                normals = mesh.face_normals[face_indices]
                normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
                normals = np.nan_to_num(normals)
                
                # 点云坐标抖动 (Jitter) - 保留原逻辑，防止点云过拟合
                # if self.training:
                #     jitter = np.random.normal(0, 0.002, size=pc.shape)
                #     pc = pc + jitter
                    
                pc_with_normals = np.concatenate([pc, normals], axis=-1).astype(np.float32)

                # 4. 🧩 Tokenize 编码
                raw_tokens = self.tokenizer.encode_mesh(mesh)
                split_data = self.tokenizer.split_tokens(raw_tokens)
                p_seq, b_seq, o_seq = split_data['stage1'], split_data['stage2'], split_data['stage3']
                
                if len(p_seq) == 0 or len(b_seq) == 0:
                    raise ValueError("生成的序列为空 (Mesh可能存在严重问题)")
                    
                part_p = p_seq[:-1] 
                part_b = b_seq[1:-1] 
                part_o = o_seq[1:] 
                sep = np.array([self.tokenizer.SEP_TOKEN], dtype=np.int64)
                full_seq = np.concatenate([part_p, sep, part_b, sep, part_o])

                if len(full_seq) > self.maxlen:
                    raise ValueError(f"Token长度 ({len(full_seq)}) 超出限制 ({self.maxlen})")

                # 5. 填充 (Padding)
                input_ids = full_seq.copy()
                labels = input_ids.copy()
                pad_len = self.maxlen - len(input_ids)
                if pad_len > 0:
                    input_ids = np.pad(input_ids, (0, pad_len), constant_values=self.tokenizer.PAD_TOKEN)
                    labels = np.pad(labels, (0, pad_len), constant_values=-100)

                # ✅ 全部成功，跳出 while 循环，返回数据
                break 

            except Exception as e:
                # 容错重试
                idx = np.random.randint(0, self.length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "point_cloud": torch.tensor(pc_with_normals, dtype=torch.float32)
        }