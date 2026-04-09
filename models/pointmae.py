import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 纯 PyTorch 基础几何操作 (FPS & KNN)
# ==========================================
def square_distance(src, dst):
    """计算两组点之间的欧氏距离的平方"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """根据索引提取点云特征"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """最远点采样 (FPS)"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn_point(nsample, xyz, new_xyz):
    """K 近邻 (KNN)"""
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

# ==========================================
# 2. PointMAE 核心模块 (分块与特征提取)
# ==========================================
class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        # xyz: [B, N, 3]
        # 1. 用 FPS 选出中心点
        center_idx = farthest_point_sample(xyz, self.num_group)
        center = index_points(xyz, center_idx) # [B, num_group, 3]
        
        # 2. 用 KNN 找出每个中心点附近的点，形成局部 Patch
        idx = knn_point(self.group_size, xyz, center) 
        neighborhood = index_points(xyz, idx) # [B, num_group, group_size, 3]
        
        # 3. 将局部点云减去中心点，得到相对坐标 (归一化)
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class MiniPointNet(nn.Module):
    """用于提取每个 Patch 特征的迷你 PointNet"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, 1)
        self.conv2 = nn.Conv1d(out_channels // 2, out_channels // 4, 1)
        self.conv3 = nn.Conv1d(out_channels // 4, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels // 2)
        self.bn2 = nn.BatchNorm1d(out_channels // 4)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: [B, C, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=-1)[0] # 最大池化获取全局特征
        return x

# ==========================================
# 3. Transformer 编码器与解码器
# ==========================================
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-Norm 更稳定
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = self.norm(x)
        return x

# ==========================================
# 4. PointMAE 完整模型
# ==========================================
class PointMAE(nn.Module):
    def __init__(self, num_group=64, group_size=32, embed_dim=384, encoder_depth=12, num_heads=6):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.embed_dim = embed_dim

        # 1. 几何分组与特征提取 (Patch Embedding)
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.patch_embedding = MiniPointNet(in_channels=3, out_channels=embed_dim)
        
        # 2. 位置编码 (根据中心点坐标生成)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        
        # 3. Transformer 编码器 (我们做条件生成主要用这个)
        self.encoder = TransformerEncoder(embed_dim=embed_dim, depth=encoder_depth, num_heads=num_heads)
        
        # 4. (预训练用) Mask Token 和解码器
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        self.decoder = TransformerEncoder(embed_dim=embed_dim, depth=4, num_heads=6)
        
        # 重建点云的输出头
        self.rebuild_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3 * group_size)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, pts):
        """
        =========================================================
        🔥 iFlame 专属接口：提取不被掩码的完整特征用于交叉注意力
        =========================================================
        """
        B, N, C = pts.shape
        # 1. 分组得到 Patch 和中心点
        neighborhood, center = self.group_divider(pts)
        
        # 2. 提取 Patch 特征 (Mini PointNet)
        # neighborhood shape: [B, num_group, group_size, 3] -> [B*num_group, 3, group_size]
        neighborhood = neighborhood.view(B * self.num_group, self.group_size, 3).permute(0, 2, 1)
        patch_features = self.patch_embedding(neighborhood) # [B*num_group, embed_dim]
        patch_features = patch_features.view(B, self.num_group, self.embed_dim) # [B, num_group, embed_dim]
        
        # 3. 加上位置编码
        pos_emb = self.pos_embedding(center) # [B, num_group, embed_dim]
        x = patch_features + pos_emb
        
        # 4. 过 Transformer 编码器
        x = self.encoder(x) # [B, num_group, embed_dim]
        
        # 返回最终的 Context Tokens 供 iFlame 使用
        return x

    def forward(self, pts, mask_ratio=0.6):
        """
        标准的 MAE 前向传播 (带掩码和重建，仅用于自监督预训练)
        """
        B, N, C = pts.shape
        neighborhood, center = self.group_divider(pts)
        
        neighborhood_flat = neighborhood.view(B * self.num_group, self.group_size, 3).permute(0, 2, 1)
        patch_features = self.patch_embedding(neighborhood_flat)
        patch_features = patch_features.view(B, self.num_group, self.embed_dim)
        pos_emb = self.pos_embedding(center)

        # 生成 Mask
        bool_masked_pos = self._random_masking(B, self.num_group, mask_ratio, pts.device)
        
        # 提取未被 Mask 的可见部分
        batch_indices = torch.arange(B).unsqueeze(-1).expand(-1, self.num_group).to(pts.device)
        visible_indices = torch.nonzero(~bool_masked_pos, as_tuple=True)[1].view(B, -1)
        
        visible_features = patch_features[batch_indices, visible_indices, :]
        visible_pos_emb = pos_emb[batch_indices, visible_indices, :]
        
        # 编码器只处理可见部分
        x = self.encoder(visible_features + visible_pos_emb)

        # 解码阶段：拼接预测的 Mask Token
        mask_tokens = self.mask_token.expand(B, self.num_group - visible_features.shape[1], -1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        
        # 对齐解码器的位置编码 (保持顺序)
        full_pos_emb = self.decoder_pos_embed(center)
        x_full = x_full + full_pos_emb
        
        # 解码并重建坐标
        x_decoded = self.decoder(x_full)
        rebuilt_points = self.rebuild_head(x_decoded) # [B, num_group, 3 * group_size]
        
        return rebuilt_points, bool_masked_pos

    def _random_masking(self, B, num_patches, mask_ratio, device):
        """随机掩码逻辑"""
        num_mask = int(mask_ratio * num_patches)
        rand_indices = torch.rand(B, num_patches, device=device).argsort(dim=-1)
        mask = torch.zeros(B, num_patches, dtype=torch.bool, device=device)
        batch_indices = torch.arange(B).unsqueeze(-1).expand(-1, num_mask).to(device)
        mask[batch_indices, rand_indices[:, :num_mask]] = True
        return mask

# ==========================================
# 测试代码 (直接运行此文件可测试)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 模拟输入一个 Batch 的点云 [BatchSize, NumPoints, XYZ]
    dummy_pc = torch.randn(2, 1024, 3).to(device)
    
    model = PointMAE().to(device)
    
    # 测试特征提取 (为 iFlame 准备)
    context_tokens = model.forward_encoder(dummy_pc)
    print(f"提取的上下文特征维度 (用于 iFlame): {context_tokens.shape}") 
    # 期望输出: [2, 64, 384] (Batch=2, 64个块, 每个块384维特征)