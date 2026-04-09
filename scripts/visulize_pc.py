import numpy as np
import trimesh
import os

# 1. 加载点云
file_path = 'data/preprocessed_tensors_1_from1000/val_pc.npy' # 替换为你的真实路径
print(f"📥 正在读取: {file_path}")
pc = np.load(file_path)

# 2. 自动处理形状问题 (兼容你之前说过的 (1, 1024, 3) 形状)
if len(pc.shape) == 3 and pc.shape[0] == 1:
    pc = pc.squeeze(0)
    print(f"   [处理] 已降维，当前形状: {pc.shape}")

# 3. 使用 trimesh 将点云导出为文件
pcd = trimesh.PointCloud(pc)
output_name = 'val_pc_visual_1000.ply' # 也可以存为 .obj
pcd.export(output_name)

print(f"🎉 大功告成！点云已导出为 {output_name}")
print(f"👉 请使用 VSCode 侧边栏或其他工具，将该文件下载到本地，用系统自带的 3D 查看器即可打开！")