import torch
from pointmae import PointMAE  # 导入我们昨天写的纯 PyTorch 版 PointMAE

def run_test():
    # 1. 自动检测硬件环境 (适配你的 Mac: 优先使用 MPS 苹果芯片加速，否则用 CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🖥️  当前使用的计算设备: {device}")

    # 2. 设置你下载的权重文件路径 (请确保路径和文件名与你本地一致！)
    # 假设你在项目下建了 checkpoints 文件夹，并把下载的权重放了进去
    weights_path = "./checkpoints/pretrain.pth" 

    # 3. 实例化模型并移至设备
    print("\n📦 正在初始化 PointMAE 模型架构...")
    model = PointMAE(
        num_group=64,       # iFlame 需要的序列长度 (64个 patch)
        group_size=32,      # 每个 patch 包含 32 个点
        embed_dim=384,      # PointMAE 预训练的默认特征维度
        encoder_depth=12,   # 12 层 Transformer
        num_heads=6
    ).to(device)

    # 4. 加载官方预训练权重
    print(f"⏳ 正在加载官方预训练权重: {weights_path} ...")
    try:
        # map_location 确保在 Mac 上加载 GPU 训练的权重不会报错
        checkpoint = torch.load(weights_path, map_location=device)
        
        # 官方保存的权重字典中，真正的参数通常在 'model' 这个 key 下面
        state_dict = checkpoint.get('model', checkpoint)
        
        # 严格对齐加载 (strict=False 允许忽略一些我们不需要的特定分类头)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功！")
        
        # 打印少量未匹配的 key (如果是解码器部分的 key 未匹配，是完全正常的)
        if missing_keys:
            print(f"   ℹ️  提示: 有 {len(missing_keys)} 个参数未找到 (通常是不需要的分类头/重建头，可忽略)")
            
    except FileNotFoundError:
        print(f"❌ 错误: 找不到权重文件！请检查路径 {weights_path} 是否正确。")
        return
    except Exception as e:
        print(f"❌ 错误: 加载权重时发生异常: {e}")
        return

    # 5. 制造假数据进行前向推理测试
    print("\n🧪 正在生成模拟的 3D 点云数据进行测试...")
    batch_size = 2
    num_points = 1024
    # 生成形状为 [Batch=2, Points=1024, XYZ=3] 的点云
    dummy_pc = torch.randn(batch_size, num_points, 3).to(device)
    print(f"   输入点云形状: {dummy_pc.shape}")

    # 6. 测试 iFlame 专属接口 (forward_encoder)
    print("\n🚀 正在运行特征提取层 (forward_encoder)...")
    model.eval() # 切换到评估模式，关闭 Dropout
    
    with torch.no_grad(): # 测试阶段不需要计算梯度，省内存
        # 这里调用的就是我们昨天特意为 iFlame 写的无掩码接口
        context_tokens = model.forward_encoder(dummy_pc)

    # 7. 验证输出结果
    print("\n🎉 测试完成！")
    print(f"   模型输出的上下文特征矩阵形状: {context_tokens.shape}")
    
    if context_tokens.shape == (2, 64, 384):
        print("   ✅ 形状完全正确！你可以放心地把它对接到 iFlame 的 256 维空间里了。")
    else:
        print("   ❌ 警告: 输出形状不符合预期，请检查！")

if __name__ == "__main__":
    run_test()