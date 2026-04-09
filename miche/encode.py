# -*- coding: utf-8 -*-
import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
from .michelangelo.utils.misc import instantiate_from_config

def load_surface(fp):
    
    with np.load(fp) as input_pc:
        surface = input_pc['points']
        normal = input_pc['normals']
    
    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], 4096, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])
    
    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0).cuda()
    
    return surface

def reconstruction(args, model, bounds=(-1.25, -1.25, -1.25, 1.25, 1.25, 1.25), octree_depth=7, num_chunks=10000):

    surface = load_surface(args.pointcloud_path)
    # old_surface = surface.clone()

    # surface[0,:,0]*=-1
    # surface[0,:,1]*=-1
    surface[0,:,2]*=-1

    # encoding
    shape_embed, shape_latents = model.model.encode_shape_embed(surface, return_latents=True)    
    shape_zq, posterior = model.model.shape_model.encode_kl_embed(shape_latents)

    # decoding
    latents = model.model.shape_model.decode(shape_zq)
    # geometric_func = partial(model.model.shape_model.query_geometry, latents=latents)
    
    return 0


import os
# 【关键修复】：强制使用国内镜像站！必须写在最前面！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

def get_huggingface_weight(repo_id="Maikou/Michelangelo", filename="checkpoints/aligned_shape_latents/shapevae-256.ckpt"):
    """
    自动从 Hugging Face 下载权重并返回本地缓存路径。
    如果本地已有缓存，则瞬间返回路径，不会重复下载。
    """
    print(f"正在从 Hugging Face 获取: {filename} ...")
    try:
        # 这个函数会返回文件在本地的绝对路径 (通常在 ~/.cache/huggingface/hub/ 下)
        local_ckpt_path = hf_hub_download(
            repo_id=repo_id, 
            filename=filename,
            # 如果服务器在国内，可能需要配置代理或者使用国内镜像
            # endpoint="https://hf-mirror.com"  # 取消注释这行可以使用国内镜像加速下载
        )
        print(f"✅ 下载/读取成功！本地路径: {local_ckpt_path}")
        return local_ckpt_path
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def load_model(config_path="miche/shapevae-256.yaml"):
    # 1. 自动下载完整的官方权重
    ckpt_path = get_huggingface_weight(
        repo_id="Maikou/Michelangelo", 
        filename="checkpoints/aligned_shape_latents/shapevae-256.ckpt" # 换成目标文件名
    )
    
    if ckpt_path is None:
        raise RuntimeError("无法获取模型权重，请检查网络！")

    # 2. 读取网络配置
    model_config = OmegaConf.load(config_path)
    if hasattr(model_config, "model"):
        model_config = model_config.model

    # 3. 实例化空模型
    # 注意：为了让模型自动处理内部模块，这里我们直接把 ckpt_path 传进去
    # 因为你下载的是官方完整版权重，所以不需要像裁剪版那样加 strict=False 了
    print("正在将权重加载到模型中...")
    model = instantiate_from_config(model_config, ckpt_path=ckpt_path) 
    
    model = model.eval()
    print("✅ Michelangelo 完整模型加载完毕！")
    
    return model

if __name__ == "__main__":
    '''
    1. Reconstruct point cloud
    2. Image-conditioned generation
    3. Text-conditioned generation
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--pointcloud_path", type=str, default='./example_data/surface.npz', 
                        help='Path to the input point cloud')
    parser.add_argument("--image_path", type=str, help='Path to the input image')
    parser.add_argument("--text", type=str, 
                        help='Input text within a format: A 3D model of motorcar; Porsche 911.')
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()
    
    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Output directory: {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')
    
    reconstruction(args, load_model(args))
