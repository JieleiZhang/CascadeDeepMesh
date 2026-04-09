from models.multihead_flashdiff_2 import MultiheadFlashrope,MultiheadFlashlinearrope
from models.pointmae import PointMAE # 导入你刚刚跑通的模块
from models.rms_norm import RMSNorm
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import tqdm as tqdm1
import timm
import random  


from torch.optim.lr_scheduler import LambdaLR    
import subprocess
import math
from einops import rearrange, repeat, pack
from miche.michelangelo.models.tsal.sal_perceiver import AlignedShapeLatentPerceiver


class MichelangeloAdapter(nn.Module):
    def __init__(self, weights_path, iflame_dim=256, freeze=True):
        super().__init__()
        
        # 1. 严格按照 shapevae-256.yaml 实例化 Perceiver
        self.encoder = AlignedShapeLatentPerceiver(
            device=None, 
            dtype=torch.float32,
            num_latents=256, 
            embed_dim=64,       # <-- 关键：这里不是0，是64
            point_feats=3,      # <-- 假设输入是 [B, N, 6] (带有法向量)。如果只有坐标，需改为0
            num_freqs=8,
            include_pi=False,
            heads=12,
            width=768,          # miche_dim
            num_encoder_layers=8,
            num_decoder_layers=16,
            use_ln_post=True,
            init_scale=0.25,
            qkv_bias=False,
            use_checkpoint=True # 如果显存吃紧，这个必须是 True
        )
        
        # 2. 加载你之前用 extract_encoder.py 抠出来的纯净版权重
        print(f"Loading Michelangelo encoder from {weights_path}...")
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            # 因为提取的是局部权重，用 strict=False 防止报缺少 Decoder 参数的错
            self.encoder.load_state_dict(state_dict, strict=False)
            print("Michelangelo encoder loaded successfully!")
        except Exception as e:
            print(f"Error loading weights: {e}")
        
        # 3. 冻结权重 (强烈建议在跑通前保持冻结)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.freeze = True
        else:
            self.freeze = False
            
        # 4. 维度对齐层：把 768 维降到你 iFlame 的维度 (比如 256)
        miche_dim = 768
        self.cond_head_proj = nn.Linear(miche_dim, iflame_dim)
        self.cond_proj = nn.Linear(miche_dim, iflame_dim)

    def forward(self, pc):
        """
        输入: pc [B, N, 6] (带有法向量的归一化点云) 
              如果 point_feats=0，输入则是 [B, N, 3]
        输出: context [B, 257, iflame_dim]
        """
        # 注意：如果 freeze=True，最好加上 no_grad 节省显存和计算时间
        with torch.set_grad_enabled(not self.freeze):
            
            # 【核心逻辑】：如何处理 point_feats=3
            # Michelangelo 原版设计是需要法向量(Normals)的。
            # 你的输入 pc 形状应该是 [B, N, 6] (前3维坐标，后3维法向量)
            if pc.shape[-1] == 6:
                coords = pc[..., :3]
                feats = pc[..., 3:]
            elif pc.shape[-1] == 3:
                # 如果你的预处理只存了坐标，没有法向量，那只能传 None
                # 注意：如果 yaml 里写了 point_feats=3 但 feats 传了 None
                # 可能会在底层 input_proj(data) 拼接时报维度不匹配的错！
                coords = pc
                feats = torch.zeros_like(pc) # 用全零伪造法向量特征以通过维度检查
            else:
                raise ValueError(f"Invalid point cloud feature dimension: {pc.shape[-1]}")

            # 提取特征：shape_embed 是第0个Token，latents 是后面256个
            shape_embed, latents = self.encoder.encode_latents(coords, feats)
            
        # 分别映射维度
        pc_embed_head = self.cond_head_proj(shape_embed).unsqueeze(1) # [B, 1, iflame_dim]
        pc_embed_tail = self.cond_proj(latents)                       # [B, 256, iflame_dim]
        
        # 拼起来，总长度 257
        context = torch.cat([pc_embed_head, pc_embed_tail], dim=1)    # [B, 257, iflame_dim]
        
        return context
    

class PointMAEAdapter(nn.Module):
    def __init__(self, weights_path="./checkpoints/pretrain.pth", mae_dim=384, iflame_dim=256):
        super().__init__()
        # 1. 实例化核心大脑
        self.mae = PointMAE(
            num_group=64, group_size=32, embed_dim=mae_dim, 
            encoder_depth=12, num_heads=6
        )
        
        # 2. 自动加载权重
        if weights_path:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
            state_dict = checkpoint.get('model', checkpoint)
            self.mae.load_state_dict(state_dict, strict=False)
            print(f"✅ PointMAE 权重就绪: {weights_path}")
            
        # 3. 极其关键的降维层 (384 -> 256)
        self.proj = nn.Linear(mae_dim, iflame_dim)
        
        # 4. 特征对齐：使用与 iFlame 相同的 RMSNorm 稳定梯度
        self.norm = RMSNorm(iflame_dim, eps=1e-5)

    def forward(self, pc):
        # pc: [B, N, 3]
        features = self.mae.forward_encoder(pc) # 得到 [B, 64, 384]
        context = self.proj(features)           # 降维到 [B, 64, 256]
        context = self.norm(context)            # 归一化对齐
        return context
    

class SwiGLU(nn.Module):
    def __init__(self, embed_dim):
        super(SwiGLU, self).__init__()
        self.proj1 = nn.Linear(embed_dim, int(8 / 3 * embed_dim))  # XWG
        self.proj2 = nn.Linear(embed_dim, int(8 / 3 * embed_dim))  # XW1
        self.proj3 = nn.Linear(int(8 / 3 * embed_dim), embed_dim)  # W2

    def forward(self, x):
        # Apply Swish (SiLU in PyTorch) to the first projection
        x_proj1 = F.silu(self.proj1(x))  # Swish(XWG)
        
        # Apply the second projection
        x_proj2 = self.proj2(x)  # XW1
        
        # Element-wise multiplication
        x_glu = x_proj1 * x_proj2  # (swish(XWG) ⊙ XW1)
        
        # Final projection back to the original dimension
        output = self.proj3(x_glu)  # (swish(XWG) ⊙ XW1)W2
        
        return output
class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed

def shift_sequence(x, shift_amount):

    if shift_amount == 0:
        return x
    else:
        shifted_x = torch.zeros_like(x)
        if shift_amount < x.size(1):
            shifted_x[:, shift_amount:, :] = x[:, :-shift_amount, :]

        return shifted_x


def pad_to_multiple(tensor, multiple, dim=-1, pad_value=0):

    current_size = tensor.size(dim)
    

    remainder = current_size % multiple
    if remainder == 0:
        return tensor  
    
    pad_size = multiple - remainder


    pad = [0] * (2 * tensor.dim())

    pad_start = (tensor.dim() - 1 - dim) * 2
    pad[pad_start + 1] = pad_size 


    padded_tensor = F.pad(tensor, pad, mode='constant', value=pad_value)
    return padded_tensor

# class DifferentialTransformerBlockrope(nn.Module):
#     def __init__(self, embed_dim, num_heads, depth, args, causal=False):
#         """
#         Differential Transformer Block with optional causal self-attention or cross-attention
#         and automatic application of ROPE.
        
#         Args:
#             embed_dim (int): Embedding dimension.
#             num_heads (int): Number of attention heads.
#             depth (int): Depth level for lambda initialization.
#             args (Namespace): Arguments for attention settings.
#             causal (bool): Whether to use causal masking by default for self-attention.
#         """
#         super(DifferentialTransformerBlockrope, self).__init__()
        
#         # Multihead differential attention module
#         self.attn = MultiheadFlashrope(args, embed_dim, depth, num_heads)
#         self.causal = causal
#         self.depth = depth

#         self.feed_forward = SwiGLU(embed_dim)
#         self.norm1 = RMSNorm(embed_dim, eps=1e-5)
#         self.norm2 = RMSNorm(embed_dim, eps=1e-5)
    
#     def forward(self, x, context=None, use_cache=False,return_attn=False):
#         """
#         Args:
#             x: Input tensor
#             context: Optional context for cross-attention
#             use_cache: Whether to use KV cache
#         """
#         # Enable/disable KV cache in attention module
#         self.attn.kv_cache_enabled = use_cache

#         if context is None:
#             attn_out = self.attn(self.norm1(x), causal=self.causal, return_attn=return_attn)
#         else:
#             # 💥 核心修复：只要是 Cross-Attention，绝对不允许存在 Causal Mask！
#             attn_out = self.attn(self.norm1(x), context=self.norm2(context), causal=False, return_attn=return_attn)

#         x = x + attn_out

#         # Feed-forward network with residual connection
#         ff_out = self.feed_forward(self.norm2(x))
#         x = x + ff_out
        
#         return x

class DifferentialTransformerBlockrope(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, args, causal=False):
        super(DifferentialTransformerBlockrope, self).__init__()
        
        # 1. 专门负责序列内部推导的【自注意力模块】
        self.self_attn = MultiheadFlashrope(args, embed_dim, depth, num_heads)
        
        # 2. 专门负责引入点云条件的【交叉注意力模块】
        self.cross_attn = MultiheadFlashrope(args, embed_dim, depth, num_heads) 
        
        self.causal = causal
        self.depth = depth

        self.feed_forward = SwiGLU(embed_dim)
        
        # Pre-LN 架构需要 4 个独立的 LayerNorm
        self.norm1 = RMSNorm(embed_dim, eps=1e-5) # 用于 Self-Attn
        self.norm2 = RMSNorm(embed_dim, eps=1e-5) # 用于 Cross-Attn 的 Q
        self.norm_context = RMSNorm(embed_dim, eps=1e-5) # 用于 Cross-Attn 的 K,V
        self.norm3 = RMSNorm(embed_dim, eps=1e-5) # 用于 FFN
    
    def forward(self, x, context=None, use_cache=False, return_attn=False):
        # 🟢 第一阶段：自注意力
        self.self_attn.kv_cache_enabled = use_cache
        self_attn_out = self.self_attn(self.norm1(x), causal=self.causal, return_attn=return_attn)
        x = x + self_attn_out

        # 🔵 第二阶段：交叉注意力
        if context is not None:
            self.cross_attn.kv_cache_enabled = False 
            
            # 💥 核心修复：精准计算当前 Token 的绝对位置！
            # 注意：因为 self_attn 运行完后 cache_pos 已经自动加上了序列长度，所以这里要减去 x.size(1) 才能拿到正确的当前步数
            
            cross_attn_out = self.cross_attn(
                self.norm2(x), 
                context=self.norm_context(context), 
                causal=False, 
                return_attn=return_attn,
            )
            x = x + cross_attn_out

        # 🟠 第三阶段：前馈网络
        ff_out = self.feed_forward(self.norm3(x))
        x = x + ff_out
        
        return x

    def init_kv_cache(self, batch_size, max_seq_len, dtype=torch.float32):
        """Initialize KV cache for this transformer block"""
        self.attn.empty_kv_cache(
            batch_size=batch_size,
            kv_cache_maxlen=max_seq_len,
            dtype=dtype
        )
        element_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }[dtype]
        
        # K cache size
        k_cache_size = (batch_size * 
                    max_seq_len * 
                    self.attn.num_heads * 
                    self.attn.head_dim * 
                    element_size)
        
        # V cache size
        v_cache_size = k_cache_size  
        
       
        total_cache_size = k_cache_size + v_cache_size
        cache_size_gb = total_cache_size / (1024**3)
        return cache_size_gb
    def reset_kv_cache(self):
        """Reset KV cache for this transformer block"""
        self.attn.reset_cache()

class DifferentialTransformerBlocklinearrope(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, args, causal=False):
        super(DifferentialTransformerBlocklinearrope, self).__init__()
        
        self.self_attn = MultiheadFlashlinearrope(args, embed_dim, depth, num_heads)
        self.cross_attn = MultiheadFlashlinearrope(args, embed_dim, depth, num_heads)
        
        self.causal = causal
        self.depth = depth

        self.feed_forward = SwiGLU(embed_dim)
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)
        self.norm_context = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)
    
    def forward(self, x, context=None, use_cache=False):
        # 🟢 第一阶段：自注意力
        self.self_attn.kv_cache_enabled = use_cache
        self_attn_out = self.self_attn(self.norm1(x), causal=self.causal)
        x = x + self_attn_out

        # 🔵 第二阶段：交叉注意力
        if context is not None:
            self.cross_attn.kv_cache_enabled = False
            
            # 💥 核心修复：精准计算当前 Token 的绝对位置
            current_pos = (self.self_attn.cache_pos - x.size(1)) if use_cache else 0
            
            cross_attn_out = self.cross_attn(
                self.norm2(x), 
                context=self.norm_context(context), 
                causal=False,
            )
            x = x + cross_attn_out

        # 🟠 第三阶段：前馈网络
        ff_out = self.feed_forward(self.norm3(x))
        x = x + ff_out
        
        return x

    def init_kv_cache(self, batch_size, max_seq_len, dtype=torch.float32):
        """Initialize KV cache for this transformer block"""
        self.attn.empty_kv_cache(
            batch_size=batch_size,
            kv_cache_maxlen=max_seq_len,
            dtype=dtype
        )
        element_size = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }[dtype]
        
        # K cache size
        kv_cache_size = (batch_size * 
                     self.attn.head_dim * 
                    self.attn.num_heads * 
                    self.attn.head_dim * 
                    element_size)
        

   
        total_cache_size = kv_cache_size
        cache_size_gb = total_cache_size / (1024**3)    
        return cache_size_gb
    def reset_kv_cache(self):
        """Reset KV cache for this transformer block"""
        self.attn.reset_cache()




class iFlame(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, depth=2, num_categories=0, length=10):
        super(iFlame, self).__init__()
        self.embed_dim = embed_dim
        self.depth = 2
        self.skip_weights2 = nn.Parameter(torch.ones(2))
      

        self.embedding = nn.Embedding(num_categories, embed_dim) 

        seq_len =8 * length + 16

        self.downsamplers = nn.ModuleList([     
            nn.AvgPool1d(kernel_size=3, stride=3, padding=0, ceil_mode=True),
            nn.AvgPool1d(kernel_size=3, stride=3, padding=0, ceil_mode=True)
        ])
        self.upsamplers = nn.ModuleList([
            nn.Upsample(scale_factor=3, mode='nearest'),
            nn.Upsample(scale_factor=3, mode='nearest')
        ])

        self.encoder_blocks = nn.ModuleList([
            nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)

            for i in range(0,4)
        ]),
            nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(4,8)
        ])
        ])


        self.bottlenecke = nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(8,12)
        ])
        self.bottleneckd = nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(12,16)
        ])
        self.decoder_blocks = nn.ModuleList([
            nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(16,20)
        ]),
            nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) if (i+1) % 4== 0
                else DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(20,24)
        ])
        ])

        self.output_proj = nn.Linear(embed_dim, num_categories)
        self.factor = [3, 3]
        self.norm = RMSNorm(embed_dim, eps=1e-5)

        # 【新增】实例化点云适配器
        # self.pc_adapter = PointMAEAdapter(
        #     weights_path="./checkpoints/pretrain.pth", # 你的权重路径
        #     mae_dim=384, 
        #     iflame_dim=embed_dim
        # )
        # 【新代码 - 接入 Michelangelo】
        self.pc_adapter = MichelangeloAdapter(
            weights_path="./checkpoints/michelangelo_encoder_only.pth", # 你实际的权重路径
            iflame_dim=embed_dim,
            # miche_dim=768, 
            freeze=True
        )
    def forward(self, x, pc=None, sampled_points=None):
        # 1. 记录下进来的真实长度 (比如 8191)
        original_seq_len = x.shape[1]
        if x.shape[1] % 9!= 0:
            x = pad_to_multiple(x,9, 1)
        
        x = self.embedding(x)

        x = self.norm(x) 

        # ==================================================
        # 🔥 修复版：物理连贯的 CFG (无分类器引导)
        # ==================================================
        # 1. 永远提取真实的特征 (不给 PointMAE 喂垃圾数据)
        context = None
        if pc is not None:
            context = self.pc_adapter(pc) # 输出比如 [B, N, C]
            
            # 2. 💥 在【特征层面】进行 CFG 抹除！
            # if self.training and torch.rand(1, device=x.device) < 0.15:
            #     # 直接把提取好的特征变成全零，这样传给 Cross-Attention 的 K 和 V 才是真正的 "空"
            #     context = torch.zeros_like(context) 
        else:
            # 如果推理时完全没给 pc，直接构造一个全零的 context 形状
            # 注意：这里需要你根据 pc_adapter 的输出维度写死
            dummy_b = x.shape[0]
            context = torch.zeros((dummy_b, 257, self.embed_dim), device=x.device, dtype=x.dtype)
        # ==================================================

        encoder_outputs = []

        for scale in range(self.depth):
            for block in self.encoder_blocks[scale]:
                    x = block(x)  # Self-attention

            encoder_outputs.append(x)
            x = x.transpose(1, 2)
            x = self.downsamplers[scale](x)
            x = x.transpose(1, 2)


        # 【极其重要】：在瓶颈层传入 context 进行交叉注意力
        for block in self.bottlenecke:
            x = block(x, context=context)

        for i, block in enumerate(self.bottleneckd):
            x = block(x, context=context)
                
        for scale in range(self.depth):
            x = self.upsamplers[scale](x.transpose(1, 2))
            x = x.transpose(1, 2) 
            skip = encoder_outputs[-(scale + 1)]
            


            x = shift_sequence(x, self.factor[scale] - 1)
            x =  self.skip_weights2[scale]*x + skip

            for block in self.decoder_blocks[scale]:
                    x = block(x)  # Self-attention

        x = self.output_proj(x)
        # 2. 【核心修复】：在输出前，把尾部多余的 8 个 Padding 预测切掉
        if x.shape[1] > original_seq_len:
            x = x[:, :original_seq_len, :]
        return x



    def init_kv_cache(self, batch_size, max_len=90, dtype=torch.float16):
     
        Gb=0
        self.use_cache = True
        self.inference_state = {
            'cache_initialized': False,
            'cur_pos': 0,
            # 'encoder_outputs': [],
            'dtype': dtype,
            'batch_size': batch_size,
            'max_len': max_len,
          
            'layer_states': {
                'encoder_0': None,  
                'encoder_1': None,  
                # 'bottleneck': None,  
            },
    
            'upsampled_states': {
                'decoder_0': None,
                'decoder_1': None
            }
        }
        ll=[1,3,9]
    
        for scale in range(self.depth):
            for i, block in enumerate(self.encoder_blocks[scale]):
                 Gb+=block.init_kv_cache(batch_size, max_len//ll[scale], dtype)
            
        for i, block in enumerate(self.bottlenecke):
            Gb+=block.init_kv_cache(batch_size, max_len//ll[2], dtype)
            
        for i, block in enumerate(self.bottleneckd):
            Gb+=block.init_kv_cache(batch_size, max_len//ll[2], dtype)
            
        for scale in range(self.depth):
            for i, block in enumerate(self.decoder_blocks[scale]):
                Gb+=block.init_kv_cache(batch_size, max_len//ll[1-scale], dtype)
        return Gb
    
    def reset_kv_cache(self):
   
        if hasattr(self, 'inference_state'):
          
            self.inference_state['cur_pos'] = 0
            # self.inference_state['encoder_outputs'] = []
            self.inference_state['cache_initialized'] = False
            self.inference_state['layer_states'] = {
                'encoder_0': None,
                'encoder_1': None,
                'bottleneck': None,
            }
            self.inference_state['upsampled_states'] = {
                'decoder_0': None,
                'decoder_1': None
            }
            
   
            for scale in range(self.depth):
                for block in self.encoder_blocks[scale]:
                    block.reset_kv_cache()
                
            for block in self.bottlenecke:
                block.reset_kv_cache()
                
            for block in self.bottleneckd:
                block.reset_kv_cache()
                
            for scale in range(self.depth):
                for block in self.decoder_blocks[scale]:
                    block.reset_kv_cache()
    
    def _process_first_tokens(self, x):

        x = self.embedding(x)
        x = self.norm(x)
        

        encoder_outputs = []
        

        for block in self.encoder_blocks[0]:
            x = block(x, use_cache=True)
        encoder_outputs.append(x)
        self.inference_state['layer_states']['encoder_0'] = x[:, -3:]
        

        x_downsampled = x.transpose(1, 2)
        x_downsampled = self.downsamplers[0](x_downsampled)
        x_downsampled = x_downsampled.transpose(1, 2)
        

        for block in self.encoder_blocks[1]:
            x_downsampled = block(x_downsampled, use_cache=True)
        encoder_outputs.append(x_downsampled)
        self.inference_state['layer_states']['encoder_1'] = x_downsampled[:, -3:]
        

        x_bottleneck = x_downsampled.transpose(1, 2)
        x_bottleneck = self.downsamplers[1](x_bottleneck)
        x_bottleneck = x_bottleneck.transpose(1, 2)
        

        for block in self.bottlenecke:
            x_bottleneck = block(x_bottleneck, use_cache=True)
            
        for block in self.bottleneckd:
            x_bottleneck = block(x_bottleneck, use_cache=True)
        
  
        x_upsampled = self.upsamplers[0](x_bottleneck.transpose(1, 2)).transpose(1, 2)
        self.inference_state['upsampled_states']['decoder_0'] = x_upsampled[:, -3:]
        
        skip = encoder_outputs[1]  
        
        x_upsampled = shift_sequence(x_upsampled, self.factor[0] - 1)
        x_upsampled = self.skip_weights2[0] * x_upsampled + skip
        
        for block in self.decoder_blocks[0]:
            x_upsampled = block(x_upsampled, use_cache=True)
        

        x_final = self.upsamplers[1](x_upsampled.transpose(1, 2)).transpose(1, 2)
        self.inference_state['upsampled_states']['decoder_1'] = x_final[:, -3:]
        
        skip = encoder_outputs[0]  
        
        x_final = shift_sequence(x_final, self.factor[1] - 1)
        x_final = self.skip_weights2[1] * x_final + skip
        
        for block in self.decoder_blocks[1]:
            x_final = block(x_final, use_cache=True)
        

        logits = self.output_proj(x_final)
        

        self.inference_state['cur_pos'] = x.shape[1]
        self.inference_state['cache_initialized'] = True
        
        return logits
    def _process_single_token(self, x):

        batch_size = x.shape[0]
        cur_pos = self.inference_state['cur_pos']
        

        x = self.embedding(x)
        x = self.norm(x)
        
 
        update_encoder_0 = True  
        update_encoder_1 = (cur_pos + 1) % 3 == 0 
        update_bottleneck = (cur_pos + 1) % 9 == 0 
        update_decoder_0 = (cur_pos + 1) % 3 == 0  
        update_decoder_1 = True  
   
        if update_encoder_0:
            encoder_0_output = x
            for block in self.encoder_blocks[0]:
                encoder_0_output = block(encoder_0_output, use_cache=True)
            self.inference_state['layer_states']['encoder_0'][:,cur_pos% 3:cur_pos% 3+1] = encoder_0_output
          
        if update_encoder_1:
            
            recent_tokens = 3
      
            recent_encoder_outputs = self.inference_state['layer_states']['encoder_0']#[:, (cur_pos-2)% 3:(cur_pos)%3+1 ]

            x_downsampled = recent_encoder_outputs.transpose(1, 2)
            x_downsampled = self.downsamplers[0](x_downsampled)
            x_downsampled = x_downsampled.transpose(1, 2)
            
          
            for block in self.encoder_blocks[1]:
                x_downsampled = block(x_downsampled, use_cache=True)
            self.inference_state['layer_states']['encoder_1'][:,(cur_pos-2)%9//3:(cur_pos-2)%9//3+1] = x_downsampled
     
        if update_bottleneck:
            
            recent_tokens = 3
      
            recent_encoder_1_outputs = self.inference_state['layer_states']['encoder_1']#[:, -recent_tokens:]
            
            x_bottleneck = recent_encoder_1_outputs.transpose(1, 2)
            x_bottleneck = self.downsamplers[1](x_bottleneck)
            x_bottleneck = x_bottleneck.transpose(1, 2)
            

            for block in self.bottlenecke:
                x_bottleneck = block(x_bottleneck, use_cache=True)
            
            for block in self.bottleneckd:
                x_bottleneck = block(x_bottleneck, use_cache=True)
            
   
            bottleneck_output =x_bottleneck# self.inference_state['layer_states']['bottleneck']
            x_upsampled = self.upsamplers[0](bottleneck_output.transpose(1, 2)).transpose(1, 2)
            self.inference_state['upsampled_states']['decoder_0'] = x_upsampled

        if update_decoder_0:
           
     
            upsampled_decoder0_idx = ((cur_pos-2)%9//3-2)%3 
            
     
            x_upsampled = self.inference_state['upsampled_states']['decoder_0'][:, upsampled_decoder0_idx:upsampled_decoder0_idx+1]
            
 
         
            encoder_1_skip_idx =  (cur_pos-2)%9//3  
            encoder_1_output = self.inference_state['layer_states']['encoder_1'][:, encoder_1_skip_idx:encoder_1_skip_idx+1]
            
  
            x_upsampled = self.skip_weights2[0] * x_upsampled + encoder_1_output
            

            for block in self.decoder_blocks[0]:
                x_upsampled = block(x_upsampled, use_cache=True)
            

            x_final = self.upsamplers[1](x_upsampled.transpose(1, 2)).transpose(1, 2)
            

            self.inference_state['upsampled_states']['decoder_1'] = x_final
       
        if update_decoder_1:
            upsampled_decoder1_idx =  (cur_pos-2)%3

            x_final = self.inference_state['upsampled_states']['decoder_1'][:, upsampled_decoder1_idx:upsampled_decoder1_idx+1]
  
            encoder_0_output = self.inference_state['layer_states']['encoder_0'][:, cur_pos % 3:cur_pos % 3+1]

            x_final = self.skip_weights2[1] * x_final + encoder_0_output
            
            for block in self.decoder_blocks[1]:
                x_final = block(x_final, use_cache=True)
            
       
        logits = self.output_proj(x_final)
        
      
        self.inference_state['cur_pos'] += 1
        
        return logits
    def inference_step(self, x,pc=None, use_cache=True):
      
        # self.cache_size=self.init_kv_cache(batch_size, max_seq_len, dtype=torch.float16)
        
        if not hasattr(self, 'inference_state'):
            self.cache_size=self.init_inference(x.shape[0])
        
      
        if not self.inference_state['cache_initialized']:
            return self._process_first_tokens(x)
        else:
           
            if x.shape[1] > 1:
                #
                all_logits = []
                for i in range(x.shape[1]):
                    token = x[:, i:i+1]
                    logits = self._process_single_token(token)
                    all_logits.append(logits)
               
                return torch.cat(all_logits, dim=1)
            else:
                return self._process_single_token(x)
            
    @torch.no_grad()
    def generate_sequence(
        self,
        initial_input: torch.Tensor,
        pc,
        max_seq_len: int,
        device: str,
        shorten_factor: int = 3,
        end_symbol: int = 4737,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        temperature: float = 1.0,
        pad_symbol: int = 4739  # ⚠️ 务必改成你真实的 PAD_TOKEN ID！
    ) -> np.ndarray:
        
        self.eval()
        batch_size = initial_input.size(0)
        current_len = initial_input.size(1)
        
        steps_to_generate = max_seq_len - current_len
        pbar = tqdm(total=steps_to_generate, desc="🚀 [状态机+静态画布] 正在生成 3D Mesh", unit="token")
        
        # --- 词表硬边界常量 ---
        PATCH_BASE = 0
        BLOCK_BASE = 64
        OFFSET_BASE = 576
        SPECIAL_PATCH_BASE = 4672
        VOCAB_END = 4736
        SOS_TOKEN = 4736
        EOS_TOKEN = 4737
        SEP_TOKEN = 4738
        
        with torch.no_grad():
            # ====================================================
            # 🖼️ 1. 创建“静态画布” (Static Context)
            # 永远保持 4608 的长度，彻底锁死 AvgPool1d 的滑动窗口！
            # ====================================================
            seq = torch.full((batch_size, max_seq_len), pad_symbol, dtype=torch.long, device=device)
            seq[:, :current_len] = initial_input # 填入初始的 [SOS]
            
            # 从第 current_len 个位置开始，逐个往后填
            for i in range(current_len, max_seq_len):
                
                # 每次前向传播，喂入的永远是完整的 4608 长度序列！
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = self(seq, pc=pc)
                
                # 取出用来预测第 i 个 Token 的 logits (也就是第 i-1 步的输出特征)
                last_logits = output[:, i - 1, :]
                
                # ====================================================
                # 🛡️ 2. 状态机强制拓扑约束
                # ====================================================
                # ⚠️ 注意：只看目前已经生成的真实 Token (索引 :i)，不看后面的 PAD
                seq_1d = seq[0, :i] 
                
                has_sep = (seq_1d == SEP_TOKEN).any().item()
                has_offset = ((seq_1d >= OFFSET_BASE) & (seq_1d < SPECIAL_PATCH_BASE)).any().item()
                
                mask = torch.full_like(last_logits, -float('inf'))
                
                if has_offset:
                    # 状态 3：只要出现了 Offset，彻底进入微观阶段
                    mask[:, OFFSET_BASE : SPECIAL_PATCH_BASE] = 0.0
                    mask[:, EOS_TOKEN] = 0.0
                    mask[:, SEP_TOKEN] = 0.0
                elif has_sep:
                    # 状态 2：跨过了第一座桥(SEP)，但还没生成任何 Offset
                    mask[:, BLOCK_BASE : SPECIAL_PATCH_BASE] = 0.0
                    mask[:, SEP_TOKEN] = 0.0
                else:
                    # 状态 1：连第一座桥(SEP)都没遇到
                    mask[:, 0 : BLOCK_BASE] = 0.0
                    mask[:, SPECIAL_PATCH_BASE : VOCAB_END] = 0.0
                    mask[:, SEP_TOKEN] = 0.0
                
                # 为了数值安全，先除以温度，再加上 mask (-inf 依然是 -inf)
                temp = max(temperature, 1e-5)
                logits = (last_logits / temp) + mask
                
                # ====================================================
                # 🎲 3. 采样生成 (Top-K / Top-P)
                # ====================================================
                probs = F.softmax(logits, dim=-1)
                
                # Top-K 过滤
                if top_k is not None and top_k > 0:
                    top_k = min(top_k, probs.size(-1))
                    topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                    mask_k = torch.zeros_like(probs, dtype=torch.bool)
                    mask_k.scatter_(1, topk_indices, 1)
                    probs = probs.masked_fill(~mask_k, 0.0)
                
                # Top-P 过滤
                if top_p is not None and top_p > 0.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
                    probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
                
                # 重新归一化概率并采样
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1) # [Batch]
                
                # 🖌️ 4. 将生成的 Token 填入“画布”对应的位置
                seq[:, i] = next_token

                pbar.update(1)

                # 判断是否生成了终止符
                if (next_token == end_symbol).any():
                    pbar.write(f"✨ [状态机] 捕捉到 EOS 结束符，生成正常完成于第 {i + 1} 步！")
                    # 截断后面的 Padding，只返回有用的部分
                    seq = seq[:, :i + 1]
                    break

        pbar.close()
        return seq.cpu().numpy()


class Causal_iFlame(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, depth=16, num_categories=0):
        super(Causal_iFlame, self).__init__()
        self.embed_dim = embed_dim
        
        # 1. 词嵌入与归一化 (保持不变)
        self.embedding = nn.Embedding(num_categories, embed_dim) 
        self.norm = RMSNorm(embed_dim, eps=1e-5)

        # 2. 接入 Michelangelo 点云适配器 (保持不变)
        self.pc_adapter = MichelangeloAdapter(
            weights_path="./checkpoints/michelangelo_encoder_only.pth", 
            iflame_dim=embed_dim,
            freeze=True
        )

        # ==================================================
        # 🚀 核心重构：彻底展平架构，纯因果 Transformer
        # ==================================================
        # 你的原代码总共有 24 层 (Encoder 8 + Bottleneck 8 + Decoder 8)
        # 现在我们将它们展平成一个连续的、纯自回归的层级序列
        # 注意：你的自定义 Block 里自带了 causal=True，这非常完美！现在它终于能真正生效了！
        
        self.layers = nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) 
            if (i + 1) % 4 == 0 else 
            DifferentialTransformerBlocklinearrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) #删掉
            for i in range(depth) # 默认 24 层
        ])

        # 3. 输出层 (保持不变)
        self.output_proj = nn.Linear(embed_dim, num_categories)

    def forward(self, x, pc=None, context=None, use_cache=False):
        """
        兼容双模式的前向传播：
        - 训练时：传入 pc，内部计算 context (包含 15% CFG 盲画逻辑)
        - 推理时：直接传入预计算好的 context，极致省算力
        """
        
        # ==================================================
        # 🛡️ 提取点云 Context 与 CFG 逻辑 (如果没传 context 的话)
        # ==================================================
        if context is None:
            if pc is not None:
                context = self.pc_adapter(pc) # [B, 257, embed_dim]
                
                # 训练时的 15% 盲画逻辑
                if self.training and torch.rand(1, device=x.device).item() < 0.15:
                    context = torch.zeros_like(context) 
            else:
                dummy_b = x.shape[0]
                context = torch.zeros((dummy_b, 257, self.embed_dim), device=x.device, dtype=x.dtype)

        # ==================================================
        # 🚀 核心网络层
        # ==================================================
        x = self.embedding(x)
        x = self.norm(x)
        
        for block in self.layers:
            x = block(x, context=context, use_cache=use_cache)
            
        x = self.output_proj(x)
        
        return x

    @torch.no_grad()
    def generate_sequence(
        self,
        initial_input: torch.Tensor,
        pc,
        max_seq_len: int = 4608,
        device: str = 'cuda',
        end_symbol: int = 4737,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        temperature: float = 0.3,
        cfg_scale: float = 1.0  # 🔥 新增：CFG 引导尺度 (设为 > 1.0 开启)
    ) -> np.ndarray:
        """
        专为纯 Causal Decoder 设计的生成循环，原生支持 CFG 与状态机拓扑约束
        """
        self.eval()
        generated = initial_input.to(device)
        
        steps_to_generate = max_seq_len - generated.size(1)
        pbar = tqdm(total=steps_to_generate, desc="🚀 [纯因果] 正在生成 3D Mesh", unit="token")
        
        # --- 词表硬边界常量 (对齐你的 CascadedTokenizer) ---
        PATCH_BASE = 0
        BLOCK_BASE = 64
        OFFSET_BASE = 576
        SPECIAL_PATCH_BASE = 4672
        VOCAB_END = 4736
        SOS_TOKEN = 4736
        EOS_TOKEN = 4737
        SEP_TOKEN = 4738
        
        for _ in range(steps_to_generate):
            
            # ====================================================
            # 🚀 1. 动态前向传播 (支持 CFG 双路推理)
            # 纯因果模型可以安全地接受任意长度的输入，无需 PAD 占位！
            # ====================================================
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                
                # 如果 cfg_scale > 1.0，开启强大的无分类器引导
                if cfg_scale > 1.0 and pc is not None:
                    # [路 1] 有条件输出 (盯着点云看)
                    cond_output = self(generated, pc=pc)
                    cond_logits = cond_output[:, -1, :]
                    
                    # [路 2] 无条件输出 (强行闭上眼睛猜)
                    # 我们在 forward 里写了：如果 pc=None，会自动用全零 context
                    uncond_output = self(generated, pc=None)
                    uncond_logits = uncond_output[:, -1, :]
                    
                    # 💥 CFG 魔法公式：将模型往“符合点云条件”的方向强力推拽
                    last_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
                else:
                    # 普通的单路推理 (速度快一倍)
                    output = self(generated, pc=pc)
                    last_logits = output[:, -1, :]
            
            # ====================================================
            # 🛡️ 2. 状态机强制拓扑约束 (终极特征探测白名单)
            # ====================================================
            seq_1d = generated[0]
            has_sep = (seq_1d == SEP_TOKEN).any().item()
            has_offset = ((seq_1d >= OFFSET_BASE) & (seq_1d < SPECIAL_PATCH_BASE)).any().item()
            
            mask = torch.full_like(last_logits, -float('inf'))
            
            if has_offset:
                mask[:, OFFSET_BASE : SPECIAL_PATCH_BASE] = 0.0
                mask[:, EOS_TOKEN] = 0.0
                mask[:, SEP_TOKEN] = 0.0
            elif has_sep:
                mask[:, BLOCK_BASE : SPECIAL_PATCH_BASE] = 0.0
                mask[:, SEP_TOKEN] = 0.0
            else:
                mask[:, 0 : BLOCK_BASE] = 0.0
                mask[:, SPECIAL_PATCH_BASE : VOCAB_END] = 0.0
                mask[:, SEP_TOKEN] = 0.0
            
            # 数值安全：先除以温度，再加掩码
            temp = max(temperature, 1e-5)
            logits = (last_logits / temp) + mask
            
            # ====================================================
            # 🎲 3. 采样生成 (Top-K / Top-P)
            # ====================================================
            probs = F.softmax(logits, dim=-1)
            
            if top_k is not None and top_k > 0:
                top_k = min(top_k, probs.size(-1))
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                mask_k = torch.zeros_like(probs, dtype=torch.bool)
                mask_k.scatter_(1, topk_indices, 1)
                probs = probs.masked_fill(~mask_k, 0.0)
            
            if top_p is not None and top_p > 0.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
                probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
            
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
            next_token = torch.multinomial(probs, num_samples=1) # 形状: [Batch, 1]
            
            # 🖌️ 4. 回归最原汁原味的动态拼接
            generated = torch.cat([generated, next_token], dim=1)

            pbar.update(1)

            if (next_token == end_symbol).any():
                pbar.write(f"✨ [纯因果模型] 捕捉到 EOS 结束符，生成完美完成于第 {generated.size(1)} 步！")
                break

        pbar.close()
        return generated.cpu().numpy()