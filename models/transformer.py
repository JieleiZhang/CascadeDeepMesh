from models.multihead_flashdiff_2 import MultiheadFlashrope,MultiheadFlashlinearrope
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

class DifferentialTransformerBlockrope(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, args, causal=False):
        """
        Differential Transformer Block with optional causal self-attention or cross-attention
        and automatic application of ROPE.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            depth (int): Depth level for lambda initialization.
            args (Namespace): Arguments for attention settings.
            causal (bool): Whether to use causal masking by default for self-attention.
        """
        super(DifferentialTransformerBlockrope, self).__init__()
        
        # Multihead differential attention module
        self.attn = MultiheadFlashrope(args, embed_dim, depth, num_heads)
        self.causal = causal
        self.depth = depth

        self.feed_forward = SwiGLU(embed_dim)
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)
    
    def forward(self, x, context=None, use_cache=False,return_attn=False):
        """
        Args:
            x: Input tensor
            context: Optional context for cross-attention
            use_cache: Whether to use KV cache
        """
        # Enable/disable KV cache in attention module
        self.attn.kv_cache_enabled = use_cache

        if context is None:
            attn_out = self.attn(self.norm1(x), causal=self.causal,return_attn=return_attn)
        else:
            attn_out = self.attn(self.norm1(x), context=self.norm2(context), causal=self.causal,return_attn=return_attn)

        x = x + attn_out

        # Feed-forward network with residual connection
        ff_out = self.feed_forward(self.norm2(x))
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
        """
        Differential Transformer Block with optional causal self-attention or cross-attention
        and automatic application of ROPE.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            depth (int): Depth level for lambda initialization.
            args (Namespace): Arguments for attention settings.
            causal (bool): Whether to use causal masking by default for self-attention.
        """
        super(DifferentialTransformerBlocklinearrope, self).__init__()
        
        # Multihead differential attention module
        self.attn = MultiheadFlashlinearrope(args, embed_dim, depth, num_heads)
        self.causal = causal
        self.depth = depth

        self.feed_forward = SwiGLU(embed_dim)
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)
    
    def forward(self, x, context=None, use_cache=False):
        """
        Args:
            x: Input tensor
            context: Optional context for cross-attention
            use_cache: Whether to use KV cache
        """
        # Enable/disable KV cache in attention module
        self.attn.kv_cache_enabled = use_cache

        if context is None:
            attn_out = self.attn(self.norm1(x), causal=self.causal)
        else:
            attn_out = self.attn(self.norm1(x), context=self.norm2(context), causal=self.causal)

        x = x + attn_out

        # Feed-forward network with residual connection
        ff_out = self.feed_forward(self.norm2(x))
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



from einops import rearrange

class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor
        self.proj = nn.Linear(dim * shorten_factor, dim)

    def forward(self, x):
        # x: [B, L, D]
        # 这里的 n 是下采样后的长度 L/s
        x = rearrange(x, 'b (n s) d -> b n (s d)', s=self.shorten_factor)
        return self.proj(x)

class LinearUpsample(nn.Module):
    def __init__(self, dim, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.proj = nn.Linear(dim, dim * upscale_factor)

    def forward(self, x):
        # x: [B, L, D]
        x = self.proj(x)
        # 将通道维度重新展开到长度维度: [B, L*s, D]
        x = rearrange(x, 'b n (s d) -> b (n s) d', s=self.upscale_factor)
        return x
    

class iFlame(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, depth=2, num_categories=0, length=10):
        super(iFlame, self).__init__()
        self.embed_dim = embed_dim
        self.depth = 2
        self.skip_weights2 = nn.Parameter(torch.ones(2))
      
        self.embedding = nn.Embedding(num_categories, embed_dim) 

        # 替换原有的下采样和上采样
        self.downsamplers = nn.ModuleList([     
            LinearDownsample(embed_dim, shorten_factor=3),
            LinearDownsample(embed_dim, shorten_factor=3)
        ])
        self.upsamplers = nn.ModuleList([
            LinearUpsample(embed_dim, upscale_factor=3),
            LinearUpsample(embed_dim, upscale_factor=3)
        ])

        # 【极其重要】：务必确保 DifferentialTransformerBlock 内部的 Cross-Attention 部分是双向的 (Bidirectional)！
        # 这里的 causal=True 只能作用于文本/主序列的 Self-Attention，绝不能掩码点云的 Context！
        self.encoder_blocks = nn.ModuleList([
            nn.ModuleList([
                DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
                for i in range(0, 4)
            ]),
            nn.ModuleList([
                DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) 
                for i in range(4, 8)
            ])
        ])

        self.bottlenecke = nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(8, 12)
        ])
        
        self.bottleneckd = nn.ModuleList([
            DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True)
            for i in range(12, 16)
        ])
        
        self.decoder_blocks = nn.ModuleList([
            nn.ModuleList([
                DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) 
                for i in range(16, 20)
            ]),
            nn.ModuleList([
                DifferentialTransformerBlockrope(embed_dim, num_heads, depth=i+1, args=None, causal=True) 
                for i in range(20, 24)
            ])
        ])

        self.output_proj = nn.Linear(embed_dim, num_categories)
        self.factor = [3, 3]
        self.norm = RMSNorm(embed_dim, eps=1e-5)

        # 接入 Michelangelo
        self.pc_adapter = MichelangeloAdapter(
            weights_path="./checkpoints/michelangelo_encoder_only.pth",
            iflame_dim=embed_dim,
            freeze=True
        )

        # 【修复 & 优化 1】：引入可学习的 Null Context 替代硬编码的全 0 张量
        # 假设 MichelangeloAdapter 提取的 patch 数量固定为 257 (含 cls token)
        self.miche_seq_len = 257 
        self.null_context = nn.Parameter(torch.randn(1, self.miche_seq_len, embed_dim))

    def forward(self, x, pc=None, sampled_points=None):
        batch_size = x.shape[0]
        original_seq_len = x.shape[1]
        
        if original_seq_len % 9 != 0:
            x = pad_to_multiple(x, 9, 1)
        
        x = self.embedding(x)
        x = self.norm(x) 

        # ==================================================
        # 【修复 2】：规范化 CFG (Classifier-Free Guidance) 逻辑
        # ==================================================
        if pc is not None:
            context = self.pc_adapter(pc) # 正常提取点云特征 [B, N, C]
            
            # 训练阶段：以 15% 的概率随机 Drop 掉条件，替换为 null_context
            if self.training and torch.rand(1, device=x.device).item() < 0.15:
                context = self.null_context.expand(batch_size, -1, -1)
        else:
            # 推理阶段：如果没有提供点云 (无条件生成)，直接使用 null_context
            context = self.null_context.expand(batch_size, -1, -1)
        # ==================================================

        encoder_outputs = []

        # Encoder 阶段
        for scale in range(self.depth):
            for block in self.encoder_blocks[scale]:
                x = block(x)  # 这里仅进行 Self-Attention
            
            encoder_outputs.append(x)
            x = self.downsamplers[scale](x)

        # Bottleneck 阶段（注入点云特征）
        for block in self.bottlenecke:
            x = block(x, context=context)

        for block in self.bottleneckd:
            x = block(x, context=context)
                
        # Decoder 阶段
        for scale in range(self.depth):
            x = self.upsamplers[scale](x)
            skip = encoder_outputs[-(scale + 1)]

            x = shift_sequence(x, self.factor[scale] - 1)
            x = self.skip_weights2[scale] * x + skip

            # 【优化 3】：在 Decoder 层也注入 context，加强模型对 3D 几何条件的感知对齐
            # 注意：前提是你的 Decoder Block 内部支持 context 参数的解析
            for block in self.decoder_blocks[scale]:
                x = block(x, context=context)  

        x = self.output_proj(x)
        
        # 裁剪掉 padding 部分
        if x.shape[1] > original_seq_len:
            x = x[:, :original_seq_len, :]
            
        return x


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