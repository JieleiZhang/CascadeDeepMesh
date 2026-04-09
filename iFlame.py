import glob
import os
# 🚀 极其关键：禁止底层 C++ 库在 DataLoader 子进程里多开线程内卷！
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
import math
import trimesh  
import numpy as np
from collections import OrderedDict
import pickle
import numpy as np
# from scipy.special import comb
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import trimesh
import random
# from models_ae1 import *
from flash_attn.modules.mha import MHA
from torch.cuda.amp import autocast, GradScaler
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

# from tqdm import tqdm
import wandb
from utils.datasetmy import TrianglesDataset, NpyTensorDataset, DynamicAugmentDataset
import re
import json
from typing import List, Optional, Tuple
import subprocess
import time
import torch._dynamo
# torch._dynamo.config.suppress_errors = True
import math
from models.transformer import *
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from flash_attn.modules.mlp import  GatedMlp
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.optimize_ddp = False
import sys

# desired_faceso=262
pad_symbol=-100
from pathlib import Path
name='shapenet'
argvname=name#sys.argv[1]
# 0.001 muon  
argvlr='0.001'#sys.argv[2]
optname='muon'#sys.argv[3]
class Args:
    def __init__(self):
        self.weight_decay = 0.1  # Default weight decay
        self.lr = float(argvlr)      # Absolute learning rate (to be computed)
        self.blr = 1e-4          # Base learning rate
        self.layer_decay = 0.75   # Layer-wise learning rate decay
        self.min_lr = 1.28e-6        # Minimum learning rate
        self.warmup_epochs =10    # Number of warmup epochs
        self.epochs =400      # Total number of training epochs
        self.batch_size = 16  #40#57   # Batch size
        # self.world_size = 4       # Number of dist5ributed processes (adjust if using DDP)

args = Args()


from torch.optim.lr_scheduler import LambdaLR    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5000, num_training_steps=10000, lr_start=1e-4, lr_end=1e-5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = lr_end + (lr_start - lr_end) * cosine_decay
        return lr / lr_start  # 归一化
    return LambdaLR(optimizer, lr_lambda)
def accuracy_(y_pred, y_true, ignore_label=None, device=None):
    y_pred = y_pred.argmax(dim=-1)

    if ignore_label:
        normalizer = torch.sum(y_true != ignore_label)  # type: ignore
        ignore_mask = torch.where(  # type: ignore
            y_true == ignore_label,
            torch.zeros_like(y_true, device=device),
            torch.ones_like(y_true, device=device)
        ).type(torch.float32)
    else:
        normalizer = y_true.shape[0]
        ignore_mask = torch.ones_like(y_true, device=device).type(torch.float32)
    acc = (y_pred.reshape(-1) == y_true.reshape(-1)).type(torch.float32)  # type: ignore
    acc = torch.sum(acc*ignore_mask.flatten())
    return acc / normalizer


def get_nvidia_smi():

    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return f"Error running nvidia-smi: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Exception occurred while running nvidia-smi: {str(e)}"


def discretizeit(vertices, num_tokens=256):

    vertices_scaled = np.clip(vertices + 0.5, 0, 1) * num_tokens
    vertices_quantized = np.round(vertices_scaled).astype(int)
    
    return vertices_quantized


def inverse_discretize(vertices_quantized, num_tokens=256):

    vertices_normalized = vertices_quantized.astype(float) / num_tokens - 0.5
    
    return vertices_normalized

def train_model(model, scheduler, optimizer, criterion, dataloader, device, scaler, epochs=200, rank=0, data_iter_step=0, startepoc=0, dataloader1=None):
    model.train()
    best_val_loss = float('inf') 
    
    if rank == 0:
        wandb.init(
            project="meshplayground",
            name=argvname + argvlr,
            config={"lr": args.lr, "epochs": epochs, "batch_size": dataloader.batch_size}
        )

    iter_idx = 0
    accumulation_steps = 1
  
    for epoch in range(1 + startepoc, epochs + 1):

        torch.cuda.empty_cache()
        total_loss = 0
        total_train_acc = 0.0 # 🌟 新增：用于累加训练集 epoch 的总准确率
        dataloader.sampler.set_epoch(epoch)
        
        # ==========================================================
        # 🏃‍♂️ 阶段 1：训练阶段 (Train)
        # ==========================================================
        with tqdm(dataloader, desc=f"Epoch {epoch}/{epochs} [Train]", disable=(rank != 0)) as pbar:
            for batch_idx, batch in enumerate(pbar):
                iter_idx += 1
                
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                sampled_points = batch['point_cloud'].to(device)
                
                inputs = input_ids[:, :-1].contiguous()
                targets = labels[:, 1:].contiguous().view(-1)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs, pc=sampled_points)
                    vocab_size = outputs.size(-1) 
                    logits = outputs.contiguous().view(-1, vocab_size)  
                    loss = criterion(logits, targets)
                    
                    suspicious_keywords = ['null_context', 'cond_proj', 'cond_head_proj']
                    
                    dummy_loss = 0.0
                    for name, param in model.module.named_parameters():
                        if param.requires_grad and any(kw in name for kw in suspicious_keywords):
                            dummy_loss += param.sum() * 0.0
                    
                    loss += dummy_loss
                
                if torch.isnan(loss):
                    print(f"⚠️ Rank {rank}: 检测到 NaN，跳过该 batch。")
                    optimizer.zero_grad() 
                    del outputs, logits, loss
                    torch.cuda.empty_cache() 
                    continue 
                
                loss_value = loss.item() 
                total_loss += loss_value
                
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                
                acc = accuracy_(logits.detach(), targets, ignore_label=-100, device=device)
                total_train_acc += acc.item() # 🌟 新增：累加 Batch Accuracy
                
                pbar.set_postfix({'loss': loss_value, 'acc': acc.item()})
                            
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step() 
                    optimizer.zero_grad()

                    if rank == 0 and iter_idx % 10 == 0:
                        wandb.log({"iter_train_loss": loss_value, "iter_train_acc": acc.item(), "lr": optimizer.param_groups[0]['lr']})

        avg_train_loss = total_loss / len(dataloader)
        avg_train_acc = total_train_acc / len(dataloader) # 🌟 新增：计算 Epoch 平均训练准确率

        # ==========================================================
        # 🧪 阶段 2：验证阶段 (Validation)
        # ==========================================================
        avg_val_loss = float('inf') 
        avg_val_acc = 0.0 # 🌟 新增：验证集平均准确率默认值
        
        if dataloader1 is not None:
            model.eval() 
            val_total_loss = 0
            val_total_acc = 0.0 # 🌟 新增：用于累加验证集 epoch 的总准确率
            
            with torch.no_grad(): 
                with tqdm(dataloader1, desc=f"Epoch {epoch}/{epochs} [Val]", disable=(rank != 0)) as val_pbar:
                    for val_batch in val_pbar:
                        val_input_ids = val_batch['input_ids'].to(device)
                        val_labels = val_batch['labels'].to(device)
                        val_sampled_points = val_batch['point_cloud'].to(device)
                        
                        val_inputs = val_input_ids[:, :-1].contiguous()
                        val_targets = val_labels[:, 1:].contiguous().view(-1)
                        
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            val_outputs = model(val_inputs, pc=val_sampled_points)
                            val_logits = val_outputs.contiguous().view(-1, val_outputs.size(-1))  
                            v_loss = criterion(val_logits, val_targets)
                            
                        # 🌟 新增：计算验证集的 Accuracy
                        v_acc = accuracy_(val_logits.detach(), val_targets, ignore_label=-100, device=device)
                        
                        val_total_loss += v_loss.item()
                        val_total_acc += v_acc.item() # 🌟 新增：累加验证集 Accuracy
                        
                        val_pbar.set_postfix({'val_loss': v_loss.item(), 'val_acc': v_acc.item()})
                        
            avg_val_loss = val_total_loss / len(dataloader1)
            avg_val_acc = val_total_acc / len(dataloader1) # 🌟 新增：计算 Epoch 平均验证准确率
            
            model.train() 
            try:
                del val_input_ids, val_labels, val_sampled_points, val_inputs, val_targets, val_outputs, val_logits, v_loss, v_acc
            except NameError:
                pass
            torch.cuda.empty_cache()

        # ==========================================================
        # 💾 阶段 3：日志打印与模型保存
        # ==========================================================
        if rank == 0:
            # 🌟 新增：在终端打印出 Train Acc 和 Val Acc
            print(f"✅ Epoch {epoch} 完成 | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
            
            # 🌟 新增：将 Accuracy 记录到 Wandb
            wandb_log_dict = {
                "epoch": epoch, 
                "epoch_train_loss": avg_train_loss,
                "epoch_train_acc": avg_train_acc
            }
            if dataloader1 is not None:
                wandb_log_dict["epoch_val_loss"] = avg_val_loss
                wandb_log_dict["epoch_val_acc"] = avg_val_acc
            wandb.log(wandb_log_dict)

            metric_to_track = avg_val_loss if dataloader1 is not None else avg_train_loss
            
            os.makedirs('checkpoints', exist_ok=True)
            current_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': metric_to_track,
                'data_iter_step': data_iter_step
            }
            
            if metric_to_track < best_val_loss:
                best_val_loss = metric_to_track
                best_path = "checkpoints/iFlame_best_400.pth"
                torch.save(current_checkpoint, best_path)
                print(f"⭐ 发现更低 Val Loss: {best_val_loss:.6f}, 已更新 best 权重。")
            
            if epoch % 10 == 0:
                epoch_path = f"checkpoints/iFlame_400_epoch_{epoch}_.pth"
                torch.save(current_checkpoint, epoch_path)
                print(f"💾 已备份 Epoch {epoch} 权重。")


def main(rank, world_size):
    # 1. 环境与随机种子初始化
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # A100上通常开启benchmark以加速卷积/矩阵运算
    args.seed=3407
    random.seed(args.seed)     
    np.random.seed(args.seed)  
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # DDP 环境设置
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12336'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # 2. 模型初始化
    num_categories = 4740  
    model = iFlame(num_categories=num_categories, embed_dim=768, num_heads=16)
    model.to(device)

    # 3. 特征冻结逻辑 (必须在 DDP 包装和 Optimizer 定义之前！)
    if rank == 0: 
        print("🛡️ 开启两阶段训练法：阶段一（精准冻结老教授，狂练新生儿！）")

    # 默认全开启
    # for param in model.parameters():
    #     param.requires_grad = True

    # # 针对性冻结
    # for name, param in model.named_parameters():
    #     # 注意：在DDP包装前，name不带'module.'前缀
    #     if 'pc_adapter.encoder' in name: 
    #         param.requires_grad = False
    #     if 'pc_adapter.cond_head_proj' in name or 'pc_adapter.cond_proj' in name:
    #         param.requires_grad = True

    # 4. 加载预训练权重 (在 DDP 包装前加载，逻辑更清晰)
    checkpoint_path = None # "checkpoints/iFlame_best.pth"
    startepoc = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        # 针对 ARM 平台，map_location 必须明确
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # 自动处理权重文件可能带有的 'module.' 前缀（如果权重是 DDP 保存的）
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        try:
            model.load_state_dict(new_state_dict, strict=True)
            if rank == 0: print(f"✅ 成功严格加载权重 '{checkpoint_path}'")
        except Exception as e:
            if rank == 0: print(f"⚠️ 严格加载失败，尝试非严格加载... (Missing/Unexpected keys)")
            model.load_state_dict(new_state_dict, strict=False)
        
        if rank == 0: print(f"🚀 已开启微调模式！")
    else:
        if rank == 0: print(f"⚠️ 未找到检查点，将从头开始训练！")

    # 5. DDP 包装与编译
    # find_unused_parameters: 如果你确信冻结的层在 forward 中仍被调用，设为 False 性能更好
    model = DDP(model, device_ids=[rank], find_unused_parameters=False, gradient_as_bucket_view=True)

    # torch.compile 建议放在 DDP 之后，且在 A100 上建议针对 bf16 优化
    # 如果运行报错，可以先注释掉这一行，ARM 上的兼容性有时比较诡异
    # model = torch.compile(model) 

    if rank == 0:
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"Total params: {total_params} | Trainable: {trainable_params}")

    # 6. 优化器与损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=pad_symbol)
    
    # 仅将 requires_grad=True 的参数传给优化器
    active_params = [p for p in model.parameters() if p.requires_grad]
    stage1_lr = 1e-4  
    optimizer = optim.AdamW(active_params, lr=stage1_lr, weight_decay=args.weight_decay)
    
    scaler = GradScaler() # 如果使用 BF16，可以不使用 GradScaler，FP16 则必须使用

    # 7. 数据集加载
    dataset = DynamicAugmentDataset(
        input_pkl='data/processed_data_face_400.pkl', 
        split='train',
        maxlen=2816,       
        num_pc_points=4096 
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        drop_last=True,       
        num_workers=8, # ARM 平台上 worker 过多可能导致内存压力，建议先从 8 开始测试
        pin_memory=True,      
        persistent_workers=True     
    )
    
    dataset1 = DynamicAugmentDataset(
        input_pkl='data/processed_data_face_400.pkl',
        split='val',
        maxlen=2816,
        num_pc_points=4096
    )
    sampler1 = DistributedSampler(dataset1, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader1 = DataLoader(
        dataset1, 
        batch_size=args.batch_size, 
        sampler=sampler1,
        drop_last=False,      
        num_workers=8,        
        pin_memory=True
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.epochs * len(dataloader) // 10, 
        num_training_steps=args.epochs * len(dataloader), 
        lr_start=stage1_lr, 
        lr_end=1e-5
    )

    # 8. 启动训练
    try:
        train_model(
            model, scheduler, optimizer, criterion, dataloader, 
            device, scaler, epochs=args.epochs, rank=rank, 
            startepoc=startepoc, dataloader1=dataloader1
        )
    except Exception as e:
        print(f"Rank {rank} 训练崩溃: {e}")
    finally:
        dist.destroy_process_group()




if __name__ == "__main__":
    world_size = int(sys.argv[1])#torch.cuda.device_count()
    # torch.set_float32_matmul_precision('high')
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
