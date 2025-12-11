#!/usr/bin/env python3
"""
权重迁移脚本：将单路交叉注意力权重迁移到双路交叉注意力

功能：
1. 加载原始 checkpoint (使用单路 WanT2VCrossAttention)
2. 将 cross_attn.k/v/norm_k 权重复制到 cross_attn.k_fused/v_fused/norm_k_fused
3. 新增的 VAE 路径 (k_vae/v_vae/norm_k_vae) 和投影层 (clip_proj/vae_proj) 使用 Xavier 初始化
4. 保存新的 checkpoint

使用方法：
    python scripts/migrate_weights_to_dual_crossattn.py \
        --input_ckpt longlive_models/models/longlive_base.pt \
        --output_ckpt longlive_models/models/longlive_base_dual.pt \
        --num_blocks 30 \
        --dim 1536 \
        --clip_dim 1280 \
        --vae_latent_dim 16
"""

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path


def create_mlp_proj_weights(in_dim: int, out_dim: int, dtype=torch.float32, zero_output: bool = True):
    """创建 MLP 投影层的初始化权重 (CLIPProjMLP / VAEProjMLP 结构)

    结构: LayerNorm -> Linear -> GELU -> Linear -> LayerNorm
    proj.0: LayerNorm(in_dim)
    proj.1: Linear(in_dim, out_dim)
    proj.2: GELU (无参数)
    proj.3: Linear(out_dim, out_dim)
    proj.4: LayerNorm(out_dim)

    Args:
        zero_output: 如果为 True，将最后一个 Linear 层 (proj.3) 权重设为 0，
                     确保未训练时 MLP 输出为零，不影响原始模型
    """
    weights = {}

    # proj.0: LayerNorm(in_dim)
    weights['proj.0.weight'] = torch.ones(in_dim, dtype=dtype)
    weights['proj.0.bias'] = torch.zeros(in_dim, dtype=dtype)

    # proj.1: Linear(in_dim, out_dim)
    w1 = torch.empty(out_dim, in_dim, dtype=dtype)
    nn.init.xavier_uniform_(w1)
    weights['proj.1.weight'] = w1
    weights['proj.1.bias'] = torch.zeros(out_dim, dtype=dtype)

    # proj.2: GELU - 无参数

    # proj.3: Linear(out_dim, out_dim)
    if zero_output:
        # 零初始化: 确保未训练时输出为 0
        weights['proj.3.weight'] = torch.zeros(out_dim, out_dim, dtype=dtype)
    else:
        w3 = torch.empty(out_dim, out_dim, dtype=dtype)
        nn.init.xavier_uniform_(w3)
        weights['proj.3.weight'] = w3
    weights['proj.3.bias'] = torch.zeros(out_dim, dtype=dtype)

    # proj.4: LayerNorm(out_dim)
    weights['proj.4.weight'] = torch.ones(out_dim, dtype=dtype)
    weights['proj.4.bias'] = torch.zeros(out_dim, dtype=dtype)

    return weights


def create_dual_crossattn_weights(dim: int, dtype=torch.float32, zero_v_vae: bool = True):
    """创建双路交叉注意力层的 VAE 路径权重

    需要初始化:
    - k_vae: Linear(dim, dim)
    - v_vae: Linear(dim, dim)
    - norm_k_vae: WanRMSNorm(dim) - 只有 weight，无 bias

    Args:
        zero_v_vae: 如果为 True，将 v_vae 权重设为 0，
                    确保未训练时 VAE 路径输出为零，不影响原始模型
    """
    weights = {}

    # k_vae
    k_vae_w = torch.empty(dim, dim, dtype=dtype)
    nn.init.xavier_uniform_(k_vae_w)
    weights['k_vae.weight'] = k_vae_w
    weights['k_vae.bias'] = torch.zeros(dim, dtype=dtype)

    # v_vae
    if zero_v_vae:
        # 零初始化: 确保未训练时 VAE 路径贡献为 0
        weights['v_vae.weight'] = torch.zeros(dim, dim, dtype=dtype)
    else:
        v_vae_w = torch.empty(dim, dim, dtype=dtype)
        nn.init.xavier_uniform_(v_vae_w)
        weights['v_vae.weight'] = v_vae_w
    weights['v_vae.bias'] = torch.zeros(dim, dtype=dtype)

    # norm_k_vae (WanRMSNorm 只有 weight)
    weights['norm_k_vae.weight'] = torch.ones(dim, dtype=dtype)

    return weights


def migrate_checkpoint(
    input_ckpt: str,
    output_ckpt: str,
    num_blocks: int = 30,
    dim: int = 1536,
    clip_dim: int = 1280,
    vae_latent_dim: int = 16,
    use_ema: bool = False
):
    """迁移 checkpoint 权重

    Args:
        input_ckpt: 输入 checkpoint 路径
        output_ckpt: 输出 checkpoint 路径
        num_blocks: Transformer block 数量
        dim: 模型隐藏层维度
        clip_dim: CLIP 嵌入维度
        vae_latent_dim: VAE latent 维度
        use_ema: 是否使用 EMA 权重
    """
    print(f"Loading checkpoint from {input_ckpt}...")
    checkpoint = torch.load(input_ckpt, map_location='cpu')

    # 获取原始 state_dict
    if 'generator' in checkpoint or 'generator_ema' in checkpoint:
        key = 'generator_ema' if use_ema and 'generator_ema' in checkpoint else 'generator'
        old_state_dict = checkpoint[key]
    elif 'model' in checkpoint:
        old_state_dict = checkpoint['model']
    else:
        old_state_dict = checkpoint

    # 获取 dtype
    sample_tensor = next(iter(old_state_dict.values()))
    dtype = sample_tensor.dtype
    print(f"Using dtype: {dtype}")

    new_state_dict = OrderedDict()
    migrated_count = 0
    new_init_count = 0

    # 1. 复制所有非 cross_attn 的权重
    for key, value in old_state_dict.items():
        if 'cross_attn' not in key:
            new_state_dict[key] = value
        else:
            # 处理 cross_attn 相关的权重
            # 原始: model.blocks.X.cross_attn.{k,v,norm_k,q,o}.{weight,bias}
            # 目标: model.blocks.X.cross_attn.{k_fused,v_fused,norm_k_fused,q,o}.{weight,bias}

            if '.cross_attn.k.' in key:
                # k -> k_fused
                new_key = key.replace('.cross_attn.k.', '.cross_attn.k_fused.')
                new_state_dict[new_key] = value
                migrated_count += 1
            elif '.cross_attn.v.' in key:
                # v -> v_fused
                new_key = key.replace('.cross_attn.v.', '.cross_attn.v_fused.')
                new_state_dict[new_key] = value
                migrated_count += 1
            elif '.cross_attn.norm_k.' in key:
                # norm_k -> norm_k_fused
                new_key = key.replace('.cross_attn.norm_k.', '.cross_attn.norm_k_fused.')
                new_state_dict[new_key] = value
                migrated_count += 1
            else:
                # q, o, norm_q 保持不变
                new_state_dict[key] = value

    print(f"Migrated {migrated_count} cross_attn weights to fused path")

    # 2. 添加 clip_proj 权重
    print(f"Initializing clip_proj (in={clip_dim}, out={dim})...")
    clip_proj_weights = create_mlp_proj_weights(clip_dim, dim, dtype=dtype)
    for k, v in clip_proj_weights.items():
        new_state_dict[f'model.clip_proj.{k}'] = v
        new_init_count += 1

    # 3. 添加 vae_proj 权重
    print(f"Initializing vae_proj (in={vae_latent_dim}, out={dim})...")
    vae_proj_weights = create_mlp_proj_weights(vae_latent_dim, dim, dtype=dtype)
    for k, v in vae_proj_weights.items():
        new_state_dict[f'model.vae_proj.{k}'] = v
        new_init_count += 1

    # 4. 为每个 block 添加 VAE 路径的权重
    print(f"Initializing VAE cross-attn weights for {num_blocks} blocks...")
    for block_idx in range(num_blocks):
        vae_weights = create_dual_crossattn_weights(dim, dtype=dtype)
        for k, v in vae_weights.items():
            new_state_dict[f'model.blocks.{block_idx}.cross_attn.{k}'] = v
            new_init_count += 1

    print(f"Newly initialized {new_init_count} parameters")

    # 5. 保存新的 checkpoint
    new_checkpoint = {
        'generator': new_state_dict,
    }

    # 如果原始 checkpoint 有其他键，也复制过来
    for key in checkpoint:
        if key not in ['generator', 'generator_ema', 'model']:
            new_checkpoint[key] = checkpoint[key]

    output_path = Path(output_ckpt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving migrated checkpoint to {output_ckpt}...")
    torch.save(new_checkpoint, output_ckpt)

    # 打印统计信息
    print("\n" + "=" * 60)
    print("Migration Summary:")
    print("=" * 60)
    print(f"  Original parameters: {len(old_state_dict)}")
    print(f"  New parameters: {len(new_state_dict)}")
    print(f"  Migrated (k/v/norm_k -> k_fused/v_fused/norm_k_fused): {migrated_count}")
    print(f"  Newly initialized: {new_init_count}")
    print(f"    - clip_proj: 8 tensors")
    print(f"    - vae_proj: 8 tensors")
    print(f"    - VAE cross-attn per block: 5 tensors x {num_blocks} blocks = {5 * num_blocks} tensors")
    print("=" * 60)
    print(f"Output saved to: {output_ckpt}")
    print("\nNote: The fused path (k_fused/v_fused) now has pretrained weights.")
    print("      The VAE path (k_vae/v_vae) and projections need training.")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate single cross-attn weights to dual cross-attn'
    )
    parser.add_argument(
        '--input_ckpt', type=str, required=True,
        help='Input checkpoint path (with single cross-attn)'
    )
    parser.add_argument(
        '--output_ckpt', type=str, required=True,
        help='Output checkpoint path (with dual cross-attn)'
    )
    parser.add_argument(
        '--num_blocks', type=int, default=30,
        help='Number of transformer blocks (default: 30 for Wan2.1-T2V-1.3B)'
    )
    parser.add_argument(
        '--dim', type=int, default=1536,
        help='Model hidden dimension (default: 1536 for Wan2.1-T2V-1.3B)'
    )
    parser.add_argument(
        '--clip_dim', type=int, default=1280,
        help='CLIP embedding dimension (default: 1280)'
    )
    parser.add_argument(
        '--vae_latent_dim', type=int, default=16,
        help='VAE latent dimension (default: 16)'
    )
    parser.add_argument(
        '--use_ema', action='store_true',
        help='Use EMA weights if available'
    )

    args = parser.parse_args()

    migrate_checkpoint(
        input_ckpt=args.input_ckpt,
        output_ckpt=args.output_ckpt,
        num_blocks=args.num_blocks,
        dim=args.dim,
        clip_dim=args.clip_dim,
        vae_latent_dim=args.vae_latent_dim,
        use_ema=args.use_ema
    )


if __name__ == '__main__':
    main()
