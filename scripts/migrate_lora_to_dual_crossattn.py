#!/usr/bin/env python3
"""
LoRA 权重迁移脚本：将单路交叉注意力 LoRA 权重迁移到双路交叉注意力

功能：
1. 加载原始 LoRA checkpoint (使用单路 cross_attn.k/v)
2. 将 cross_attn.k/v 的 LoRA 权重复制到 cross_attn.k_fused/v_fused
3. 新增的 VAE 路径 (k_vae/v_vae) 的 LoRA 保持零初始化（不改变基础权重行为）
4. 保存新的 LoRA checkpoint

使用方法：
    python scripts/migrate_lora_to_dual_crossattn.py \
        --input_lora longlive_models/models/lora.pt \
        --output_lora longlive_models/models/lora_dual.pt
"""

import argparse
import torch
from collections import OrderedDict


def migrate_lora_checkpoint(input_lora: str, output_lora: str):
    """迁移 LoRA checkpoint 权重

    映射规则：
    - cross_attn.k.lora_A/B -> cross_attn.k_fused.lora_A/B
    - cross_attn.v.lora_A/B -> cross_attn.v_fused.lora_A/B
    - 其他权重保持不变
    """
    print(f"Loading LoRA checkpoint from {input_lora}...")
    checkpoint = torch.load(input_lora, map_location='cpu')

    # 处理不同格式的 checkpoint
    if isinstance(checkpoint, dict) and 'generator_lora' in checkpoint:
        old_state_dict = checkpoint['generator_lora']
        has_wrapper = True
    else:
        old_state_dict = checkpoint
        has_wrapper = False

    print(f"Original LoRA keys: {len(old_state_dict)}")

    new_state_dict = OrderedDict()
    migrated_count = 0

    for key, value in old_state_dict.items():
        new_key = key

        # 迁移 cross_attn.k -> cross_attn.k_fused
        if '.cross_attn.k.lora_' in key:
            new_key = key.replace('.cross_attn.k.lora_', '.cross_attn.k_fused.lora_')
            migrated_count += 1
        # 迁移 cross_attn.v -> cross_attn.v_fused
        elif '.cross_attn.v.lora_' in key:
            new_key = key.replace('.cross_attn.v.lora_', '.cross_attn.v_fused.lora_')
            migrated_count += 1

        new_state_dict[new_key] = value

        if new_key != key:
            print(f"  {key} -> {new_key}")

    print(f"\nMigrated {migrated_count} LoRA weight keys")
    print(f"New LoRA keys: {len(new_state_dict)}")

    # 保存
    if has_wrapper:
        output_checkpoint = {'generator_lora': new_state_dict}
    else:
        output_checkpoint = new_state_dict

    torch.save(output_checkpoint, output_lora)
    print(f"\nSaved migrated LoRA checkpoint to {output_lora}")

    # 验证
    print("\nSample migrated keys:")
    for k in list(new_state_dict.keys())[:10]:
        print(f"  {k}")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate single cross-attn LoRA weights to dual cross-attn'
    )
    parser.add_argument(
        '--input_lora', type=str, required=True,
        help='Input LoRA checkpoint path'
    )
    parser.add_argument(
        '--output_lora', type=str, required=True,
        help='Output LoRA checkpoint path'
    )

    args = parser.parse_args()
    migrate_lora_checkpoint(args.input_lora, args.output_lora)


if __name__ == '__main__':
    main()
