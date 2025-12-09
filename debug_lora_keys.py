#!/usr/bin/env python3
"""
诊断 LoRA 权重加载失败问题
运行: python debug_lora_keys.py
"""

import torch
import sys
sys.path.insert(0, '.')

def main():
    lora_path = './longlive_models/models/lora_dual.pt'

    print("=" * 60)
    print("LoRA 权重加载诊断")
    print("=" * 60)

    # 1. 加载 LoRA 检查点
    print(f"\n[1] 加载 LoRA 检查点: {lora_path}")
    try:
        lora_ckpt = torch.load(lora_path, map_location='cpu')
    except FileNotFoundError:
        print(f"  错误: 文件不存在 {lora_path}")
        return
    except Exception as e:
        print(f"  加载错误: {e}")
        return

    print(f"  顶层 keys: {list(lora_ckpt.keys())}")

    # 获取实际的 state dict
    if 'generator_lora' in lora_ckpt:
        lora_sd = lora_ckpt['generator_lora']
        print("  使用 'generator_lora' 作为 state dict")
    else:
        lora_sd = lora_ckpt
        print("  直接使用检查点作为 state dict")

    lora_keys = list(lora_sd.keys())
    print(f"  LoRA state dict 共 {len(lora_keys)} 个 keys")

    # 2. 分析 LoRA key 的模式
    print(f"\n[2] 分析 LoRA key 模式")
    print("  前 15 个 keys:")
    for key in lora_keys[:15]:
        shape = lora_sd[key].shape if hasattr(lora_sd[key], 'shape') else 'N/A'
        print(f"    {key} -> {shape}")

    # 检查是否是 PEFT 格式
    has_lora_a = any('lora_A' in k or 'lora_a' in k for k in lora_keys)
    has_lora_b = any('lora_B' in k or 'lora_b' in k for k in lora_keys)
    has_base_model = any('base_model' in k for k in lora_keys)

    print(f"\n  包含 'lora_A/lora_a': {has_lora_a}")
    print(f"  包含 'lora_B/lora_b': {has_lora_b}")
    print(f"  包含 'base_model': {has_base_model}")

    # 3. 加载模型查看 key 结构
    print(f"\n[3] 加载模型查看 key 结构")
    try:
        from utils.wan_wrapper import WanDiffusionWrapper

        # 初始化模型（不加载权重）
        generator = WanDiffusionWrapper(
            model_name="Wan2.1-T2V-1.3B",
            is_causal=True,
            local_attn_size=12,
            sink_size=3,
            use_reference_image=True,
        )

        model_keys = list(generator.state_dict().keys())
        print(f"  模型 state dict 共 {len(model_keys)} 个 keys")
        print("  前 15 个 keys:")
        for key in model_keys[:15]:
            print(f"    {key}")

    except Exception as e:
        print(f"  加载模型失败: {e}")
        model_keys = []

    # 4. 找出不匹配的原因
    print(f"\n[4] 分析 key 不匹配原因")

    if model_keys:
        # 检查前缀差异
        lora_prefixes = set()
        model_prefixes = set()

        for k in lora_keys[:50]:
            parts = k.split('.')
            if len(parts) >= 2:
                lora_prefixes.add('.'.join(parts[:2]))

        for k in model_keys[:50]:
            parts = k.split('.')
            if len(parts) >= 2:
                model_prefixes.add('.'.join(parts[:2]))

        print(f"  LoRA key 前缀 (前两级): {sorted(lora_prefixes)[:10]}")
        print(f"  模型 key 前缀 (前两级): {sorted(model_prefixes)[:10]}")

        # 检查是否需要添加/移除前缀
        sample_lora_key = lora_keys[0] if lora_keys else ''
        sample_model_key = model_keys[0] if model_keys else ''

        print(f"\n  示例 LoRA key: {sample_lora_key}")
        print(f"  示例模型 key: {sample_model_key}")

        # 尝试找到转换规则
        print("\n[5] 建议的修复方案:")

        if has_base_model and not any('base_model' in k for k in model_keys):
            print("  - LoRA keys 包含 'base_model' 前缀，但模型没有")
            print("  - 解决方案: 移除 'base_model.model.' 前缀")
            print("    lora_sd = {k.replace('base_model.model.', ''): v for k, v in lora_sd.items()}")

        if not has_base_model and 'model.' in sample_model_key and 'model.' not in sample_lora_key:
            print("  - 模型 keys 有 'model.' 前缀，LoRA keys 没有")
            print("  - 解决方案: 添加 'model.' 前缀")
            print("    lora_sd = {'model.' + k: v for k, v in lora_sd.items()}")

        if has_lora_a or has_lora_b:
            print("  - 检测到 PEFT LoRA 格式")
            print("  - 这些是 LoRA adapter 权重，不能直接 load_state_dict")
            print("  - 需要使用 peft 库的 set_peft_model_state_dict 或类似方法")

    # 5. 尝试自动修复
    print("\n[6] 尝试自动匹配")

    def try_match(lora_sd, model_keys):
        """尝试不同的 key 转换方案"""
        transformations = [
            ("原始", lambda k: k),
            ("移除 'base_model.model.'", lambda k: k.replace('base_model.model.', '')),
            ("移除 'base_model.'", lambda k: k.replace('base_model.', '')),
            ("添加 'model.'", lambda k: 'model.' + k),
            ("移除 'model.'", lambda k: k.replace('model.', '', 1) if k.startswith('model.') else k),
        ]

        model_key_set = set(model_keys)

        for name, transform in transformations:
            transformed_keys = [transform(k) for k in lora_sd.keys()]
            matches = sum(1 for k in transformed_keys if k in model_key_set)
            print(f"  {name}: {matches}/{len(lora_sd)} 匹配")

            if matches > 0:
                # 显示一些匹配的例子
                matched_examples = [k for k in transformed_keys if k in model_key_set][:3]
                if matched_examples:
                    print(f"    匹配示例: {matched_examples}")

    if model_keys:
        try_match(lora_sd, model_keys)

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
