# Reference Image Flow Matching Training Entry
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
参考图 Flow Matching 训练入口

用法：
    torchrun --nproc_per_node=8 train_refimg.py \
        --config_path configs/train_refimg.yaml \
        --logdir outputs/refimg_training

数据格式 (JSONL)：
    {"video_path": "videos/001.mp4", "prompt": "a woman dancing", "reference_images": ["images/001_ref.jpg"]}
"""

import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer.refimg_trainer import RefImgFlowMatchingTrainer


def main():
    parser = argparse.ArgumentParser(description="Reference Image Flow Matching Training")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to training config yaml")
    parser.add_argument("--logdir", type=str, default="outputs/refimg_training",
                        help="Path to save checkpoints and logs")
    parser.add_argument("--wandb-save-dir", type=str, default="",
                        help="Path to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--no-auto-resume", action="store_true",
                        help="Disable auto resume from latest checkpoint")

    args = parser.parse_args()

    # 加载配置
    config = OmegaConf.load(args.config_path)

    # 尝试加载默认配置并合并
    default_config_path = "configs/default_config.yaml"
    if os.path.exists(default_config_path):
        default_config = OmegaConf.load(default_config_path)
        config = OmegaConf.merge(default_config, config)

    # 命令行参数覆盖
    config_name = os.path.dirname(args.config_path).split("/")[-1]
    if not config_name:
        config_name = os.path.basename(args.config_path).replace(".yaml", "")
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir

    if args.disable_wandb:
        config.disable_wandb = True

    if args.no_auto_resume:
        config.auto_resume = False

    # 确保输出目录存在
    os.makedirs(args.logdir, exist_ok=True)

    # 保存配置副本
    config_save_path = os.path.join(args.logdir, "config.yaml")
    OmegaConf.save(config, config_save_path)

    # 创建训练器并开始训练
    trainer = RefImgFlowMatchingTrainer(config)
    trainer.train()

    # 清理
    if not getattr(config, "disable_wandb", True):
        wandb.finish()


if __name__ == "__main__":
    main()
