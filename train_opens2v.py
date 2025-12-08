# OpenS2V Training Entry Point
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
OpenS2V 数据集训练入口

训练目标：
  输入：subject_image + text
  目标：生成的视频，包含参考图中的主体
  数据：is_cross_frame=False
  训练：冻结原模型，只训练新模块
  验证：生成不再是噪声，主体基本能出现

用法：
    # 单机 8 卡
    torchrun --nproc_per_node=8 train_opens2v.py \
        --config_path configs/train_opens2v.yaml \
        --logdir outputs/opens2v_training

    # 多机训练
    torchrun --nnodes=2 --nproc_per_node=8 \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_opens2v.py \
        --config_path configs/train_opens2v.yaml \
        --logdir outputs/opens2v_training

配置说明：
    1. 修改 configs/train_opens2v.yaml 中的数据路径：
       - opens2v.json_paths
       - opens2v.video_base_paths
       - opens2v.background_base_paths

    2. 修改模型检查点路径：
       - generator_ckpt: 包含双路交叉注意力的基础模型

    3. 调整训练参数：
       - num_training_frames: 训练帧数 (建议 21)
       - batch_size: 批次大小 (建议先用 1)
       - max_iters: 最大训练步数
       - lr: 学习率
"""

import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer.opens2v_trainer import OpenS2VTrainer


def main():
    parser = argparse.ArgumentParser(description="OpenS2V Training")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to training config yaml")
    parser.add_argument("--logdir", type=str, default="outputs/opens2v_training",
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
    trainer = OpenS2VTrainer(config)
    trainer.train()

    # 清理
    if not getattr(config, "disable_wandb", True):
        wandb.finish()


if __name__ == "__main__":
    main()
