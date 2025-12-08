# OpenS2V Reference Image Trainer
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
OpenS2V 数据集的参考图训练器

输入：subject_image + text
目标：生成的视频，包含参考图中的主体
数据：is_cross_frame=False
训练：冻结原模型，只训练新模块
验证：生成不再是噪声，主体基本能出现
"""

import gc
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullOptimStateDictConfig
from omegaconf import OmegaConf
import wandb

from model.refimg_flow_matching import RefImgFlowMatchingModel
from utils.dataset_opens2v import create_opens2v_dataloader, cycle
from utils.distributed import fsdp_wrap, launch_distributed_job
from utils.misc import set_seed

# LoRA
import peft
from peft import get_peft_model_state_dict

from utils.memory import get_cuda_free_memory_gb, log_gpu_memory
from utils.debug_option import DEBUG, LOG_GPU_MEMORY


class OpenS2VTrainer:
    """OpenS2V 数据集的参考图训练器

    使用 OpenS2V 数据集训练参考图相关的新增层。

    训练策略：
    1. 冻结原模型 (text encoder, VAE, generator 主体)
    2. 只训练参考图相关的新增层：
       - clip_proj: CLIP 嵌入投影层
       - vae_proj: VAE latent 投影层
       - k_vae/v_vae/norm_k_vae: VAE 路径交叉注意力
    3. 使用真实视频作为监督信号
    4. Flow Matching 损失
    """

    def __init__(self, config):
        self.config = config
        self.step = 0

        # ========================= Step 1: 分布式环境初始化 =========================
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = getattr(config, "disable_wandb", True)

        # 随机种子
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)

        # WandB
        if self.is_main_process and not self.disable_wandb:
            wandb.login(key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=getattr(config, "wandb_save_dir", "")
            )

        self.output_path = config.logdir

        if self.is_main_process:
            print(f"\n{'=' * 60}")
            print(f"OpenS2V Reference Image Training")
            print(f"{'=' * 60}")
            print(f"World size: {self.world_size}")
            print(f"Device: {self.device}")
            print(f"Dtype: {self.dtype}")
            print(f"Output path: {self.output_path}")
            print(f"is_cross_frame: {getattr(config, 'is_cross_frame', False)}")
            print(f"{'=' * 60}\n")

        # ========================= Step 2: 模型初始化 =========================
        if self.is_main_process:
            print("[1/6] Initializing model...")

        self.model = RefImgFlowMatchingModel(config, device=self.device)

        # 加载基础权重（包含双路交叉注意力的完整权重）
        if config.generator_ckpt:
            if self.is_main_process:
                print(f"  Loading base checkpoint: {config.generator_ckpt}")
            checkpoint = torch.load(config.generator_ckpt, map_location="cpu")
            state_dict = checkpoint.get("generator", checkpoint.get("generator_ema", checkpoint))
            missing, unexpected = self.model.generator.load_state_dict(state_dict, strict=False)
            if self.is_main_process:
                print(f"    Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                if len(missing) > 0:
                    print(f"    Sample missing keys: {missing[:5]}")

        # 加载预训练的 LoRA 权重（可选）
        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path and os.path.exists(lora_ckpt_path):
            if self.is_main_process:
                print(f"\n  Loading pretrained LoRA weights: {lora_ckpt_path}")
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")

            if "generator_lora" in lora_checkpoint:
                lora_state_dict = lora_checkpoint["generator_lora"]
            else:
                lora_state_dict = lora_checkpoint

            missing, unexpected = self.model.generator.load_state_dict(lora_state_dict, strict=False)
            if self.is_main_process:
                loaded_count = len(lora_state_dict) - len(unexpected)
                print(f"    LoRA weights loaded: {loaded_count}, Unexpected: {len(unexpected)}")
        elif lora_ckpt_path:
            if self.is_main_process:
                print(f"\n  Warning: LoRA checkpoint not found: {lora_ckpt_path}")

        # 设置可训练参数（冻结除新增层外的所有参数）
        if self.is_main_process:
            print("\n[2/6] Setting up trainable parameters...")
        self.model.setup_trainable_params()

        # ========================= Step 3: 新 LoRA 配置 (可选) =========================
        self.is_lora_enabled = False
        self.lora_config = None

        if hasattr(config, 'adapter') and config.adapter is not None:
            if self.is_main_process:
                print("\n[3/6] Applying new LoRA for training...")
            self.is_lora_enabled = True
            self.lora_config = config.adapter
            self._apply_lora()
        else:
            if self.is_main_process:
                print("\n[3/6] No new LoRA adapter, training new layers directly...")

        # ========================= Step 4: FSDP 包装 =========================
        if self.is_main_process:
            print("\n[4/6] Applying FSDP...")

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=getattr(config, "sharding_strategy", "hybrid_full"),
            mixed_precision=config.mixed_precision,
            wrap_strategy=getattr(config, "generator_fsdp_wrap_strategy", "size"),
        )

        # 移动其他模型到 GPU
        self.model.vae = self.model.vae.to(device=self.device)
        self.model.clip_encoder = self.model.clip_encoder.to(device=self.device)

        # ========================= Step 5: 优化器初始化 =========================
        if self.is_main_process:
            print("\n[5/6] Initializing optimizer...")

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        if self.is_main_process:
            trainable_params = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
            print(f"  Trainable params: {trainable_params:,}")
            print(f"  Learning rate: {config.lr}")

        # ========================= Step 6: OpenS2V 数据加载器初始化 =========================
        if self.is_main_process:
            print("\n[6/6] Initializing OpenS2V dataloader...")

        # 从配置中读取 OpenS2V 数据集路径
        json_paths = config.opens2v.json_paths
        video_base_paths = config.opens2v.video_base_paths

        # OpenS2V 数据集参数
        dataset_kwargs = {
            "height": config.pixel_height,
            "width": config.pixel_width,
            "sample_num_frames": config.num_training_frames,
            "sample_stride": getattr(config.opens2v, "sample_stride", 3),
            "max_subjects_per_sample": getattr(config.opens2v, "max_subjects_per_sample", 1),
            "subject_selection": getattr(config.opens2v, "subject_selection", "first"),
        }

        # 创建 DataLoader
        raw_dataloader = create_opens2v_dataloader(
            json_paths=json_paths,
            video_base_paths=video_base_paths,
            batch_size=1,  # OpenS2V 先用 batch_size=1 生成样本
            shuffle=True,
            num_workers=getattr(config, "num_workers", 4),
            **dataset_kwargs
        )

        # 包装为分布式采样器（简单起见，直接使用 cycle）
        self.dataloader = cycle(raw_dataloader)

        if self.is_main_process:
            print(f"  OpenS2V dataset initialized")
            print(f"    JSON paths: {json_paths}")
            print(f"    Video paths: {video_base_paths}")
            print(f"    Batch size: {config.batch_size}")
            print(f"    Num frames: {config.num_training_frames}")
            print(f"    Subject selection: {dataset_kwargs['subject_selection']}")

        # 梯度累积
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.max_grad_norm = getattr(config, "max_grad_norm", 1.0)

        # ========================= 检查点加载 =========================
        self._load_checkpoint()

        if self.is_main_process:
            print(f"\n{'=' * 60}")
            print("Initialization complete!")
            print(f"{'=' * 60}\n")

    def _apply_lora(self):
        """应用新的 LoRA 配置到 generator（可选）"""
        from utils.lora_utils import configure_lora_for_model

        self.model.generator.model = configure_lora_for_model(
            self.model.generator.model,
            model_name="generator",
            lora_config=self.lora_config,
            is_main_process=self.is_main_process
        )

    def _load_checkpoint(self):
        """加载训练检查点"""
        auto_resume = getattr(self.config, "auto_resume", True)

        if auto_resume and self.output_path:
            latest_checkpoint = self._find_latest_checkpoint(self.output_path)
            if latest_checkpoint:
                if self.is_main_process:
                    print(f"\nAuto resume: Loading {latest_checkpoint}")
                self._resume_from_checkpoint(latest_checkpoint)
            else:
                if self.is_main_process:
                    print("\nAuto resume: No checkpoint found, starting fresh")

    def _find_latest_checkpoint(self, output_path):
        """查找最新检查点"""
        output_dir = Path(output_path)
        if not output_dir.exists():
            return None

        checkpoints = list(output_dir.glob("checkpoint_opens2v_*/model.pt"))
        if not checkpoints:
            return None

        def get_step(path):
            try:
                return int(path.parent.name.split("_")[-1])
            except:
                return 0

        checkpoints.sort(key=get_step)
        return str(checkpoints[-1])

    def _resume_from_checkpoint(self, checkpoint_path):
        """从检查点恢复"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 加载模型权重
        if self.is_lora_enabled:
            if "generator_lora" in checkpoint:
                peft.set_peft_model_state_dict(
                    self.model.generator.model,
                    checkpoint["generator_lora"]
                )
        else:
            if "generator" in checkpoint:
                self.model.generator.load_state_dict(checkpoint["generator"], strict=False)

        # 加载优化器
        if "optimizer" in checkpoint:
            try:
                opt_state = FSDP.optim_state_dict_to_load(
                    self.model.generator,
                    self.optimizer,
                    checkpoint["optimizer"]
                )
                self.optimizer.load_state_dict(opt_state)
            except:
                if self.is_main_process:
                    print("  Warning: Failed to load optimizer state")

        # 加载步数
        if "step" in checkpoint:
            self.step = checkpoint["step"]
            if self.is_main_process:
                print(f"  Resumed from step {self.step}")

    def fwdbwd_one_step(self, batch):
        """前向/后向传播一步"""
        self.model.generator.train()

        # 获取数据
        text_prompts = batch["prompts"]
        video_frames = batch["video_frames"]  # [B, T, C, H, W], [-1, 1]
        reference_images = batch["reference_images"]  # List[List[PIL.Image]]

        # 移动视频到 GPU
        video_frames = video_frames.to(device=self.device, dtype=self.dtype)

        # 计算 Flow Matching 损失
        loss, log_dict = self.model.flow_matching_loss(
            video_frames=video_frames,
            text_prompts=text_prompts,
            reference_images=reference_images,
            pixel_height=self.config.pixel_height,
            pixel_width=self.config.pixel_width
        )

        # 梯度累积的缩放
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        if LOG_GPU_MEMORY:
            log_gpu_memory("After backward", device=self.device, rank=dist.get_rank())

        log_dict["loss"] = loss.detach()
        return log_dict

    def save(self):
        """保存检查点"""
        if self.is_main_process:
            print(f"\nSaving checkpoint at step {self.step}...")

        if self.is_lora_enabled:
            # 只保存 LoRA 权重
            gen_lora_sd = self._gather_lora_state_dict(self.model.generator.model)
            state_dict = {
                "generator_lora": gen_lora_sd,
                "step": self.step,
            }
        else:
            # 保存完整模型
            with FSDP.state_dict_type(
                self.model.generator,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True),
            ):
                generator_state_dict = self.model.generator.state_dict()
                optimizer_state_dict = FSDP.optim_state_dict(
                    self.model.generator, self.optimizer
                )

            state_dict = {
                "generator": generator_state_dict,
                "optimizer": optimizer_state_dict,
                "step": self.step,
            }

        if self.is_main_process:
            checkpoint_dir = os.path.join(
                self.output_path, f"checkpoint_opens2v_{self.step:06d}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_file)
            print(f"  Saved: {checkpoint_file}")

            # 清理旧检查点
            max_checkpoints = getattr(self.config, "max_checkpoints", 5)
            if max_checkpoints > 0:
                self._cleanup_old_checkpoints(self.output_path, max_checkpoints)

        torch.cuda.empty_cache()
        gc.collect()

    def _gather_lora_state_dict(self, lora_model):
        """收集 LoRA state dict"""
        with FSDP.state_dict_type(
            lora_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        ):
            full_state = lora_model.state_dict()
        return get_peft_model_state_dict(lora_model, state_dict=full_state)

    def _cleanup_old_checkpoints(self, output_path, max_checkpoints):
        """清理旧检查点"""
        output_dir = Path(output_path)
        checkpoints = sorted(output_dir.glob("checkpoint_opens2v_*"))

        if len(checkpoints) > max_checkpoints:
            for old_ckpt in checkpoints[:-max_checkpoints]:
                import shutil
                shutil.rmtree(old_ckpt)
                print(f"  Removed old checkpoint: {old_ckpt}")

    def train(self):
        """主训练循环"""
        config = self.config

        if self.is_main_process:
            print(f"\n{'=' * 60}")
            print("Starting OpenS2V training...")
            print(f"  Max iterations: {config.max_iters}")
            print(f"  Save interval: {config.log_iters}")
            print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
            print(f"{'=' * 60}\n")

        accumulated_logs = []

        while self.step < config.max_iters:
            # 梯度累积循环
            for accumulation_step in range(self.gradient_accumulation_steps):
                batch = next(self.dataloader)
                log_dict = self.fwdbwd_one_step(batch)
                accumulated_logs.append(log_dict)

            # 梯度裁剪
            if self.max_grad_norm > 0:
                grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
            else:
                grad_norm = torch.tensor(0.0)

            # 优化器更新
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 步数递增
            self.step += 1

            # 日志
            if self.step % 10 == 0 and self.is_main_process:
                avg_loss = sum(d["loss"].item() for d in accumulated_logs) / len(accumulated_logs)
                print(f"Step {self.step}/{config.max_iters} | Loss: {avg_loss:.4f} | Grad norm: {grad_norm:.2f}")

                if not self.disable_wandb:
                    wandb.log({
                        "loss": avg_loss,
                        "grad_norm": grad_norm,
                        "step": self.step
                    })

            accumulated_logs = []

            # 保存检查点
            if self.step % config.log_iters == 0:
                self.save()

            # 垃圾回收
            gc_interval = getattr(config, "gc_interval", 100)
            if self.step % gc_interval == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # 最终保存
        self.save()

        if self.is_main_process:
            print(f"\n{'=' * 60}")
            print("Training completed!")
            print(f"{'=' * 60}")
