# Reference Image Flow Matching Model
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
参考图 Flow Matching 模型

使用真实视频作为监督信号，训练参考图相关的新增层：
- clip_proj: CLIP 嵌入投影层 (1280 -> 1536)
- vae_proj: VAE latent 投影层 (16 -> 1536)
- k_vae/v_vae/norm_k_vae: 每个 block 的 VAE 路径交叉注意力

损失函数：Flow Matching Loss
- x_t = (1 - sigma_t) * x0 + sigma_t * noise
- flow_pred = model(x_t, t, context, clip_embeds, vae_latents)
- target = noise - x0
- loss = MSE(flow_pred, target)
"""

from typing import Tuple, List, Optional
import torch
from torch import nn
import torch.distributed as dist
import numpy as np

from utils.wan_wrapper import (
    WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, WanCLIPEncoder,
    _crop_and_resize_pad
)
from utils.loss import get_denoising_loss


class RefImgFlowMatchingModel(nn.Module):
    """参考图 Flow Matching 训练模型

    与 DMD 不同，这里使用真实视频作为监督信号，
    训练参考图相关的新增层。
    """

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        # 初始化模型
        self._initialize_models(args, device)

        # 训练超参数
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)
        self.num_training_frames = getattr(args, "num_training_frames", 21)
        self.num_train_timestep = getattr(args, "num_train_timestep", 1000)

        # 时间步采样范围
        self.min_step = int(getattr(args, "min_step_ratio", 0.02) * self.num_train_timestep)
        self.max_step = int(getattr(args, "max_step_ratio", 0.98) * self.num_train_timestep)

        # 损失函数
        denoising_loss_type = getattr(args, "denoising_loss_type", "flow")
        self.denoising_loss_func = get_denoising_loss(denoising_loss_type)()

        # 要训练的参数模式
        self.trainable_patterns = [
            'clip_proj',      # CLIP 投影层
            'vae_proj',       # VAE 投影层
            'k_vae',          # VAE 路径 Key
            'v_vae',          # VAE 路径 Value
            'norm_k_vae',     # VAE 路径 Key 归一化
        ]

        # 可选：也训练 fused 路径
        if getattr(args, "train_fused_path", False):
            self.trainable_patterns.extend(['k_fused', 'v_fused', 'norm_k_fused'])

    def _initialize_models(self, args, device):
        """初始化模型"""
        # Generator (带参考图支持)
        self.generator = WanDiffusionWrapper(
            model_name=getattr(args, "model_name", "Wan2.1-T2V-1.3B"),
            timestep_shift=getattr(args, "timestep_shift", 5.0),
            is_causal=True,
            local_attn_size=getattr(args.model_kwargs, "local_attn_size", 12),
            sink_size=getattr(args.model_kwargs, "sink_size", 3),
            use_reference_image=True,
            clip_dim=getattr(args, "clip_dim", 1280),
            vae_latent_dim=getattr(args, "vae_latent_dim", 16)
        )

        # Text encoder (冻结)
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        # VAE (冻结)
        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        # CLIP encoder (冻结)
        clip_path = getattr(args, "clip_path", "Skywork/SkyReels-A2")
        self.clip_encoder = WanCLIPEncoder(clip_path)
        self.clip_encoder.requires_grad_(False)

        # 调度器
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        self.scheduler.sigmas = self.scheduler.sigmas.to(device)

    def setup_trainable_params(self):
        """设置可训练参数

        冻结所有参数，只解冻参考图相关的新增层。
        注意：此方法必须在 FSDP 包装之前调用，以便通过原始参数名匹配可训练层。
        """
        # 先冻结所有参数
        for param in self.generator.parameters():
            param.requires_grad = False

        trainable_count = 0
        total_count = 0

        for name, param in self.generator.named_parameters():
            total_count += param.numel()
            if any(pattern in name for pattern in self.trainable_patterns):
                param.requires_grad = True
                trainable_count += param.numel()

        # 安全地检查是否是主进程（处理未初始化的情况）
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if is_main:
            print(f"Trainable parameters: {trainable_count:,} / {total_count:,} "
                  f"({100 * trainable_count / total_count:.2f}%)")
            print(f"Trainable patterns: {self.trainable_patterns}")

        return trainable_count, total_count

    def encode_reference_images(
        self,
        reference_images_batch: List[List],
        pixel_height: int,
        pixel_width: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """编码参考图像

        Args:
            reference_images_batch: List[List[PIL.Image]] - batch of reference image lists
            pixel_height: 像素空间高度
            pixel_width: 像素空间宽度

        Returns:
            clip_embeds: [B, N*257, 1280] or None
            vae_latents: [B, 16, N, H, W] or None
        """
        device = self.device

        # 检查是否所有样本都有参考图
        has_ref = all(len(refs) > 0 for refs in reference_images_batch)
        if not has_ref:
            return None, None

        all_clip_embeds = []
        all_vae_latents = []

        for ref_imgs in reference_images_batch:
            # CLIP 编码
            with torch.no_grad():
                clip_emb = self.clip_encoder(ref_imgs)  # [1, N*257, 1280]
            all_clip_embeds.append(clip_emb)

            # VAE 编码
            vae_latent_list = []
            for img in ref_imgs:
                # 调整图像大小
                img_resized = _crop_and_resize_pad(img, height=pixel_height, width=pixel_width)
                img_tensor = torch.from_numpy(np.array(img_resized)).float() / 127.5 - 1.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
                img_tensor = img_tensor.to(device=device, dtype=self.dtype)

                with torch.no_grad():
                    latent = self.vae.encode_to_latent(img_tensor)  # [1, 1, 16, h, w]
                vae_latent_list.append(latent)

            # 拼接多张参考图的 VAE latent
            vae_latents = torch.cat(vae_latent_list, dim=1)  # [1, N, 16, h, w]
            vae_latents = vae_latents.permute(0, 2, 1, 3, 4)  # [1, 16, N, h, w]
            all_vae_latents.append(vae_latents)

        clip_embeds = torch.cat(all_clip_embeds, dim=0)  # [B, N*257, 1280]
        vae_latents = torch.cat(all_vae_latents, dim=0)  # [B, 16, N, h, w]

        return clip_embeds, vae_latents

    def _get_timestep(
        self,
        batch_size: int,
        num_frames: int
    ) -> torch.Tensor:
        """采样随机时间步

        返回 shape [B, F] 的时间步张量，每个 block 内时间步相同。
        """
        # 采样批次时间步
        timesteps = torch.randint(
            self.min_step,
            self.max_step,
            (batch_size, num_frames),
            device=self.device,
            dtype=torch.long
        )

        # 每个 block 内使用相同时间步
        num_frame_per_block = self.num_frame_per_block
        timesteps = timesteps.reshape(batch_size, -1, num_frame_per_block)
        timesteps[:, :, 1:] = timesteps[:, :, 0:1]
        timesteps = timesteps.reshape(batch_size, -1)

        return timesteps

    def flow_matching_loss(
        self,
        video_frames: torch.Tensor,
        text_prompts: List[str],
        reference_images: List[List],
        pixel_height: int,
        pixel_width: int
    ) -> Tuple[torch.Tensor, dict]:
        """计算 Flow Matching 损失

        Args:
            video_frames: [B, T, C, H, W] 像素空间视频帧，范围 [-1, 1]
            text_prompts: List[str] 文本提示
            reference_images: List[List[PIL.Image]] 参考图像
            pixel_height: 像素空间高度
            pixel_width: 像素空间宽度

        Returns:
            loss: 标量损失
            log_dict: 日志字典
        """
        batch_size = len(text_prompts)
        num_frames_pixel = video_frames.shape[1]
        device = self.device

        # Step 1: 编码视频为 latent (x0)
        # video_frames: [B, T, C, H, W] -> [B, C, T, H, W]
        video_for_vae = video_frames.permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            x0 = self.vae.encode_to_latent(video_for_vae)  # [B, T_latent, 16, h, w]

        # VAE 有 4x 时间压缩，latent 帧数 = (pixel_frames + 3) // 4
        # 例如: 49 帧 -> 13 latent frames, 21 帧 -> 6 latent frames
        num_frames_latent = x0.shape[1]

        # Step 2: 编码文本
        with torch.no_grad():
            conditional_dict = self.text_encoder(text_prompts)
        # 将 text embeddings 转换为列表格式供模型使用
        context_list = [conditional_dict["prompt_embeds"][i] for i in range(batch_size)]

        # Step 3: 编码参考图
        clip_embeds, vae_latents = self.encode_reference_images(
            reference_images, pixel_height, pixel_width
        )

        # Step 4: 采样随机噪声
        noise = torch.randn_like(x0)

        # Step 5: 采样随机时间步 (基于 latent 帧数)
        timesteps = self._get_timestep(batch_size, num_frames_latent)  # [B, F_latent]

        # Step 6: 计算 x_t = (1 - sigma_t) * x0 + sigma_t * noise
        # 获取每帧的 sigma
        sigmas = self.scheduler.sigmas[timesteps]  # [B, F_latent]
        sigmas = sigmas.view(batch_size, num_frames_latent, 1, 1, 1)  # [B, F_latent, 1, 1, 1]

        x_t = (1 - sigmas) * x0 + sigmas * noise

        # Step 7: 准备时间步 (转换为调度器的时间步格式)
        timesteps_for_model = self.scheduler.timesteps[
            (self.num_train_timestep - 1 - timesteps).clamp(0, self.num_train_timestep - 1)
        ]  # [B, F_latent]

        # Step 8: 前向传播
        # 将 x_t 转换为模型期望的格式: [B, F, C, H, W] -> [B, C, F, H, W]
        x_t_for_model = x_t.permute(0, 2, 1, 3, 4)  # [B, 16, F, h, w]

        # 调用训练前向传播 (不需要 kv_cache)
        flow_pred = self.generator.model(
            x_t_for_model,
            t=timesteps_for_model.float(),
            context=context_list,
            seq_len=self.generator.seq_len,
            clip_embeds=clip_embeds,
            vae_latents=vae_latents
        )  # [B, 16, F, h, w]

        # 转回 [B, F, C, H, W] 格式
        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)

        # Step 9: 计算 Flow Matching 损失
        # target = noise - x0 (velocity field)
        target = noise - x0

        loss = self.denoising_loss_func(
            x=x0,
            x_pred=None,
            noise=noise,
            noise_pred=None,
            alphas_cumprod=None,
            timestep=timesteps,
            flow_pred=flow_pred
        )

        # 日志
        log_dict = {
            "flow_loss": loss.detach(),
            "timestep_mean": timesteps.float().mean().detach(),
            "num_frames_pixel": num_frames_pixel,
            "num_frames_latent": num_frames_latent,
        }

        return loss, log_dict
