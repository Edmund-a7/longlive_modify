# Copyright 2024-2025 Multi-Subject Consistency Module for LongLive
# Training-free subject identity injection via cross-attention maps

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple


class SubjectFeatureBank:
    """
    Training-free 的参考图特征银行
    存储每个主体的 VAE 特征和对应的文本 token 索引
    """

    def __init__(self, vae_wrapper, patch_embedding: nn.Module, device: torch.device):
        """
        Args:
            vae_wrapper: WanVAEWrapper 实例，用于编码参考图像
            patch_embedding: 模型的 patch_embedding 层，用于投影到模型维度
            device: 计算设备
        """
        self.vae = vae_wrapper
        self.patch_embedding = patch_embedding
        self.device = device
        self.bank: Dict[str, Dict] = {}

    def add_subject(
        self,
        name: str,
        ref_image: torch.Tensor,
        text_token_idx: int,
        text_token_indices: Optional[List[int]] = None
    ):
        """
        添加一个主体到特征银行

        Args:
            name: 主体名称（如 "man", "dog"）
            ref_image: [1, 3, H, W] 或 [3, H, W] 参考图像（RGB，-1到1范围）
            text_token_idx: 该主体在文本中对应的主要 token 索引
            text_token_indices: 可选，该主体对应的所有 token 索引列表
        """
        if ref_image.dim() == 3:
            ref_image = ref_image.unsqueeze(0)  # [1, 3, H, W]

        ref_image = ref_image.to(self.device)

        with torch.no_grad():
            # VAE 编码
            # WanVAEWrapper.encode_to_latent 期望 [B, C, T, H, W]
            ref_video = ref_image.unsqueeze(2)  # [1, 3, 1, H, W]

            # 编码得到潜空间表示
            latent = self.vae.encode_to_latent(ref_video)  # [1, 1, 16, H', W']

            # 调整维度用于 patch_embedding
            # patch_embedding 期望 [B, C, T, H, W]
            latent_for_patch = latent.permute(0, 2, 1, 3, 4)  # [1, 16, 1, H', W']

            # 通过 patch embedding 投影到模型维度
            features = self.patch_embedding(latent_for_patch)  # [1, dim, 1, H'', W'']
            features = features.flatten(2).transpose(1, 2)  # [1, L, dim]

            # 计算全局特征（用于注入）
            global_feature = features.mean(dim=1, keepdim=True)  # [1, 1, dim]

            self.bank[name] = {
                "features": features,  # [1, L, 2048]
                "token_idx": text_token_idx,
                "token_indices": text_token_indices or [text_token_idx],
                "global_feature": global_feature  # [1, 1, 2048]
            }

    def get_subject(self, name: str) -> Optional[Dict]:
        """获取指定主体的特征"""
        return self.bank.get(name)

    def get_all_subjects(self) -> Dict:
        """获取所有主体的特征"""
        return self.bank

    def clear(self):
        """清空特征银行"""
        self.bank.clear()


def attention_with_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    标准注意力计算，返回 attention scores 和输出

    Args:
        q: [B, L_q, num_heads, head_dim]
        k: [B, L_k, num_heads, head_dim]
        v: [B, L_k, num_heads, head_dim]
        scale: 可选的缩放因子

    Returns:
        output: [B, L_q, num_heads, head_dim] 注意力输出
        attn_scores_avg: [B, L_q, L_k] head 平均的注意力分数
    """
    B, L_q, num_heads, head_dim = q.shape
    L_k = k.shape[1]

    if scale is None:
        scale = head_dim ** -0.5

    # 转置为 [B, num_heads, L_q/L_k, head_dim]
    q = q.transpose(1, 2)  # [B, num_heads, L_q, head_dim]
    k = k.transpose(1, 2)  # [B, num_heads, L_k, head_dim]
    v = v.transpose(1, 2)  # [B, num_heads, L_k, head_dim]

    # 计算注意力分数
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, L_q, L_k]
    attn_probs = F.softmax(attn_scores, dim=-1)  # [B, num_heads, L_q, L_k]

    # 计算注意力输出
    output = torch.matmul(attn_probs, v)  # [B, num_heads, L_q, head_dim]
    output = output.transpose(1, 2)  # [B, L_q, num_heads, head_dim]

    # 返回 head 平均的 attention scores 用于定位
    attn_scores_avg = attn_probs.mean(dim=1)  # [B, L_q, L_k]

    return output, attn_scores_avg


def inject_subject_features(
    x_out: torch.Tensor,
    attn_scores: torch.Tensor,
    subject_bank: Dict,
    grid_sizes: torch.Tensor,
    injection_strength: float = 0.3,
    threshold: float = 0.1
) -> torch.Tensor:
    """
    根据注意力图将参考图特征注入到视频 token

    Args:
        x_out: 原始 cross-attention 输出 [B, L, dim]
        attn_scores: 注意力分数 [B, L_video, L_text]
        subject_bank: 主体特征银行 {"name": {"token_idx": int, "global_feature": tensor}}
        grid_sizes: [B, 3] = [T, H, W] 时空维度信息
        injection_strength: 注入强度 λ，范围 0-1
        threshold: 注意力阈值，低于此值不注入

    Returns:
        result: 注入后的特征 [B, L, dim]
    """
    B, L, dim = x_out.shape

    # grid_sizes 可能是 tensor，提取值
    if isinstance(grid_sizes, torch.Tensor):
        T, H, W = grid_sizes[0].tolist()
    else:
        T, H, W = grid_sizes

    result = x_out.clone()

    for name, subject_info in subject_bank.items():
        token_idx = subject_info["token_idx"]
        token_indices = subject_info.get("token_indices", [token_idx])
        global_feat = subject_info["global_feature"]  # [1, 1, dim]

        # 提取该主体对应的注意力分数
        # 如果有多个 token，取平均
        if len(token_indices) > 1:
            alpha = torch.stack([attn_scores[:, :, idx] for idx in token_indices], dim=-1)
            alpha = alpha.mean(dim=-1)  # [B, L_video]
        else:
            alpha = attn_scores[:, :, token_idx]  # [B, L_video]

        # 归一化到 0-1
        alpha_min = alpha.min(dim=-1, keepdim=True)[0]
        alpha_max = alpha.max(dim=-1, keepdim=True)[0]
        alpha_norm = (alpha - alpha_min) / (alpha_max - alpha_min + 1e-8)

        # 应用阈值，只在高注意力区域注入
        mask = (alpha_norm > threshold).float()
        alpha_masked = alpha_norm * mask

        # 扩展维度以便广播
        alpha_masked = alpha_masked.unsqueeze(-1)  # [B, L, 1]

        # 确保 global_feat 与 result 在同一设备
        global_feat = global_feat.to(result.device)

        # 混合注入
        # result = (1 - λ*α) * result + λ*α * global_feat
        global_feat_expanded = global_feat.expand(B, L, dim)
        result = result * (1 - injection_strength * alpha_masked) + \
                 global_feat_expanded * (injection_strength * alpha_masked)

    return result


def compute_subject_mask(
    attn_scores: torch.Tensor,
    token_idx: int,
    grid_sizes: Tuple[int, int, int],
    threshold: float = 0.1
) -> torch.Tensor:
    """
    从注意力分数计算主体的空间 mask

    Args:
        attn_scores: [B, L_video, L_text] 注意力分数
        token_idx: 主体对应的文本 token 索引
        grid_sizes: (T, H, W) 时空维度
        threshold: 阈值

    Returns:
        mask: [B, T, H, W] 归一化的空间 mask
    """
    B = attn_scores.shape[0]
    T, H, W = grid_sizes

    # 提取该 token 对应的注意力分数
    alpha = attn_scores[:, :, token_idx]  # [B, L_video]

    # 归一化
    alpha_min = alpha.min(dim=-1, keepdim=True)[0]
    alpha_max = alpha.max(dim=-1, keepdim=True)[0]
    alpha_norm = (alpha - alpha_min) / (alpha_max - alpha_min + 1e-8)

    # reshape 到时空维度
    mask = alpha_norm.view(B, T, H, W)

    # 应用阈值
    mask = (mask > threshold).float() * mask

    return mask
