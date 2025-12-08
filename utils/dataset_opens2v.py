# OpenS2V Dataset Adapter for longlive_modify
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
OpenS2V 数据集适配器

将 OpenS2V-Nexus 格式的数据转换为 longlive_modify 训练所需的格式。

输入：subject_image + text
目标：生成的视频，包含参考图中的主体
数据：is_cross_frame=False
训练：冻结原模型，只训练新模块
"""

import os
import sys
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from torch.utils.data import Dataset

# 添加 OpenS2V-Nexus 的路径
OPENS2V_PATH = Path(__file__).parent.parent.parent / "OpenS2V-Nexus" / "data_process"
if str(OPENS2V_PATH) not in sys.path:
    sys.path.insert(0, str(OPENS2V_PATH))

from demo_dataloader import OpenS2VDataset, MultiPartOpenS2VDataset


class OpenS2VAdapterDataset(Dataset):
    """
    OpenS2V 数据集适配器

    将 OpenS2V 数据格式转换为 longlive_modify 的训练格式：
    - subject_image + text -> video

    Args:
        json_path: OpenS2V JSON 文件路径
        video_base_path: 视频基础路径
        background_base_path: 背景图片基础路径
        is_cross_frame: 是否使用跨帧数据 (默认 False)
        height: 视频高度 (默认 480)
        width: 视频宽度 (默认 832)
        sample_num_frames: 采样帧数 (默认 49)
        sample_stride: 采样步长 (默认 3)
        max_subjects_per_sample: 每个样本最多使用多少个 subject (默认 1)
        subject_selection: subject 选择策略 ('first', 'random', 'best_score')
    """

    def __init__(
        self,
        json_path: str,
        video_base_path: str,
        background_base_path: str,
        cross_frames_cluster_path: str = "",
        cross_frames_base_path: str = "",
        is_cross_frame: bool = False,
        height: int = 480,
        width: int = 832,
        sample_num_frames: int = 49,
        sample_stride: int = 3,
        max_subjects_per_sample: int = 1,
        subject_selection: str = "first",  # 'first', 'random', 'best_score'
    ):
        self.max_subjects = max_subjects_per_sample
        self.subject_selection = subject_selection

        # 创建底层的 OpenS2V 数据集
        self.opens2v_dataset = OpenS2VDataset(
            json_path=json_path,
            video_base_path=video_base_path,
            background_base_path=background_base_path,
            cross_frames_cluster_path=cross_frames_cluster_path or "dummy_path",
            cross_frames_base_path=cross_frames_base_path or "dummy_path",
            is_cross_frame=is_cross_frame,
            height=height,
            width=width,
            sample_num_frames=sample_num_frames,
            sample_stride=sample_stride,
        )

    def __len__(self):
        return len(self.opens2v_dataset)

    def _select_subjects(self, subjects: List[Dict]) -> List[Dict]:
        """
        从 subjects 列表中选择要使用的 subject

        Args:
            subjects: OpenS2V 返回的 subjects 列表

        Returns:
            选中的 subjects 列表
        """
        if len(subjects) == 0:
            return []

        if self.subject_selection == "first":
            # 选择第一个
            return subjects[:self.max_subjects]

        elif self.subject_selection == "random":
            # 随机选择
            num_to_select = min(self.max_subjects, len(subjects))
            return random.sample(subjects, num_to_select)

        elif self.subject_selection == "best_score":
            # 根据分数排序后选择
            # 使用 aes_score (美学分数) 作为排序依据
            sorted_subjects = sorted(
                subjects,
                key=lambda x: x.get("subject_aes_score", 0.0),
                reverse=True
            )
            return sorted_subjects[:self.max_subjects]

        else:
            raise ValueError(f"Unknown subject_selection: {self.subject_selection}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回适配后的数据格式

        Returns:
            {
                "prompts": str,  # 文本描述
                "video_frames": Tensor,  # [T, C, H, W], 范围 [0, 255] uint8
                "reference_images": List[PIL.Image],  # 参考图列表
                "idx": int,  # 索引
                "metadata": dict,  # 元数据 (可选)
            }
        """
        # 获取 OpenS2V 原始数据
        data = self.opens2v_dataset[idx]

        # 提取视频帧 (已经是 [T, C, H, W] 格式，范围 [0, 255])
        video_frames = data["video"]  # Tensor[T, C, H, W]

        # 提取文本描述
        caption = data["caption"]

        # 提取 subjects
        subjects = data["image_info"]["subjects"]
        selected_subjects = self._select_subjects(subjects)

        # 如果没有可用的 subject，使用 foreground_image 作为备选
        if len(selected_subjects) == 0:
            reference_images = [data["image_info"]["foreground_image"]]
        else:
            reference_images = [s["subject_image"] for s in selected_subjects]

        # 返回格式
        return {
            "prompts": caption,
            "video_frames": video_frames,  # [T, C, H, W], uint8
            "reference_images": reference_images,  # List[PIL.Image]
            "idx": idx,
            # 可选的元数据
            "metadata": {
                "key": data["key"],
                "num_subjects": len(selected_subjects),
                "subject_names": [s.get("class_name", "unknown") for s in selected_subjects],
                "aesthetic_score": data.get("aesthetic_score", 0.0),
                "motion_score": data.get("motion_score", 0.0),
            }
        }


class MultiPartOpenS2VAdapterDataset(Dataset):
    """
    多数据集部分的 OpenS2V 适配器

    支持从多个 JSON 文件加载数据。

    Args:
        json_paths: JSON 文件路径列表
        video_base_paths: 视频基础路径列表
        background_base_paths: 背景图片基础路径列表
        **kwargs: 传递给 OpenS2VAdapterDataset 的其他参数
    """

    def __init__(
        self,
        json_paths: List[str],
        video_base_paths: List[str],
        background_base_paths: List[str],
        **kwargs
    ):
        assert len(json_paths) == len(video_base_paths) == len(background_base_paths), (
            "Each dataset part must have corresponding JSON, video, and background paths"
        )

        self.datasets = []
        for json_path, video_path, bg_path in zip(json_paths, video_base_paths, background_base_paths):
            ds = OpenS2VAdapterDataset(
                json_path=json_path,
                video_base_path=video_path,
                background_base_path=bg_path,
                **kwargs
            )
            self.datasets.append(ds)

        self.lengths = [len(ds) for ds in self.datasets]
        self.cum_lengths = np.cumsum([0] + self.lengths)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        # 找到对应的数据集
        dset_idx = np.searchsorted(self.cum_lengths, idx, side="right") - 1
        local_idx = idx - self.cum_lengths[dset_idx]
        return self.datasets[dset_idx][local_idx]


def collate_opens2v_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义 collate 函数，用于 DataLoader

    处理可变数量的 reference_images。

    Args:
        batch: 样本列表

    Returns:
        批次字典
    """
    prompts = [item["prompts"] for item in batch]

    # video_frames 需要转换为 float 并归一化到 [-1, 1]
    video_frames_list = []
    for item in batch:
        vf = item["video_frames"]  # [T, C, H, W], uint8 [0, 255]
        vf = vf.float() / 255.0 * 2.0 - 1.0  # 转换到 [-1, 1]
        video_frames_list.append(vf)

    video_frames = torch.stack(video_frames_list, dim=0)  # [B, T, C, H, W]

    reference_images = [item["reference_images"] for item in batch]  # List[List[PIL.Image]]
    indices = [item["idx"] for item in batch]
    metadata = [item.get("metadata", {}) for item in batch]

    return {
        "prompts": prompts,
        "video_frames": video_frames,  # [B, T, C, H, W], float32 [-1, 1]
        "reference_images": reference_images,
        "idx": indices,
        "metadata": metadata,
    }


def create_opens2v_dataloader(
    json_paths: List[str],
    video_base_paths: List[str],
    background_base_paths: List[str],
    batch_size: int = 1,
    is_cross_frame: bool = False,
    shuffle: bool = False,
    num_workers: int = 4,
    **dataset_kwargs
):
    """
    创建 OpenS2V DataLoader

    Args:
        json_paths: JSON 文件路径列表
        video_base_paths: 视频基础路径列表
        background_base_paths: 背景图片基础路径列表
        batch_size: 批大小
        is_cross_frame: 是否使用跨帧数据
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        **dataset_kwargs: 传递给数据集的其他参数

    Returns:
        DataLoader
    """
    # 如果只有一个数据集，直接创建单个适配器
    if len(json_paths) == 1:
        dataset = OpenS2VAdapterDataset(
            json_path=json_paths[0],
            video_base_path=video_base_paths[0],
            background_base_path=background_base_paths[0],
            is_cross_frame=is_cross_frame,
            **dataset_kwargs
        )
    else:
        # 多个数据集，使用 MultiPart
        dataset = MultiPartOpenS2VAdapterDataset(
            json_paths=json_paths,
            video_base_paths=video_base_paths,
            background_base_paths=background_base_paths,
            is_cross_frame=is_cross_frame,
            **dataset_kwargs
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_opens2v_batch,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return dataloader


# 为了支持 cycle 迭代
def cycle(dl):
    """无限循环 DataLoader"""
    while True:
        for data in dl:
            yield data
