# OpenS2V Dataset for longlive_modify
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
OpenS2V 数据集 - 简化版

直接从 total_part1.json + 视频文件提取数据，不需要 background。

数据格式说明 (total_part1.json):
{
  "key_id": {
    "metadata": {
      "path": "part_xxx/video_id/video.mp4",  # 相对于 video_base_path
      "face_cut": [start, end],               # 有效帧范围
      "crop": [s_x, e_x, s_y, e_y],           # 裁剪区域（去水印）
      "face_cap_qwen": "描述文本",             # caption
    },
    "annotation": {
      "ann_frame_data": {
        "ann_frame_idx": 200,                 # 标注帧索引
        "annotations": [{"bbox": [...], "aes_score": ..., "gme_score": ...}, ...]
      },
      "mask_map": {"1": {"class_name": "person"}, ...},
      "mask_annotation": {"200": {"1": {"counts": "...", "size": [h, w]}, ...}}
    }
  }
}

使用方法:
    from utils.dataset_opens2v import create_opens2v_dataloader

    dataloader = create_opens2v_dataloader(
        json_paths=["path/to/total_part1.json"],
        video_base_paths=["path/to/videos"],
        batch_size=1,
        shuffle=True,
    )
"""

import gc
import json
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from torch.utils.data import Dataset
from torchvision import transforms

import decord
from decord import VideoReader


def rle_to_mask(rle: Dict, img_width: int, img_height: int) -> np.ndarray:
    """将 RLE 格式转换为 mask"""
    rle_obj = {"counts": rle["counts"].encode("utf-8"), "size": [img_height, img_width]}
    return mask_util.decode(rle_obj)


def extract_subject_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    use_white_background: bool = True
) -> Image.Image:
    """
    从图像中根据 mask 提取 subject

    Args:
        image: BGR 格式的图像 [H, W, 3]
        mask: 二值 mask [H, W]
        use_white_background: 是否使用白色背景

    Returns:
        PIL.Image: 裁剪后的 subject 图像
    """
    # 找到 mask 的边界框
    rows, cols = np.where(mask == 1)
    if len(rows) == 0 or len(cols) == 0:
        return None

    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)

    # 裁剪
    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

    if use_white_background:
        # 白色背景
        result = np.ones_like(cropped_image) * 255
        result[cropped_mask == 1] = cropped_image[cropped_mask == 1]
    else:
        result = cropped_image

    # BGR -> RGB, 转为 PIL
    pil_image = Image.fromarray(
        cv2.cvtColor(result, cv2.COLOR_BGR2RGB).astype(np.uint8)
    ).convert("RGB")

    return pil_image


class OpenS2VDataset(Dataset):
    """
    OpenS2V 数据集

    直接从 JSON + 视频提取 subject，不需要预处理的 background。

    Args:
        json_path: total_part1.json 路径
        video_base_path: 视频根目录
        height: 输出视频高度
        width: 输出视频宽度
        sample_num_frames: 采样帧数
        sample_stride: 采样步长
        max_subjects_per_sample: 每个样本最多使用多少个 subject
        subject_selection: subject 选择策略 ('first', 'random', 'best_score')
    """

    def __init__(
        self,
        json_path: str,
        video_base_path: str,
        height: int = 480,
        width: int = 832,
        sample_num_frames: int = 49,
        sample_stride: int = 3,
        max_subjects_per_sample: int = 1,
        subject_selection: str = "first",
    ):
        self.json_path = json_path
        self.video_base_path = video_base_path
        self.height = height
        self.width = width
        self.sample_num_frames = sample_num_frames
        self.sample_stride = sample_stride
        self.max_subjects = max_subjects_per_sample
        self.subject_selection = subject_selection

        # 加载 JSON
        print(f"Loading JSON: {json_path}")
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        print(f"  Loaded {len(self.keys)} samples")

    def __len__(self):
        return len(self.keys)

    def _get_frame_indices(self, valid_start: int, valid_end: int) -> List[int]:
        """生成采样帧索引"""
        adjusted_length = valid_end - valid_start
        n_frames = self.sample_num_frames
        sample_stride = self.sample_stride

        if adjusted_length <= n_frames:
            # 视频太短，需要重复帧
            indices = list(range(adjusted_length))
            additional = n_frames - adjusted_length
            repeat_indices = [i % adjusted_length for i in range(additional)]
            all_indices = sorted(indices + repeat_indices)
            return [i + valid_start for i in all_indices]
        else:
            # 正常采样
            clip_length = min(adjusted_length, (n_frames - 1) * sample_stride + 1)
            start_idx = random.randint(valid_start, valid_end - clip_length)
            return np.linspace(start_idx, start_idx + clip_length - 1, n_frames, dtype=int).tolist()

    def _resize_and_crop(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize 并中心裁剪到目标尺寸"""
        T, C, H, W = frames.shape
        target_h, target_w = self.height, self.width

        # 计算缩放比例，保持宽高比
        aspect_ratio = W / H
        if aspect_ratio > target_w / target_h:
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
            if new_w < target_w:
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
        else:
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
            if new_h < target_h:
                new_h = target_h
                new_w = int(target_h * aspect_ratio)

        resize_transform = transforms.Resize((new_h, new_w))
        crop_transform = transforms.CenterCrop((target_h, target_w))

        frames = resize_transform(frames)
        frames = crop_transform(frames)
        return frames

    def _extract_subjects(
        self,
        image: np.ndarray,
        annotation: Dict,
        frame_idx: int
    ) -> List[Dict]:
        """
        从图像中提取所有 subjects

        Returns:
            List[Dict]: 每个 dict 包含 subject_image, class_name, aes_score 等
        """
        img_height, img_width = image.shape[:2]
        subjects = []

        mask_map = annotation.get("mask_map", {})
        mask_annotation = annotation.get("mask_annotation", {})
        ann_frame_data = annotation.get("ann_frame_data", {})
        bbox_annotations = ann_frame_data.get("annotations", [])

        # 获取该帧的 mask 数据
        frame_masks = mask_annotation.get(str(frame_idx), {})

        for annotation_idx, mask_rle in frame_masks.items():
            try:
                # 获取类别名
                class_info = mask_map.get(str(annotation_idx), {})
                class_name = class_info.get("class_name", "unknown")

                # 获取分数
                bbox_idx = int(annotation_idx) - 1
                if 0 <= bbox_idx < len(bbox_annotations):
                    aes_score = bbox_annotations[bbox_idx].get("aes_score", 0.0)
                    gme_score = bbox_annotations[bbox_idx].get("gme_score", 0.0)
                else:
                    aes_score = 0.0
                    gme_score = 0.0

                # 解码 mask 并提取 subject
                mask = rle_to_mask(mask_rle, img_width, img_height)
                subject_image = extract_subject_from_mask(image, mask)

                if subject_image is not None:
                    # 计算裁剪比例
                    crop_ratio = (subject_image.size[0] * subject_image.size[1]) / (img_width * img_height)

                    subjects.append({
                        "subject_image": subject_image,
                        "class_name": class_name,
                        "subject_aes_score": aes_score,
                        "subject_gme_score": gme_score,
                        "subject_crop_ratio": crop_ratio,
                    })
            except Exception as e:
                continue

        return subjects

    def _select_subjects(self, subjects: List[Dict]) -> List[Dict]:
        """选择要使用的 subjects"""
        if len(subjects) == 0:
            return []

        if self.subject_selection == "first":
            return subjects[:self.max_subjects]
        elif self.subject_selection == "random":
            n = min(self.max_subjects, len(subjects))
            return random.sample(subjects, n)
        elif self.subject_selection == "best_score":
            sorted_subjects = sorted(
                subjects,
                key=lambda x: x.get("subject_aes_score", 0.0),
                reverse=True
            )
            return sorted_subjects[:self.max_subjects]
        else:
            return subjects[:self.max_subjects]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.keys[idx]
        item = self.data[key]
        metadata = item["metadata"]
        annotation = item["annotation"]

        # 构建视频路径
        video_path = os.path.join(self.video_base_path, metadata["path"])

        try:
            decord.bridge.set_bridge("torch")
            vr = VideoReader(video_path, num_threads=2)

            # 获取裁剪和有效帧范围
            crop = metadata.get("crop", [0, vr[0].shape[1], 0, vr[0].shape[0]])
            s_x, e_x, s_y, e_y = crop
            start_frame, end_frame = metadata["face_cut"]

            # 采样帧索引
            frame_indices = self._get_frame_indices(start_frame, end_frame)

            # 读取视频帧
            video = vr.get_batch(frame_indices).float()  # [T, H, W, C]
            video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
            video = video[:, :, s_y:e_y, s_x:e_x]  # 裁剪去水印
            video = self._resize_and_crop(video)  # resize

            # 读取标注帧用于提取 subject
            ann_frame_idx = annotation["ann_frame_data"]["ann_frame_idx"]
            ann_frame = vr.get_batch([int(ann_frame_idx)]).asnumpy()[0]  # [H, W, C] RGB
            ann_frame = ann_frame[..., ::-1]  # RGB -> BGR (for cv2)
            ann_frame = ann_frame[s_y:e_y, s_x:e_x]  # 裁剪

            del vr
            gc.collect()

        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # 返回空数据，让 DataLoader 跳过
            return self.__getitem__((idx + 1) % len(self))

        # 提取 subjects
        subjects = self._extract_subjects(ann_frame, annotation, ann_frame_idx)
        selected_subjects = self._select_subjects(subjects)

        # 如果没有 subject，使用整个前景帧作为备选
        if len(selected_subjects) == 0:
            foreground_image = Image.fromarray(
                cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
            ).convert("RGB")
            reference_images = [foreground_image]
        else:
            reference_images = [s["subject_image"] for s in selected_subjects]

        # 获取 caption
        caption = metadata.get("face_cap_qwen", "")

        return {
            "prompts": caption,
            "video_frames": video,  # [T, C, H, W], float32 [0, 255]
            "reference_images": reference_images,  # List[PIL.Image]
            "idx": idx,
            "metadata": {
                "key": key,
                "num_subjects": len(selected_subjects),
                "subject_names": [s.get("class_name", "unknown") for s in selected_subjects],
                "aesthetic_score": metadata.get("aesthetic", 0.0),
                "motion_score": metadata.get("motion", 0.0),
            }
        }


class MultiPartOpenS2VDataset(Dataset):
    """多部分 OpenS2V 数据集"""

    def __init__(
        self,
        json_paths: List[str],
        video_base_paths: List[str],
        **kwargs
    ):
        assert len(json_paths) == len(video_base_paths), (
            "json_paths and video_base_paths must have the same length"
        )

        self.datasets = []
        for json_path, video_path in zip(json_paths, video_base_paths):
            ds = OpenS2VDataset(
                json_path=json_path,
                video_base_path=video_path,
                **kwargs
            )
            self.datasets.append(ds)

        self.lengths = [len(ds) for ds in self.datasets]
        self.cum_lengths = np.cumsum([0] + self.lengths)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        dset_idx = np.searchsorted(self.cum_lengths, idx, side="right") - 1
        local_idx = idx - self.cum_lengths[dset_idx]
        return self.datasets[dset_idx][local_idx]


def collate_opens2v_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DataLoader 的 collate 函数"""
    prompts = [item["prompts"] for item in batch]

    # video_frames: [0, 255] -> [-1, 1]
    video_frames_list = []
    for item in batch:
        vf = item["video_frames"]  # [T, C, H, W], float32 [0, 255]
        vf = vf / 255.0 * 2.0 - 1.0  # 转换到 [-1, 1]
        video_frames_list.append(vf)

    video_frames = torch.stack(video_frames_list, dim=0)  # [B, T, C, H, W]

    reference_images = [item["reference_images"] for item in batch]
    indices = [item["idx"] for item in batch]
    metadata = [item.get("metadata", {}) for item in batch]

    return {
        "prompts": prompts,
        "video_frames": video_frames,  # [B, T, C, H, W], float32 [-1, 1]
        "reference_images": reference_images,  # List[List[PIL.Image]]
        "idx": indices,
        "metadata": metadata,
    }


def create_opens2v_dataloader(
    json_paths: List[str],
    video_base_paths: List[str],
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 4,
    # 以下参数保持向后兼容，但不再使用
    background_base_paths: List[str] = None,
    is_cross_frame: bool = False,
    cross_frames_cluster_path: str = "",
    cross_frames_base_path: str = "",
    **dataset_kwargs
):
    """
    创建 OpenS2V DataLoader

    Args:
        json_paths: JSON 文件路径列表 (e.g., ["total_part1.json"])
        video_base_paths: 视频根目录列表 (e.g., ["/data/OpenS2V-5M/Videos/total_part1"])
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        **dataset_kwargs: 传递给 OpenS2VDataset 的参数
            - height: 视频高度 (默认 480)
            - width: 视频宽度 (默认 832)
            - sample_num_frames: 采样帧数 (默认 49)
            - sample_stride: 采样步长 (默认 3)
            - max_subjects_per_sample: 每样本最多 subject 数 (默认 1)
            - subject_selection: 选择策略 ('first', 'random', 'best_score')

    Returns:
        DataLoader
    """
    # 忽略不再需要的参数
    if background_base_paths is not None:
        print("[Info] background_base_paths is no longer needed, ignoring...")
    if is_cross_frame:
        print("[Info] is_cross_frame is not supported in this version, ignoring...")

    if len(json_paths) == 1:
        dataset = OpenS2VDataset(
            json_path=json_paths[0],
            video_base_path=video_base_paths[0],
            **dataset_kwargs
        )
    else:
        dataset = MultiPartOpenS2VDataset(
            json_paths=json_paths,
            video_base_paths=video_base_paths,
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


def cycle(dl):
    """无限循环 DataLoader"""
    while True:
        for data in dl:
            yield data


# ============== 测试代码 ==============
if __name__ == "__main__":
    # 测试数据加载
    json_path = "../total_part1.json"  # 修改为你的路径
    video_base_path = "/path/to/videos"  # 修改为你的路径

    if os.path.exists(json_path):
        dataloader = create_opens2v_dataloader(
            json_paths=[json_path],
            video_base_paths=[video_base_path],
            batch_size=1,
            shuffle=False,
            num_workers=0,
            sample_num_frames=21,
        )

        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Prompts: {batch['prompts'][0][:50]}...")
            print(f"  Video shape: {batch['video_frames'].shape}")
            print(f"  Num reference images: {len(batch['reference_images'][0])}")
            print(f"  Metadata: {batch['metadata'][0]}")

            if batch_idx >= 2:
                break
    else:
        print(f"JSON not found: {json_path}")
