# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import datasets



class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TwoTextDataset(Dataset):
    """Dataset that returns two text prompts per sample for prompt-switch training.

    The dataset behaves similarly to :class:`TextDataset` but instead of a single
    prompt, it provides *two* prompts – typically the first prompt is used for the
    first segment of the video, and the second prompt is used after a temporal
    switch during training.

    Args:
        prompt_path (str): Path to a text file containing the *first* prompt for
            each sample. One prompt per line.
        switch_prompt_path (str): Path to a text file containing the *second*
            prompt for each sample. Must have the **same number of lines** as
            ``prompt_path`` so that prompts are paired 1-to-1.
    """
    def __init__(self, prompt_path: str, switch_prompt_path: str):
        # Load the first-segment prompts.
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        # Load the second-segment prompts.
        with open(switch_prompt_path, encoding="utf-8") as f:
            self.switch_prompt_list = [line.rstrip() for line in f]

        assert len(self.switch_prompt_list) == len(self.prompt_list), (
            "The two prompt files must contain the same number of lines so that "
            "each first-segment prompt is paired with exactly one second-segment prompt."
        )

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {
            "prompts": self.prompt_list[idx],            # first-segment prompt
            "switch_prompts": self.switch_prompt_list[idx],  # second-segment prompt
            "idx": idx,
        }


class MultiTextDataset(Dataset):
    """Dataset for multi-segment prompts stored in a JSONL file.

    Each line is a JSON object, e.g.
        {"prompts": ["a cat", "a dog", "a bird"]}

    Args
    ----
    prompt_path : str
        Path to the JSONL file
    field       : str
        Name of the list-of-strings field, default "prompts"
    cache_dir   : str | None
        ``cache_dir`` passed to HF Datasets (optional)
    """

    def __init__(self, prompt_path: str, field: str = "prompts", cache_dir: str | None = None):
        self.ds = datasets.load_dataset(
            "json",
            data_files=prompt_path,
            split="train",
            cache_dir=cache_dir,
            streaming=False, 
        )

        assert len(self.ds) > 0, "JSONL is empty"
        assert field in self.ds.column_names, f"Missing field '{field}'"

        seg_len = len(self.ds[0][field])
        for i, ex in enumerate(self.ds):
            val = ex[field]
            assert isinstance(val, list), f"Line {i} field '{field}' is not a list"
            assert len(val) == seg_len,  f"Line {i} list length mismatch"

        self.field = field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "prompts_list": self.ds[idx][self.field],  # List[str]
        }


class ReferenceImageDataset(Dataset):
    """Dataset for training with reference images.

    Each line in the JSONL file should be:
    {
        "prompt": "a woman dancing in the park",
        "reference_images": ["path/to/image1.jpg", "path/to/image2.jpg"]
    }

    Args:
        data_path: Path to the JSONL file
        image_root: Root directory for reference images (optional)
    """

    def __init__(self, data_path: str, image_root: str = None):
        self.image_root = image_root or ""
        self.data = []

        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.data.append(item)

        assert len(self.data) > 0, f"No data found in {data_path}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]

        # Load reference images
        reference_images = []
        image_paths = item.get("reference_images", [])

        for img_path in image_paths:
            full_path = os.path.join(self.image_root, img_path) if self.image_root else img_path
            if os.path.exists(full_path):
                img = Image.open(full_path).convert("RGB")
                reference_images.append(img)
            else:
                print(f"Warning: Reference image not found: {full_path}")

        return {
            "prompts": prompt,
            "reference_images": reference_images,  # List[PIL.Image]
            "idx": idx,
        }


class VideoReferenceImageDataset(Dataset):
    """Dataset for training with video + reference images + prompt.

    训练参考图功能的数据集，使用真实视频作为监督信号。

    JSONL 格式:
    {
        "video_path": "videos/001.mp4",
        "prompt": "a woman dancing in the park",
        "reference_images": ["images/001_ref1.jpg", "images/001_ref2.jpg"]
    }

    Args:
        data_path: Path to the JSONL file
        video_root: Root directory for video files (optional)
        image_root: Root directory for reference images (optional)
        num_frames: Number of frames to sample from video
        frame_interval: Interval between sampled frames (default: 1)
        height: Target height for video frames (default: 480)
        width: Target width for video frames (default: 832)
    """

    def __init__(
        self,
        data_path: str,
        video_root: str = None,
        image_root: str = None,
        num_frames: int = 21,
        frame_interval: int = 1,
        height: int = 480,
        width: int = 832
    ):
        self.video_root = video_root or ""
        self.image_root = image_root or ""
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.height = height
        self.width = width

        self.data = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.data.append(item)

        assert len(self.data) > 0, f"No data found in {data_path}"

    def __len__(self):
        return len(self.data)

    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video frames.

        Returns:
            frames: [num_frames, 3, H, W] tensor, normalized to [-1, 1]
        """
        import decord
        from decord import VideoReader, cpu

        decord.bridge.set_bridge('torch')

        full_path = os.path.join(self.video_root, video_path) if self.video_root else video_path

        vr = VideoReader(full_path, ctx=cpu(0))
        total_frames = len(vr)

        # 计算需要的总帧数
        required_frames = self.num_frames * self.frame_interval

        if total_frames >= required_frames:
            # 随机选择起始帧
            start_idx = np.random.randint(0, total_frames - required_frames + 1)
            frame_indices = list(range(start_idx, start_idx + required_frames, self.frame_interval))
        else:
            # 视频太短，重复最后一帧
            frame_indices = list(range(0, total_frames, self.frame_interval))
            while len(frame_indices) < self.num_frames:
                frame_indices.append(frame_indices[-1])
            frame_indices = frame_indices[:self.num_frames]

        # 读取帧
        frames = vr.get_batch(frame_indices)  # [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2).float()  # [T, C, H, W]

        # Resize
        import torch.nn.functional as F
        frames = F.interpolate(frames, size=(self.height, self.width), mode='bilinear', align_corners=False)

        # Normalize to [-1, 1] (input range is [0, 255])
        frames = (frames / 255.0) * 2.0 - 1.0

        return frames

    def _load_reference_images(self, image_paths: list) -> list:
        """Load reference images as PIL Images."""
        images = []
        for img_path in image_paths:
            full_path = os.path.join(self.image_root, img_path) if self.image_root else img_path
            if os.path.exists(full_path):
                img = Image.open(full_path).convert("RGB")
                images.append(img)
            else:
                print(f"Warning: Reference image not found: {full_path}")
        return images

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = item["prompt"]
        video_path = item["video_path"]
        image_paths = item.get("reference_images", [])

        # Load video frames
        video_frames = self._load_video_frames(video_path)  # [T, C, H, W]

        # Load reference images
        reference_images = self._load_reference_images(image_paths)

        return {
            "prompts": prompt,
            "video_frames": video_frames,  # [T, C, H, W] tensor
            "reference_images": reference_images,  # List[PIL.Image]
            "idx": idx,
        }


def collate_video_refimg(batch):
    """Custom collate function for VideoReferenceImageDataset.

    Handles variable number of reference images per sample.
    """
    prompts = [item["prompts"] for item in batch]
    video_frames = torch.stack([item["video_frames"] for item in batch], dim=0)  # [B, T, C, H, W]
    reference_images = [item["reference_images"] for item in batch]  # List[List[PIL.Image]]
    indices = [item["idx"] for item in batch]

    return {
        "prompts": prompts,
        "video_frames": video_frames,
        "reference_images": reference_images,
        "idx": indices,
    }


def cycle(dl):
    while True:
        for data in dl:
            yield data
