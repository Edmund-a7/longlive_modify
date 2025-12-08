import os
import gc
import cv2
import json
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from pycocotools import mask as mask_util
from typing import Dict, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader

import decord
from decord import VideoReader


def rle_to_mask(rle, img_width, img_height):
    rle_obj = {"counts": rle["counts"].encode("utf-8"), "size": [img_height, img_width]}
    return mask_util.decode(rle_obj)


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon


def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(1,))
    return mask


def refine_masks(
    masks: torch.BoolTensor, polygon_refinement: bool = False
) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)
    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask
    return masks


class OpenS2VDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        video_base_path: str,
        background_base_path: str,
        cross_frames_cluster_path: str,
        cross_frames_base_path: str,
        is_cross_frame: bool = False,
        height=480,
        width=832,
        sample_num_frames=49,
        sample_stride=3,
    ):
        self.json_path = json_path
        self.video_base_path = video_base_path
        self.background_base_path = background_base_path
        self.cross_frames_cluster_path = cross_frames_cluster_path
        self.cross_frames_base_path = cross_frames_base_path

        self.is_cross_frame = is_cross_frame

        self.height = height
        self.width = width
        self.sample_num_frames = sample_num_frames
        self.sample_stride = sample_stride

        with open(self.json_path, "r") as f:
            self.data = json.load(f)
            self.keys = list(self.data.keys())

        if is_cross_frame:
            with open(self.cross_frames_cluster_path, "r") as f:
                self.cross_frame_cluster_data = json.load(f)

    def _save_video(self, torch_frames, name="output.mp4"):
        from moviepy import ImageSequenceClip

        frames_np = torch_frames.cpu().numpy()
        if frames_np.dtype != "uint8":
            frames_np = frames_np.astype("uint8")
        frames_list = list(frames_np)
        desired_fps = 24
        clip = ImageSequenceClip(frames_list, fps=desired_fps)
        clip.write_videofile(name, codec="libx264")

    def _get_frame_indices_adjusted(self, video_length, n_frames):
        indices = list(range(video_length))
        additional_frames_needed = n_frames - video_length

        repeat_indices = []
        for i in range(additional_frames_needed):
            index_to_repeat = i % video_length
            repeat_indices.append(indices[index_to_repeat])

        all_indices = indices + repeat_indices
        all_indices.sort()

        return all_indices

    def _generate_frame_indices(self, valid_start, valid_end, n_frames, sample_stride):
        adjusted_length = valid_end - valid_start

        if adjusted_length <= n_frames:
            frame_indices = self._get_frame_indices_adjusted(adjusted_length, n_frames)
            frame_indices = [i + valid_start for i in frame_indices]
        else:
            clip_length = min(adjusted_length, (n_frames - 1) * sample_stride + 1)
            start_idx = random.randint(valid_start, valid_end - clip_length)
            frame_indices = np.linspace(
                start_idx, start_idx + clip_length - 1, n_frames, dtype=int
            ).tolist()

        return frame_indices

    def _short_resize_and_crop(self, frames, target_width, target_height):
        T, C, H, W = frames.shape
        aspect_ratio = W / H

        if aspect_ratio > target_width / target_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
            if new_height < target_height:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
            if new_width < target_width:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)

        resize_transform = transforms.Resize((new_height, new_width))
        crop_transform = transforms.CenterCrop((target_height, target_width))

        frames_tensor = frames  # (T, C, H, W)
        resized_frames = resize_transform(frames_tensor)
        cropped_frames = crop_transform(resized_frames)
        sample = cropped_frames

        return sample

    def get_cropped_subiect_image(
        self,
        input_image,
        annotation_idx,
        class_names,
        mask_data,
        image_width,
        image_height,
        bbox_data,
        use_bbox=False,
    ):
        class_name = class_names[f"{annotation_idx}"]["class_name"]
        gme_score = bbox_data[int(annotation_idx) - 1]["gme_score"]
        aes_score = bbox_data[int(annotation_idx) - 1]["aes_score"]

        if use_bbox:
            bbox = bbox_data[int(annotation_idx) - 1]["bbox"]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]

            x_min = int(max(x_min, 0))
            y_min = int(max(y_min, 0))
            x_max = int(min(x_max, image_width - 1))
            y_max = int(min(y_max, image_height - 1))

            resized_image = input_image[y_min : y_max + 1, x_min : x_max + 1]
        else:
            mask_rle = mask_data[annotation_idx]
            mask = rle_to_mask(mask_rle, image_width, image_height)

            # Find the bounding box of the mask
            rows, cols = np.where(mask == 1)
            if len(rows) == 0 or len(cols) == 0:
                return None

            y_min, y_max = np.min(rows), np.max(rows)
            x_min, x_max = np.min(cols), np.max(cols)

            # Adjust if the region goes out of bounds
            x_min = int(max(x_min, 0))
            y_min = int(max(y_min, 0))
            x_max = int(min(x_max, image_width - 1))
            y_max = int(min(y_max, image_height - 1))

            # Crop the region from the original image and mask
            cropped_image = input_image[y_min : y_max + 1, x_min : x_max + 1]
            cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

            # Create a white background of the same size as the crop
            white_background = np.ones_like(cropped_image) * 255

            # Apply the mask to the cropped image
            white_background[cropped_mask == 1] = cropped_image[cropped_mask == 1]
            resized_image = white_background

        pil_image = Image.fromarray(
            cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        ).convert("RGB")
        # pil_image.save(f"{class_name}_{i}_ratio{crop_ratio:.4f}.png")  # For sanity check

        crop_ratio = (pil_image.size[0] * pil_image.size[1]) / (
            image_height * image_width
        )

        return pil_image, class_name, gme_score, aes_score, crop_ratio

    def get_batch(self, item, key, main_part, cross_part):
        decord.bridge.set_bridge("torch")
        # train_transforms = transforms.Compose(
        #     [
        #         transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
        #     ]
        # )

        video_path = os.path.join(
            self.video_base_path.replace(main_part, cross_part),
            item["metadata"]["path"],
        )
        vr = VideoReader(video_path, num_threads=2)

        # crop (remove watermark) & cut (remove transition)
        s_x, e_x, s_y, e_y = item["metadata"]["crop"]
        start_frame = item["metadata"]["face_cut"][0]
        end_frame = item["metadata"]["face_cut"][1]
        frame_idx = self._generate_frame_indices(
            start_frame, end_frame, self.sample_num_frames, self.sample_stride
        )

        # For output gt video
        video = vr.get_batch(frame_idx).float()
        # video = train_transforms(video)
        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
        video = video[:, :, s_y:e_y, s_x:e_x]
        video = self._short_resize_and_crop(video, self.width, self.height)
        # self._save_video(video.permute(0, 2, 3, 1))  # For sanity check

        # For input subject image
        frame_idx = item["annotation"]["ann_frame_data"]["ann_frame_idx"]
        input_image = (
            vr.get_batch([int(frame_idx)]).numpy()[0][..., ::-1].astype(np.uint8)
        )
        input_image = input_image[s_y:e_y, s_x:e_x]
        image_width = input_image.shape[1]
        image_height = input_image.shape[0]

        del vr
        gc.collect()

        batch_info = {
            "foreground_image": Image.fromarray(input_image),
            "foreground_ase_score": item["annotation"]["ann_frame_data"][
                "foreground_ase_score"
            ],
            "background_image": Image.open(
                os.path.join(
                    self.background_base_path.replace(main_part, cross_part),
                    f"{key}.png",
                )
            ),
            "background_ase_score": item["annotation"]["ann_frame_data"][
                "background_ase_score"
            ],
            "subjects": [],
        }

        class_names = item["annotation"]["mask_map"]
        bbox_data = item["annotation"]["ann_frame_data"]["annotations"]
        mask_data = item["annotation"]["mask_annotation"][str(frame_idx)]

        for i, annotation_idx in enumerate(mask_data):
            subject_image, class_name, gme_score, aes_score, crop_ratio = (
                self.get_cropped_subiect_image(
                    input_image,
                    annotation_idx,
                    class_names,
                    mask_data,
                    image_width,
                    image_height,
                    bbox_data,
                    use_bbox=False,
                )
            )
            if subject_image is None:
                continue

            batch_info["subjects"].append(
                {
                    "annotation_idx": annotation_idx,
                    "class_name": class_name,
                    "subject_image": subject_image,
                    "subject_gme_score": gme_score,
                    "subject_aes_score": aes_score,
                    "subject_crop_ratio": crop_ratio,
                }
            )

        return video, batch_info

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.keys[idx]
        item = self.data[key]
        main_part = self.video_base_path.rsplit("/", 1)[1]

        video, image_info = self.get_batch(
            item=item, key=key, main_part=main_part, cross_part=main_part
        )
        filtered_cross_data = None
        cross_image_info = None
        if self.is_cross_frame:
            main_data_id = self.video_base_path.rsplit("/", 1)[1] + "/" + key
            # For mapping
            cluster_id = key.split("_segment")[0]
            if cluster_id == key:
                cluster_id = key.split("_part")[0]
            selected_cross_data_id = random.choice(
                self.cross_frame_cluster_data[cluster_id]
            )
            selected_cross_data_key = random.choice(
                self.cross_frame_cluster_data[cluster_id]
            ).split("/", 1)[1]
            selected_cross_data_path = os.path.join(
                self.cross_frames_base_path, f"{cluster_id}.json"
            )
            with open(selected_cross_data_path, "r") as f:
                selected_cross_data = json.load(f)
            filtered_cross_data = []
            for temp_item in selected_cross_data:
                if (
                    temp_item["cur_id"] == main_data_id
                    and temp_item["aft_id"] == selected_cross_data_id
                ) or (
                    temp_item["aft_id"] == main_data_id
                    and temp_item["cur_id"] == selected_cross_data_id
                ):
                    filtered_cross_data.append(temp_item)
            # For image
            cross_item = self.data[selected_cross_data_key]
            cross_part = selected_cross_data_id.split("/", 1)[0]
            _, cross_image_info = self.get_batch(
                item=cross_item,
                key=selected_cross_data_key,
                main_part=main_part,
                cross_part=cross_part,
            )

        return {
            "key": key,
            "video": video,
            "image_info": image_info,
            "cross_image_info": cross_image_info,
            "filtered_cross_data": filtered_cross_data,
            "crop": item["metadata"]["crop"],
            "cut": item["metadata"]["face_cut"],
            "caption": item["metadata"]["face_cap_qwen"],
            "tech_score": item["metadata"]["tech"],
            "motion_score": item["metadata"]["motion"],
            "aesthetic_score": item["metadata"]["aesthetic"],
            "fps": item["metadata"]["fps"],
            "num_frames": item["metadata"]["num_frames"],
            "resolution": item["metadata"]["resolution"],
        }


class MultiPartOpenS2VDataset(Dataset):
    def __init__(
        self,
        json_paths,
        video_base_paths,
        background_base_paths,
        cross_frames_cluster_path,
        cross_frames_base_path,
        is_cross_frame=False,
        **kwargs,
    ):
        assert len(json_paths) == len(video_base_paths) == len(background_base_paths), (
            "each part must have a corresponding path"
        )
        self.datasets = []
        for j, v, b in zip(json_paths, video_base_paths, background_base_paths):
            ds = OpenS2VDataset(
                j,
                v,
                b,
                cross_frames_cluster_path,
                cross_frames_base_path,
                is_cross_frame=is_cross_frame,
                **kwargs,
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


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [item["key"] for item in batch]

    return {
        "keys": keys,
        "videos": [item["video"] for item in batch],
        "image_infos": [item["image_info"] for item in batch],
        "cross_image_infos": [item.get("cross_image_info") for item in batch],
        "filtered_cross_datas": [item.get("filtered_cross_data") for item in batch],
        "crops": [item["crop"] for item in batch],
        "cuts": [item["cut"] for item in batch],
        "captions": [item["caption"] for item in batch],
        "tech_scores": torch.tensor([item["tech_score"] for item in batch]),
        "motion_scores": torch.tensor([item["motion_score"] for item in batch]),
        "aesthetic_scores": torch.tensor([item["aesthetic_score"] for item in batch]),
        "fps_values": torch.tensor([item["fps"] for item in batch]),
        "num_frames": torch.tensor([item["num_frames"] for item in batch]),
        "resolutions": [item["resolution"] for item in batch],
    }


def create_dataloader(
    # json_path: str,
    # video_base_path: str,
    # background_base_path: str,
    json_paths: list,
    video_base_paths: list,
    background_base_paths: list,
    cross_frames_cluster_path: str,
    cross_frames_base_path: str,
    batch_size: int = 1,
    is_cross_frame: bool = True,
    shuffle: bool = False,
    num_workers: int = 0,
):
    # dataset = OpenS2VDataset(json_path, video_base_path, background_base_path)
    dataset = MultiPartOpenS2VDataset(
        json_paths,
        video_base_paths,
        background_base_paths,
        cross_frames_cluster_path,
        cross_frames_base_path,
        is_cross_frame=is_cross_frame,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return dataloader


if __name__ == "__main__":
    # video_base_path = "demo_result/step0/videos/dataset1"
    # background_base_path = "demo_result/step5/final_output/dataset1/foreground"
    # mask_and_bbox_json_file = "demo_result/step5/merge_final_json/dataset1.json"
    # dataloader = create_dataloader(mask_and_bbox_json_file, video_base_path, background_base_path)

    cross_frames_cluster_path = (
        "demo_result/step6/cross-frames-pairs/cluster_videos.json"
    )
    cross_frames_base_path = "demo_result/step6/cross-frames-pairs/final_output"

    mask_and_bbox_json_paths = [
        "demo_result/step5/merge_final_json/dataset1.json",
        "demo_result/step5/merge_final_json/dataset1_cluster.json",
    ]
    video_base_paths = [
        "demo_result/step0/videos/dataset1",
        "demo_result/step0/videos/dataset1_cluster",
    ]
    background_base_paths = [
        "demo_result/step5/final_output/dataset1/foreground",
        "demo_result/step5/final_output/dataset1_cluster/foreground",
    ]
    dataloader = create_dataloader(
        mask_and_bbox_json_paths,
        video_base_paths,
        background_base_paths,
        cross_frames_cluster_path,
        cross_frames_base_path,
    )

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Keys: {batch['keys']}...")

        # only selct the first sample here
        print(f"Videos shape: {batch['videos'][0].shape}")
        print(f"Subject images len: {len(batch['image_infos'][0]['subjects'])}")
        print(f"Foreground_image: {batch['image_infos'][0]['foreground_image']}")
        print(f"Foreground_ase: {batch['image_infos'][0]['foreground_ase_score']}")
        print(f"Background_image: {batch['image_infos'][0]['background_image']}")
        print(f"Background_ase: {batch['image_infos'][0]['background_ase_score']}")
        # only selct the first sample here

        # cross frame pairs
        print(
            f"Cross subject images len: {len(batch['cross_image_infos'][0]['subjects'])}"
        )
        print(f"Cross pairs len: {len(batch['filtered_cross_datas'][0])}")
        # cross frame pairs

        print(f"Crops: {batch['crops']}")
        print(f"Cuts: {batch['cuts']}")
        print(f"Captions: {batch['captions']}")

        print(f"Tech scores: {batch['tech_scores']}")
        print(f"Motion scores: {batch['motion_scores']}")
        print(f"Aesthetic scores: {batch['aesthetic_scores']}")

        print(f"FPS values: {batch['fps_values']}")
        print(f"Num frames: {batch['num_frames']}")
        print(f"Resolutions: {batch['resolutions']}\n")