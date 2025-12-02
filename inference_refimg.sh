#!/bin/bash
# 参考图推理脚本
# 用法: bash inference_refimg.sh [GPU_ID] [参考图1] [参考图2] [参考图3]

GPU_ID=${1:-0}
REF_IMG1=${2:-"reference_images/human.png"}
REF_IMG2=${3:-"reference_images/thing.png"}
REF_IMG3=${4:-"reference_images/env.png"}

echo "=== LongLive 参考图推理 ==="
echo "GPU: $GPU_ID"
echo "参考图: $REF_IMG1, $REF_IMG2, $REF_IMG3"

CUDA_VISIBLE_DEVICES=$GPU_ID \
torchrun \
  --nproc_per_node=1 \
  --master_port=29502 \
  inference.py \
  --config_path configs/longlive_inference_refimg.yaml \
  --use_reference_image \
  --reference_images "$REF_IMG1" "$REF_IMG2" "$REF_IMG3" \
  --clip_path "Skywork/SkyReels-A2"
