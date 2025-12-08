HIP_VISIBLE_DEVICES=7 \
torchrun \
  --nproc_per_node=1 \
  --master_port=29501 \
  inference.py \
  --config_path configs/longlive_inference.yaml