export HIP_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=8 train_opens2v.py \
        --config_path configs/train_opens2v.yaml \
        --logdir outputs/opens2v_training