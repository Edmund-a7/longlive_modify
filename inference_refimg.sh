#!/bin/bash
# 参考图推理脚本 (灵活版)
# 支持任意数量的参考图 (0-3张) 和可选的 text prompt
#
# 用法:
#   bash inference_refimg.sh GPU_ID [参考图1] [参考图2] [参考图3] ["text prompt"]
#
# 示例:
#   bash inference_refimg.sh 0                                    # 无参考图，使用配置文件 prompt
#   bash inference_refimg.sh 0 human.png                          # 1张参考图
#   bash inference_refimg.sh 0 human.png thing.png                # 2张参考图
#   bash inference_refimg.sh 0 human.png thing.png env.png        # 3张参考图
#   bash inference_refimg.sh 0 "a cat walking"                    # 无参考图，命令行 prompt
#   bash inference_refimg.sh 0 human.png "a cat walking"          # 1张参考图 + prompt
#   bash inference_refimg.sh 0 human.png thing.png "a cat"        # 2张参考图 + prompt
#   bash inference_refimg.sh 0 human.png thing.png env.png "cat"  # 3张参考图 + prompt

GPU_ID=${1:-0}
shift  # 移除第一个参数 (GPU_ID)

# 解析剩余参数: 区分图片文件和 text prompt
REF_IMGS=()
TEXT_PROMPT=""

for arg in "$@"; do
    # 检查是否是图片文件 (存在且是文件)
    if [[ -f "$arg" ]]; then
        REF_IMGS+=("$arg")
    else
        # 如果参数看起来像图片路径 (以 .png/.jpg/.jpeg 结尾) 但文件不存在，警告
        if [[ "$arg" =~ \.(png|jpg|jpeg|webp)$ ]]; then
            echo "警告: 图片文件不存在: $arg"
        else
            # 否则当作 text prompt
            TEXT_PROMPT="$arg"
        fi
    fi
done

echo "=== LongLive 参考图推理 ==="
echo "GPU: $GPU_ID"
echo "参考图数量: ${#REF_IMGS[@]}"
if [ ${#REF_IMGS[@]} -gt 0 ]; then
    echo "参考图: ${REF_IMGS[*]}"
fi
if [ -n "$TEXT_PROMPT" ]; then
    echo "Text Prompt: $TEXT_PROMPT"
fi

# 构建命令参数
CMD_ARGS=(
    --config_path configs/longlive_inference_refimg.yaml
    --clip_path "Skywork/SkyReels-A2"
)

# 添加参考图参数 (如果有)
if [ ${#REF_IMGS[@]} -gt 0 ]; then
    CMD_ARGS+=(--use_reference_image)
    CMD_ARGS+=(--reference_images "${REF_IMGS[@]}")
fi

# 添加 prompt 参数 (如果有)
if [ -n "$TEXT_PROMPT" ]; then
    CMD_ARGS+=(--prompt "$TEXT_PROMPT")
fi

echo "执行命令: torchrun ... inference.py ${CMD_ARGS[*]}"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID \
torchrun \
  --nproc_per_node=1 \
  --master_port=29502 \
  inference.py \
  "${CMD_ARGS[@]}"
