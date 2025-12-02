# LongLive 参考图推理 - 服务器部署指南

## 1. 文件同步到服务器

需要同步的文件 (新增/修改):
```
longlive_modify/
├── utils/wan_wrapper.py          # 新增 WanCLIPEncoder
├── wan/modules/model.py          # 新增 CLIPProjMLP, VAEProjMLP, WanDualCrossAttention
├── wan/modules/causal_model.py   # 修改 CausalWanAttentionBlock, CausalWanModel
├── pipeline/causal_inference.py  # 修改推理流程
├── inference.py                  # 新增参考图参数
├── configs/longlive_inference_refimg.yaml  # 新配置文件
└── inference_refimg.sh           # 新启动脚本
```

## 2. 下载 CLIP 模型

在服务器上下载 SkyReels-A2 的 CLIP 组件:

```bash
# 方式1: 使用 huggingface-cli (推荐)
pip install huggingface_hub[cli]
huggingface-cli download Skywork/SkyReels-A2 --include "image_encoder/*" "image_processor/*" --local-dir ./SkyReels-A2

# 方式2: 使用 Python
python -c "
from transformers import CLIPVisionModel, CLIPImageProcessor
model = CLIPVisionModel.from_pretrained('Skywork/SkyReels-A2', subfolder='image_encoder')
processor = CLIPImageProcessor.from_pretrained('Skywork/SkyReels-A2', subfolder='image_processor')
model.save_pretrained('./SkyReels-A2/image_encoder')
processor.save_pretrained('./SkyReels-A2/image_processor')
"
```

## 3. 准备参考图像

创建参考图像目录并放入图片:
```bash
mkdir -p reference_images
# 放入 3 张参考图:
# - human.png (人物/主体参考)
# - thing.png (物体参考)
# - env.png (环境/背景参考)
```

图像要求:
- 格式: PNG/JPG
- 建议分辨率: 512x512 或更高
- 会自动裁剪/缩放到视频分辨率

## 4. 修改配置 (如需要)

编辑 `configs/longlive_inference_refimg.yaml`:
```yaml
# 修改输出目录
output_folder: videos/refimg

# 修改提示词文件
data_path: longlive_models/prompts/infer_test.txt

# 如果 CLIP 模型在本地
clip_path: "./SkyReels-A2"  # 改为本地路径
```

## 5. 运行推理

### 方式 A: 使用脚本 (推荐)
```bash
cd longlive_modify
chmod +x inference_refimg.sh

# 基本用法
bash inference_refimg.sh 0 reference_images/human.png reference_images/thing.png reference_images/env.png

# 参数说明:
# $1: GPU ID (默认 0)
# $2: 人物参考图路径
# $3: 物体参考图路径
# $4: 环境参考图路径
```

### 方式 B: 直接命令
```bash
CUDA_VISIBLE_DEVICES=0 \
torchrun \
  --nproc_per_node=1 \
  --master_port=29502 \
  inference.py \
  --config_path configs/longlive_inference_refimg.yaml \
  --use_reference_image \
  --reference_images human.png thing.png env.png \
  --clip_path "Skywork/SkyReels-A2"
```

### 方式 C: 不使用参考图 (回退到普通模式)
```bash
bash inference.sh
```

## 6. 显存要求

- 基础推理: ~24GB VRAM
- 带参考图推理: ~28-32GB VRAM (额外的 CLIP encoder + 双路交叉注意力)

如显存不足:
- 减少 `num_output_frames` (如 120 -> 60)
- 减少参考图数量 (3 -> 1)

## 7. 预期输出

```
加载参考图像: reference_images/human.png
加载参考图像: reference_images/thing.png
加载参考图像: reference_images/env.png
CLIP encoder 已移动到 cuda:0
...
使用 3 张参考图像进行生成
Generated 120 frames in XX.XX seconds
```

视频保存在 `videos/refimg/` 目录下。

## 8. 故障排查

### 问题: CLIP 模型加载失败
```
解决: 检查 clip_path 是否正确，确保已下载模型
```

### 问题: 参考图未找到
```
解决: 使用绝对路径，或确保相对路径正确
```

### 问题: 显存不足 (OOM)
```
解决:
1. 减少 num_output_frames
2. 使用 low_memory 模式 (代码已支持)
3. 减少参考图数量
```

### 问题: 维度不匹配
```
解决: 确保配置中 model_kwargs.use_reference_image: true
这会让模型初始化时创建必要的投影层
```
