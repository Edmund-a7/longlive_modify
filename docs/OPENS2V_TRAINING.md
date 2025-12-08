# OpenS2V 训练指南

本文档介绍如何使用 OpenS2V 数据集训练参考图功能。

## 训练目标

- **输入**: subject_image + text
- **目标**: 生成的视频，包含参考图中的主体
- **数据**: is_cross_frame=False
- **训练**: 冻结原模型，只训练新模块
- **验证**: 生成不再是噪声，主体基本能出现

## 文件说明

### 新增文件

1. **数据加载器** - [utils/dataset_opens2v.py](utils/dataset_opens2v.py)
   - `OpenS2VAdapterDataset`: 将 OpenS2V 数据转换为训练格式
   - `MultiPartOpenS2VAdapterDataset`: 支持多数据集部分
   - `collate_opens2v_batch`: 自定义 collate 函数
   - `create_opens2v_dataloader`: 创建 DataLoader

2. **训练器** - [trainer/opens2v_trainer.py](trainer/opens2v_trainer.py)
   - `OpenS2VTrainer`: 专门用于 OpenS2V 数据集的训练器
   - 自动冻结原模型，只训练参考图相关的新增层
   - 支持 FSDP 分布式训练

3. **配置文件** - [configs/train_opens2v.yaml](configs/train_opens2v.yaml)
   - 完整的训练配置
   - 包含数据路径、模型参数、训练超参数等

4. **训练脚本** - [train_opens2v.py](train_opens2v.py)
   - 训练入口点
   - 支持单机/多机分布式训练

5. **文档** - [docs/OPENS2V_TRAINING.md](docs/OPENS2V_TRAINING.md)
   - 本文档

## 快速开始

### 1. 准备数据

确保你已经处理好 OpenS2V 数据集，目录结构如下：

```
OpenS2V/
├── demo_result/
│   ├── step0/videos/dataset1/          # 视频文件
│   ├── step5/
│   │   ├── merge_final_json/
│   │   │   └── dataset1.json           # 数据标注
│   │   └── final_output/dataset1/
│   │       └── foreground/             # 背景图片
│   └── step6/cross-frames-pairs/       # 跨帧数据 (可选)
```

### 2. 修改配置文件

编辑 [configs/train_opens2v.yaml](configs/train_opens2v.yaml)，修改以下路径：

```yaml
# 数据路径
opens2v:
  json_paths:
    - "/path/to/OpenS2V/demo_result/step5/merge_final_json/dataset1.json"
  video_base_paths:
    - "/path/to/OpenS2V/demo_result/step0/videos/dataset1"
  background_base_paths:
    - "/path/to/OpenS2V/demo_result/step5/final_output/dataset1/foreground"

# 模型检查点 (包含双路交叉注意力的基础模型)
generator_ckpt: "/path/to/dual_crossattn_checkpoint.pt"
```

### 3. 启动训练

#### 单机 8 卡训练

```bash
torchrun --nproc_per_node=8 train_opens2v.py \
    --config_path configs/train_opens2v.yaml \
    --logdir outputs/opens2v_training
```

#### 多机训练 (例如 2 台机器，每台 8 卡)

**节点 0:**
```bash
torchrun --nnodes=2 --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_opens2v.py \
    --config_path configs/train_opens2v.yaml \
    --logdir outputs/opens2v_training
```

**节点 1:**
```bash
torchrun --nnodes=2 --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_opens2v.py \
    --config_path configs/train_opens2v.yaml \
    --logdir outputs/opens2v_training
```

### 4. 监控训练

训练日志会保存在 `outputs/opens2v_training/`：

```
outputs/opens2v_training/
├── config.yaml                          # 配置副本
├── checkpoint_opens2v_000500/           # 检查点
│   └── model.pt
├── checkpoint_opens2v_001000/
│   └── model.pt
└── ...
```

如果启用了 WandB，可以在 WandB 网页界面监控训练进度。

## 配置说明

### 数据配置

```yaml
opens2v:
  # 数据文件路径 (支持多个数据集)
  json_paths:
    - "path/to/dataset1.json"
    - "path/to/dataset2.json"  # 可选
  video_base_paths:
    - "path/to/videos/dataset1"
    - "path/to/videos/dataset2"  # 可选
  background_base_paths:
    - "path/to/backgrounds/dataset1"
    - "path/to/backgrounds/dataset2"  # 可选

  # 数据采样参数
  sample_stride: 3                       # 帧采样步长
  max_subjects_per_sample: 1             # 每个样本使用的 subject 数量
  subject_selection: "first"             # subject 选择策略: 'first', 'random', 'best_score'

# 是否使用跨帧数据
is_cross_frame: false
```

**Subject 选择策略:**
- `first`: 选择第一个 subject
- `random`: 随机选择
- `best_score`: 根据美学分数选择最佳 subject

### 训练参数

```yaml
# 视频参数
num_training_frames: 21                  # 训练帧数 (建议 21，OpenS2V 默认 49)
pixel_height: 480
pixel_width: 832

# 优化器
lr: 1.0e-4                               # 学习率
gradient_accumulation_steps: 1           # 梯度累积
max_grad_norm: 1.0                       # 梯度裁剪

# 训练轮次
max_iters: 10000                         # 最大训练步数
log_iters: 500                           # 保存检查点间隔
batch_size: 1                            # 批次大小
```

### 模型冻结策略

训练器会自动冻结以下模块：
- Text encoder (完全冻结)
- VAE (完全冻结)
- Generator 主体 (大部分冻结)

只训练以下新增层：
- `clip_proj`: CLIP 嵌入投影层 (1280 -> 1536)
- `vae_proj`: VAE latent 投影层 (16 -> 1536)
- `k_vae`, `v_vae`, `norm_k_vae`: VAE 路径交叉注意力

可选训练 (默认关闭):
- `k_fused`, `v_fused`, `norm_k_fused`: Fused 路径交叉注意力

通过设置 `train_fused_path: true` 来启用。

### LoRA 微调 (可选)

如果想额外应用 LoRA 微调，取消配置文件中的注释：

```yaml
adapter:
  adapter_type: "lora"
  r: 64
  lora_alpha: 64
  lora_dropout: 0.0
  target_modules: ["to_q", "to_k", "to_v", "to_out"]
```

## 训练流程

1. **加载基础模型**: 加载包含双路交叉注意力的完整模型
2. **冻结参数**: 冻结除新增层外的所有参数
3. **数据加载**: 从 OpenS2V 数据集加载 subject_image + text + video
4. **前向传播**:
   - 编码文本 (text encoder)
   - 编码参考图 (CLIP + VAE)
   - 编码视频为 latent (VAE)
   - Generator 生成预测
5. **计算损失**: Flow Matching Loss
6. **反向传播**: 只更新新增层的梯度
7. **保存检查点**: 每 `log_iters` 步保存一次

## 常见问题

### Q: 显存不足怎么办？

A: 可以尝试以下方法：
1. 减少 `num_training_frames` (例如从 21 降到 13)
2. 使用 `gradient_accumulation_steps` 来模拟更大的 batch size
3. 启用 FSDP 的 CPU offload (需要修改代码)

### Q: 如何恢复训练？

A: 默认会自动恢复最新的检查点。如果要禁用，使用 `--no-auto-resume` 参数。

### Q: 如何使用多个数据集？

A: 在配置文件中添加多个路径：

```yaml
opens2v:
  json_paths:
    - "path/to/dataset1.json"
    - "path/to/dataset2.json"
  video_base_paths:
    - "path/to/videos/dataset1"
    - "path/to/videos/dataset2"
  background_base_paths:
    - "path/to/backgrounds/dataset1"
    - "path/to/backgrounds/dataset2"
```

### Q: 训练完成后如何使用模型？

A: 加载训练好的检查点，然后使用推理脚本：

```python
# 加载检查点
checkpoint = torch.load("outputs/opens2v_training/checkpoint_opens2v_010000/model.pt")
model.generator.load_state_dict(checkpoint["generator"], strict=False)

# 使用模型进行推理
# (参考 inference.py)
```

## 性能优化建议

1. **数据预处理**: OpenS2V 数据集已经预处理好，无需额外操作
2. **批次大小**: 建议先用 `batch_size=1`，确保稳定后再增加
3. **帧数**: 从较少的帧数 (例如 13) 开始训练，稳定后增加到 21
4. **学习率**: 建议从 `1e-4` 开始，根据损失曲线调整
5. **梯度累积**: 如果显存不足，使用 `gradient_accumulation_steps=2` 或 `4`

## 代码结构

```
longlive_modify/
├── train_opens2v.py                    # 训练入口
├── configs/
│   └── train_opens2v.yaml              # 训练配置
├── trainer/
│   └── opens2v_trainer.py              # OpenS2V 训练器
├── utils/
│   └── dataset_opens2v.py              # OpenS2V 数据加载器
├── model/
│   └── refimg_flow_matching.py         # 参考图 Flow Matching 模型
└── docs/
    └── OPENS2V_TRAINING.md             # 本文档
```

## 参考

- OpenS2V-Nexus: [OpenS2V-Nexus/data_process/demo_dataloader.py](../OpenS2V-Nexus/data_process/demo_dataloader.py)
- longlive_modify 原始训练代码: [train_refimg.py](../train_refimg.py)
