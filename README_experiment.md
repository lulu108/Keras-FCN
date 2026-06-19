# Keras-FCN 实验工程化改造

## 实验目标

本实验基于 Pascal VOC 2012 数据集，完成以下目标：

1. **深度语义分割 (21 类)**: 使用 Pascal VOC 原始 21 类语义分割标签 (0=背景, 1–20=前景物体, 255=ignore/boundary) 训练 FCN-32s 和 FCN-8s 模型。

2. **前景/背景二分类对比**: 将 VOC 21 类标签转换为前景/背景二分类标签 (0=背景, 1=前景, 255=ignore)，用于传统图像分割方法 (Canny, Mean Shift, SLIC) 与深度方法 (FCN) 的公平对比。

3. **多维度评价**: 在 21 类和二分类两个任务上分别评估像素准确率、Mean IoU、边界质量等指标。

## 当前阶段说明

**本阶段 (Phase 1)** 只完成数据准备、目录结构搭建和基础脚本实现，**不包含模型训练**。

后续阶段将包含:
- Phase 2: FCN-32s / FCN-8s 模型训练与 21 类评价
- Phase 3: 传统方法 (Canny, Mean Shift, SLIC) 实现与二分类评价
- Phase 4: 综合对比分析与报告

## Pascal VOC 数据放置

用户需手动将 Pascal VOC 2012 数据集放到以下位置:

```
data/VOCdevkit/VOC2012/
├── JPEGImages/           # 原始图像 (.jpg)
├── SegmentationClass/    # 21 类语义分割标签 (.png)
└── ImageSets/
    └── Segmentation/
        ├── train.txt     # 训练集 ID 列表
        ├── val.txt       # 验证集 ID 列表
        └── trainval.txt  # 训练+验证集 ID 列表
```

数据集可从 [Pascal VOC 官网](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 下载。

## 目录结构

```
Keras-FCN/
├── data/
│   ├── VOCdevkit/              # Pascal VOC 原始数据 (用户手动放置)
│   ├── processed/              # 处理后的标准化数据
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── masks_21class/     # 21 类语义分割 mask
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── masks_binary/      # 前景/背景二分类 mask
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── tfrecords/             # TFRecord 格式训练数据
│
├── scripts/                    # 数据准备脚本
│   ├── prepare_voc.py
│   ├── convert_voc_to_binary.py
│   ├── check_voc_dataset.py
│   └── make_tfrecords.py
│
├── checkpoints/               # 模型权重存放
├── logs/                      # 训练日志
├── results/                   # 预测与评价结果
│   ├── pred_21class/
│   ├── pred_binary/
│   ├── visual_compare_21class/
│   ├── visual_compare_binary/
│   ├── boundary_overlay/
│   ├── error_maps/
│   └── metrics/
└── report/                    # 实验报告
    ├── figures/
    └── tables/
```

## 数据准备命令

### 第一步: 准备 VOC 数据 (复制图像和 21 类 mask)

```bash
# 准备训练集
python scripts/prepare_voc.py \
  --voc-root data/VOCdevkit/VOC2012 \
  --out-root data/processed \
  --split train

# 准备验证集
python scripts/prepare_voc.py \
  --voc-root data/VOCdevkit/VOC2012 \
  --out-root data/processed \
  --split val
```

**参数说明**:
- `--voc-root`: VOC2012 根目录
- `--out-root`: 输出目录，默认 `data/processed`
- `--split`: `train` / `val` / `test` / `trainval`
- `--copy-mode`: `copy` (默认) 或 `symlink`
- `--overwrite`: 是否覆盖已存在的文件 (默认不覆盖)

### 第二步: 生成二分类 mask

```bash
# 生成训练集二分类 mask
python scripts/convert_voc_to_binary.py \
  --mask-root data/processed/masks_21class \
  --out-root data/processed/masks_binary \
  --split train

# 生成验证集二分类 mask
python scripts/convert_voc_to_binary.py \
  --mask-root data/processed/masks_21class \
  --out-root data/processed/masks_binary \
  --split val
```

**转换规则**:
- 像素值 0 (背景) → 0
- 像素值 1–20 (前景物体) → 1
- 像素值 255 (ignore/boundary) → 255 (保留不变)

### 第三步: 检查数据集完整性

```bash
# 检查训练集
python scripts/check_voc_dataset.py \
  --processed-root data/processed \
  --split train

# 检查验证集
python scripts/check_voc_dataset.py \
  --processed-root data/processed \
  --split val
```

**检查内容**:
- 图像与 mask 文件名是否一一对应
- binary mask 是否覆盖
- 随机抽查样本的格式 (RGB、单通道、像素值范围、宽高一致性)

### 第四步: 生成 TFRecord 文件 (后续训练阶段使用)

```bash
# 21 类 mask - 训练集
python scripts/make_tfrecords.py \
  --processed-root data/processed \
  --split train \
  --mask-type 21class \
  --out data/tfrecords/train_21class.tfrecords

# 21 类 mask - 验证集
python scripts/make_tfrecords.py \
  --processed-root data/processed \
  --split val \
  --mask-type 21class \
  --out data/tfrecords/val_21class.tfrecords

# 二分类 mask - 训练集
python scripts/make_tfrecords.py \
  --processed-root data/processed \
  --split train \
  --mask-type binary \
  --out data/tfrecords/train_binary.tfrecords

# 二分类 mask - 验证集
python scripts/make_tfrecords.py \
  --processed-root data/processed \
  --split val \
  --mask-type binary \
  --out data/tfrecords/val_binary.tfrecords
```

## 实验数据流

```
Pascal VOC 原始数据 (data/VOCdevkit/VOC2012/)
    │
    ▼
prepare_voc.py ──► data/processed/images/{split}/
                └─► data/processed/masks_21class/{split}/
                        │
                        ▼
            convert_voc_to_binary.py ──► data/processed/masks_binary/{split}/
                                                │
                        ┌───────────────────────┘
                        ▼
            make_tfrecords.py ──► data/tfrecords/{split}_{mask_type}.tfrecords
                        │
                        ▼
                后续模型训练与评价
```

## 传统方法对比说明

Canny 边缘检测、Mean Shift 聚类、SLIC 超像素分割等传统方法**不能直接识别 VOC 的 21 个语义类别** (如区分"猫"和"狗")。

因此，后续实验设计中:
- **21 类任务**: 仅评估 FCN-32s / FCN-8s 的语义分割能力
- **二分类任务**: 将传统方法与 FCN 方法统一在前景/背景分割任务上评价

这样可以在"是否能找出前景物体"这一基本能力上公平比较传统方法和深度学习方法。

## .gitignore 说明

本仓库**没有新增或修改 .gitignore**。

原因: `data/`、`logs/`、`results/`、`checkpoints/` 等目录中的内容需要保留在仓库中或已由用户上传，不应被忽略规则排除。

## 脚本帮助

所有脚本都支持 `--help` 查看详细参数说明:

```bash
python scripts/prepare_voc.py --help
python scripts/convert_voc_to_binary.py --help
python scripts/check_voc_dataset.py --help
python scripts/make_tfrecords.py --help
```

## 技术说明

- 所有脚本使用 `argparse` 进行参数解析
- 所有路径操作使用 `pathlib.Path`
- TFRecord 格式兼容仓库已有 `utils.py` 中的 `get_example()` / `parse_example()` 函数
- 不破坏原仓库 `models.py`、`utils.py`、`augment.py`、`train.ipynb` 的原有功能
- 不删除或覆盖仓库中已有数据、日志、结果和模型权重
