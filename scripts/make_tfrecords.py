#!/usr/bin/env python3
"""
TFRecord 生成脚本
==================

将整理后的图像和 mask 打包为 TFRecord 文件，供后续训练使用。

输出 TFRecord 字段 (兼容 utils.py 中的 get_example / parse_example):
    - height (int64)
    - width (int64)
    - image (bytes, 原始 uint8 RGB 数据)
    - label (bytes, 原始 uint8 单通道数据)

Usage:
    # 21 类 mask
    python scripts/make_tfrecords.py --processed-root data/processed --split train --mask-type 21class --out data/tfrecords/train_21class.tfrecords
    python scripts/make_tfrecords.py --processed-root data/processed --split val --mask-type 21class --out data/tfrecords/val_21class.tfrecords

    # 二分类 mask
    python scripts/make_tfrecords.py --processed-root data/processed --split train --mask-type binary --out data/tfrecords/train_binary.tfrecords
    python scripts/make_tfrecords.py --processed-root data/processed --split val --mask-type binary --out data/tfrecords/val_binary.tfrecords
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="将图像和 mask 打包为 TFRecord 文件"
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default="data/processed",
        help="处理后的数据根目录，默认 data/processed",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test", "trainval"],
        help="数据集划分: train / val / test / trainval",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        required=True,
        choices=["21class", "binary"],
        help="Mask 类型: 21class (21类语义分割) 或 binary (前景/背景二分类)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="输出 .tfrecords 文件路径",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="如果输出文件已存在，是否覆盖 (默认不覆盖)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    processed_root = Path(args.processed_root)
    split = args.split
    mask_type = args.mask_type
    out_path = Path(args.out)
    overwrite = args.overwrite

    # 确定目录
    img_dir = processed_root / "images" / split
    if mask_type == "21class":
        mask_dir = processed_root / "masks_21class" / split
    else:
        mask_dir = processed_root / "masks_binary" / split

    # 检查源目录
    if not img_dir.is_dir():
        print(f"[ERROR] 图像目录不存在: {img_dir}")
        sys.exit(1)
    if not mask_dir.is_dir():
        print(f"[ERROR] mask 目录不存在: {mask_dir}")
        sys.exit(1)

    # 检查输出文件是否已存在
    if out_path.is_file() and not overwrite:
        print(f"[INFO] 输出文件已存在，跳过 (使用 --overwrite 覆盖): {out_path}")
        sys.exit(0)

    # 确保输出目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 收集图像文件
    img_files = sorted(img_dir.glob("*.jpg"))
    if not img_files:
        print(f"[WARNING] 在 {img_dir} 中未找到 .jpg 文件")
        sys.exit(1)

    # Lazy import: 将 TensorFlow 和 utils 的导入延迟到实际运行时，
    # 确保 --help 在不安装 TensorFlow 的环境中也能正常工作。
    import tensorflow as tf
    from utils import get_example

    print(f"[INFO] 找到 {len(img_files)} 个图像文件")
    print(f"[INFO] Mask 类型: {mask_type}")
    print(f"[INFO] 输出文件: {out_path}")

    written = 0
    skipped = 0

    with tf.io.TFRecordWriter(str(out_path)) as writer:
        for img_path in img_files:
            img_id = img_path.stem
            mask_path = mask_dir / f"{img_id}.png"

            if not mask_path.is_file():
                print(f"  [WARNING] 缺少 mask: {mask_path.name}")
                skipped += 1
                continue

            try:
                # 读取图像 (RGB)
                image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
                # 读取 mask (单通道)
                mask = np.array(Image.open(mask_path), dtype=np.uint8)

                # 确保 mask 是 (H, W, 1) shape
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                elif len(mask.shape) == 3 and mask.shape[2] > 1:
                    # 如果是多通道，取第一个通道
                    mask = mask[..., 0:1]

                # 写入 TFRecord (复用 utils.get_example)
                example = get_example(image, mask)
                writer.write(example.SerializeToString())
                written += 1
            except Exception as e:
                print(f"  [ERROR] 处理 {img_id} 失败: {e}")
                skipped += 1

    # 输出统计
    print("=" * 50)
    print(f"  Split:        {split}")
    print(f"  Mask 类型:    {mask_type}")
    print(f"  写入样本数:   {written}")
    print(f"  跳过/失败:    {skipped}")
    print(f"  输出文件:     {out_path}")
    print(f"  文件大小:     {out_path.stat().st_size / (1024*1024):.1f} MB")
    print("=" * 50)

    if written == 0:
        print("\n[ERROR] 没有成功写入任何样本！")
        sys.exit(1)
    else:
        print(f"\n[INFO] TFRecord 生成完成。")


if __name__ == "__main__":
    main()
