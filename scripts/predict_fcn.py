#!/usr/bin/env python3
"""
FCN 预测脚本
=============

加载训练好的 FCN 模型，对 data/processed/images/val 中的图片进行语义分割预测，
并将预测 mask 保存为单通道 PNG 文件。

输出:
    results/pred_21class/{model_name}/  — 预测 mask (class 0–20)

Usage:
    # 使用 FCN-32s 预测
    python scripts/predict_fcn.py --model-path checkpoints/fcn32s_21class/best_model.h5 --model-name fcn32s

    # 使用 FCN-8s 预测，指定图像尺寸
    python scripts/predict_fcn.py --model-path checkpoints/fcn8s_21class/best_model.h5 --model-name fcn8s --image-size 320

    # 预测指定目录中的图片
    python scripts/predict_fcn.py --model-path checkpoints/fcn8s_21class/best_model.h5 --model-name fcn8s --image-dir data/processed/images/val
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="FCN 模型预测脚本 (21 类语义分割)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="训练好的模型文件路径 (.h5)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="模型名称，用于创建输出子目录 (例如 fcn32s, fcn8s)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/processed/images/val",
        help="预测图片目录，默认 data/processed/images/val",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="模型输入尺寸，默认 224",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/pred_21class",
        help="预测输出根目录，默认 results/pred_21class",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=0,
        help="只预测前 N 张图片 (0 表示全部)，默认 0",
    )
    return parser.parse_args()


def get_custom_objects():
    """返回加载模型所需的 custom_objects 字典。"""
    from models import BilinearInitializer, crossentropy, pixelacc, MyMeanIoU
    return {
        "BilinearInitializer": BilinearInitializer,
        "crossentropy": crossentropy,
        "pixelacc": pixelacc,
        "MyMeanIoU": MyMeanIoU,
    }


def preprocess_image(image_path, target_size):
    """读取并预处理单张图像。

    Args:
        image_path: 图像文件路径。
        target_size: (int) 目标尺寸。

    Returns:
        original_size: (height, width) 原始尺寸。
        input_tensor: (1, target_size, target_size, 3) float32 张量。
    """
    import tensorflow as tf
    from PIL import Image

    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    original_size = image.shape[:2]  # (height, width)

    # 缩放并增加 batch 维度
    input_tensor = tf.image.resize(image, (target_size, target_size), method="bilinear")
    input_tensor = tf.cast(input_tensor, tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    return original_size, input_tensor


def main():
    args = parse_args()

    # Lazy import
    import tensorflow as tf

    # ====================
    # 检查路径
    # ====================
    model_path = Path(args.model_path)
    image_dir = Path(args.image_dir)

    if not model_path.is_file():
        print(f"[ERROR] 模型文件未找到: {model_path}")
        sys.exit(1)
    if not image_dir.is_dir():
        print(f"[ERROR] 图片目录未找到: {image_dir}")
        sys.exit(1)

    # 收集图像文件
    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        print(f"[ERROR] 在 {image_dir} 中未找到 .jpg 文件")
        sys.exit(1)

    if args.num_images > 0:
        image_files = image_files[: args.num_images]

    # 输出目录
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  FCN 预测")
    print(f"  模型:      {model_path}")
    print(f"  模型名称:  {args.model_name}")
    print(f"  输入目录:  {image_dir}")
    print(f"  图片数量:  {len(image_files)}")
    print(f"  图像尺寸:  {args.image_size}x{args.image_size}")
    print(f"  输出目录:  {output_dir}")
    print("=" * 60)

    # ====================
    # 加载模型
    # ====================
    print("\n[INFO] 加载模型...")
    custom_objects = get_custom_objects()
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects=custom_objects,
        compile=False,
    )

    # ====================
    # 逐张预测
    # ====================
    print("[INFO] 开始预测...")
    success = 0
    failed = 0

    for i, img_path in enumerate(image_files):
        try:
            img_id = img_path.stem
            original_size, input_tensor = preprocess_image(img_path, args.image_size)

            # 推理
            pred = model.predict(input_tensor, verbose=0)  # (1, H, W, 21)
            pred_label = np.argmax(pred[0], axis=-1).astype(np.uint8)  # (H, W)

            # 缩放回原始尺寸
            pred_label_resized = tf.image.resize(
                pred_label[..., None],
                original_size,
                method="nearest",
            ).numpy().astype(np.uint8)[..., 0]

            # 保存为 PNG
            from PIL import Image
            out_path = output_dir / f"{img_id}.png"
            Image.fromarray(pred_label_resized).save(str(out_path))

            success += 1
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(image_files)}] {img_id}.png 已保存")

        except Exception as e:
            print(f"  [ERROR] 处理 {img_path.name} 失败: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  预测完成")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")
    print(f"  输出: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
