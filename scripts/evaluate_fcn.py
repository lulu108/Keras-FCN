#!/usr/bin/env python3
"""
FCN 评估脚本
=============

加载训练好的 FCN 模型，在验证集 TFRecord 上计算 Pixel Accuracy 和 Mean IoU，
并将结果写入 results/metrics/metrics_21class.csv。

Usage:
    # 评估 FCN-32s
    python scripts/evaluate_fcn.py --model-path checkpoints/fcn32s_21class/best_model.h5 --model-name fcn32s

    # 评估 FCN-8s
    python scripts/evaluate_fcn.py --model-path checkpoints/fcn8s_21class/best_model.h5 --model-name fcn8s

    # 指定图像尺寸
    python scripts/evaluate_fcn.py --model-path checkpoints/fcn8s_21class/best_model.h5 --model-name fcn8s --image-size 320
"""

import argparse
import os
import sys
import csv
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="FCN 模型验证集评估脚本"
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
        default="fcn",
        help="模型名称标识，用于 CSV 记录，默认 fcn",
    )
    parser.add_argument(
        "--val-record",
        type=str,
        default="data/tfrecords/val_21class.tfrecords",
        help="验证 TFRecord 路径，默认 data/tfrecords/val_21class.tfrecords",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=21,
        help="类别数，默认 21",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="评估时统一缩放到的图像尺寸，默认 224",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="评估批次大小，默认 4",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/metrics/metrics_21class.csv",
        help="评估结果 CSV 输出路径，默认 results/metrics/metrics_21class.csv",
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


def build_val_dataset(tfrecord_path, image_size, batch_size):
    """从 TFRecord 构建验证数据集。"""
    import tensorflow as tf
    from utils import parse_example

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    def _preprocess(image, label):
        image = tf.image.resize(image, (image_size, image_size), method="bilinear")
        label = tf.image.resize(label, (image_size, image_size), method="nearest")
        image = tf.cast(image, tf.float32)
        return image, label

    dataset = dataset.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    args = parse_args()

    # Lazy import
    import tensorflow as tf
    from models import crossentropy, pixelacc, MyMeanIoU

    # ====================
    # 检查文件
    # ====================
    model_path = Path(args.model_path)
    val_record = Path(args.val_record)

    if not model_path.is_file():
        print(f"[ERROR] 模型文件未找到: {model_path}")
        sys.exit(1)
    if not val_record.is_file():
        print(f"[ERROR] 验证 TFRecord 未找到: {val_record}")
        sys.exit(1)

    print("=" * 60)
    print(f"  FCN 模型评估")
    print(f"  模型:      {model_path}")
    print(f"  模型名称:  {args.model_name}")
    print(f"  验证数据:  {val_record}")
    print(f"  图像尺寸:  {args.image_size}x{args.image_size}")
    print(f"  类别数:    {args.num_classes}")
    print("=" * 60)

    # ====================
    # 构建验证数据集
    # ====================
    print("\n[INFO] 加载验证数据集...")
    val_dataset = build_val_dataset(
        str(val_record), args.image_size, args.batch_size
    )

    # ====================
    # 加载模型
    # ====================
    print("[INFO] 加载模型...")
    custom_objects = get_custom_objects()
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects=custom_objects,
        compile=True,
    )

    # ====================
    # 编译模型（确保 loss/metrics 正确）
    # ====================
    # 如果加载后没有 loss，重新编译
    if model.loss is None:
        model.compile(
            loss=crossentropy,
            metrics=[pixelacc, MyMeanIoU(num_classes=args.num_classes, name="mean_iou")],
        )

    # ====================
    # 评估
    # ====================
    print("[INFO] 开始评估...")
    results = model.evaluate(val_dataset, verbose=1)

    # results 的格式: [loss, pixelacc, mean_iou] (取决于编译时的 metrics)
    loss = results[0]
    pixel_acc = results[1] if len(results) > 1 else 0.0
    mean_iou = results[2] if len(results) > 2 else 0.0

    print("\n" + "=" * 60)
    print("  评估结果")
    print(f"  Loss:              {loss:.4f}")
    print(f"  Pixel Accuracy:    {pixel_acc:.4f}")
    print(f"  Mean IoU:          {mean_iou:.4f}")
    print("=" * 60)

    # ====================
    # 写入 CSV
    # ====================
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_csv.is_file()
    fieldnames = ["timestamp", "model_name", "model_path", "image_size",
                  "num_classes", "loss", "pixel_accuracy", "mean_iou"]

    with open(str(output_csv), "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": args.model_name,
            "model_path": str(model_path),
            "image_size": args.image_size,
            "num_classes": args.num_classes,
            "loss": f"{loss:.6f}",
            "pixel_accuracy": f"{pixel_acc:.6f}",
            "mean_iou": f"{mean_iou:.6f}",
        })

    print(f"\n[INFO] 评估结果已写入: {output_csv}")


if __name__ == "__main__":
    main()
