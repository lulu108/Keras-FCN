#!/usr/bin/env python3
"""
FCN-8s 训练脚本
================

使用 models.py 中的 vgg16 + fcn32 + fcn16 + fcn8 构建 FCN-8s 语义分割模型，
从 TFRecord 读取 21 类 VOC 数据进行训练与验证。

输出:
    - checkpoint 模型: checkpoints/fcn8s_21class/
    - 训练日志 (控制台输出): loss, pixel accuracy, mean IoU

Usage:
    # Smoke test (CPU 可运行)
    python scripts/train_fcn8s.py --epochs 1 --batch-size 1 --image-size 224

    # 完整训练
    python scripts/train_fcn8s.py --epochs 50 --batch-size 4 --image-size 320 --lr 1e-4
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="FCN-8s 语义分割训练脚本 (21 类)"
    )
    # 数据
    parser.add_argument(
        "--train-record",
        type=str,
        default="data/tfrecords/train_21class.tfrecords",
        help="训练 TFRecord 路径，默认 data/tfrecords/train_21class.tfrecords",
    )
    parser.add_argument(
        "--val-record",
        type=str,
        default="data/tfrecords/val_21class.tfrecords",
        help="验证 TFRecord 路径，默认 data/tfrecords/val_21class.tfrecords",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="训练时统一缩放到的图像尺寸，默认 224",
    )
    # 训练超参
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="训练轮数，默认 1 (smoke test 配置)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批次大小，默认 1 (smoke test 配置)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率，默认 1e-4",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.0,
        help="L2 正则化强度，默认 0",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout 比率，默认 0",
    )
    # 输出
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/fcn8s_21class",
        help="模型保存目录，默认 checkpoints/fcn8s_21class",
    )
    return parser.parse_args()


def build_dataset(tfrecord_path, image_size, batch_size, shuffle=False):
    """从 TFRecord 构建 tf.data.Dataset，完成解析、缩放和批处理。

    Args:
        tfrecord_path: TFRecord 文件路径。
        image_size: 统一缩放的目标尺寸 (int, 正方形)。
        batch_size: 批次大小。
        shuffle: 是否打乱数据。

    Returns:
        tf.data.Dataset，每个元素为 (image, label) 元组。
    """
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

    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_fcn8s_model(image_size, num_classes=21, l2=0.0, dropout=0.0):
    """构建 FCN-8s 模型: vgg16 → fcn32 → fcn16 → fcn8。

    Args:
        image_size: 输入图像尺寸。
        num_classes: 类别数，默认 21。
        l2: L2 正则化强度。
        dropout: Dropout 比率。

    Returns:
        (keras Model) FCN-8s 模型。
    """
    from models import vgg16, fcn32, fcn16, fcn8

    vgg16_model = vgg16(l2=l2, dropout=dropout)
    fcn32_model = fcn32(vgg16_model, l2=l2)
    fcn16_model = fcn16(vgg16_model, fcn32_model, l2=l2)
    model = fcn8(vgg16_model, fcn16_model, l2=l2)
    return model


def main():
    args = parse_args()

    # Lazy import: 将 TensorFlow 及项目模块的导入延迟到参数解析之后，
    # 确保 --help 在不安装 TensorFlow 的环境中也能正常工作。
    import tensorflow as tf
    from models import crossentropy, pixelacc, MyMeanIoU

    # ====================
    # 检查数据文件
    # ====================
    train_record = Path(args.train_record)
    val_record = Path(args.val_record)

    if not train_record.is_file():
        print(f"[ERROR] 训练 TFRecord 未找到: {train_record}")
        sys.exit(1)
    if not val_record.is_file():
        print(f"[ERROR] 验证 TFRecord 未找到: {val_record}")
        sys.exit(1)

    print("=" * 60)
    print(f"  FCN-8s 训练")
    print(f"  训练数据: {train_record}")
    print(f"  验证数据: {val_record}")
    print(f"  图像尺寸: {args.image_size}x{args.image_size}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  LR:        {args.lr}")
    print(f"  L2:        {args.l2}")
    print(f"  Dropout:   {args.dropout}")
    print(f"  保存目录:  {args.save_dir}")
    print("=" * 60)

    # ====================
    # 构建数据集
    # ====================
    print("\n[INFO] 加载数据集...")
    train_dataset = build_dataset(
        str(train_record), args.image_size, args.batch_size, shuffle=True
    )
    val_dataset = build_dataset(
        str(val_record), args.image_size, args.batch_size, shuffle=False
    )

    # ====================
    # 构建模型
    # ====================
    print("[INFO] 构建 FCN-8s 模型...")
    strategy = tf.distribute.MirroredStrategy() if tf.config.list_physical_devices("GPU") else tf.distribute.get_strategy()

    with strategy.scope():
        model = build_fcn8s_model(
            image_size=args.image_size,
            num_classes=21,
            l2=args.l2,
            dropout=args.dropout,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=crossentropy,
            metrics=[pixelacc, MyMeanIoU(num_classes=21, name="mean_iou")],
        )

    model.summary()

    # ====================
    # 回调
    # ====================
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / "best_model.h5"),
            monitor="val_pixelacc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(save_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")),
        ),
    ]

    # ====================
    # 训练
    # ====================
    print("\n[INFO] 开始训练...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ====================
    # 训练总结
    # ====================
    print("\n" + "=" * 60)
    print("  训练完成")
    if history.epoch:
        best_pixelacc = max(history.history.get("val_pixelacc", [0]))
        best_miou = max(history.history.get("val_mean_iou", [0]))
        print(f"  Best val pixel accuracy: {best_pixelacc:.4f}")
        print(f"  Best val mean IoU:       {best_miou:.4f}")
    print(f"  模型保存至: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
