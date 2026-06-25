#!/usr/bin/env python3
"""
FCN 预测可视化脚本
===================

将原图、GT mask (颜色化)、预测 mask (颜色化) 拼成三列对比图并保存。

输出:
    results/visual_compare_21class/{model_name}/  — 三列对比图

Usage:
    # 使用默认 20 张随机图片
    python scripts/visualize_fcn_predictions.py --model-name fcn32s

    # 指定图片数量和来源目录
    python scripts/visualize_fcn_predictions.py --model-name fcn8s --num-images 30

    # 使用自定义路径
    python scripts/visualize_fcn_predictions.py \
        --model-name fcn32s \
        --image-dir data/processed/images/val \
        --gt-dir data/processed/masks_21class/val \
        --pred-dir results/pred_21class/fcn32s
"""

import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="FCN 预测结果三列对比可视化脚本"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="模型名称，用于定位预测目录和输出子目录 (例如 fcn32s, fcn8s)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/processed/images/val",
        help="原始图像目录，默认 data/processed/images/val",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="data/processed/masks_21class/val",
        help="GT mask 目录，默认 data/processed/masks_21class/val",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default=None,
        help="预测 mask 目录。默认自动推断为 results/pred_21class/{model_name}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visual_compare_21class",
        help="可视化输出根目录，默认 results/visual_compare_21class",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="可视化图片数量 (随机选择)，默认 20",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    parser.add_argument(
        "--fig-width",
        type=int,
        default=18,
        help="拼接图宽度 (inches)，默认 18",
    )
    parser.add_argument(
        "--fig-dpi",
        type=int,
        default=150,
        help="拼接图 DPI，默认 150",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # 推断预测目录
    if args.pred_dir is None:
        pred_dir = Path("results/pred_21class") / args.model_name
    else:
        pred_dir = Path(args.pred_dir)

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir) / args.model_name

    # ====================
    # 检查目录
    # ====================
    for d, name in [(image_dir, "原始图像"), (gt_dir, "GT mask"), (pred_dir, "预测 mask")]:
        if not d.is_dir():
            print(f"[ERROR] {name}目录未找到: {d}")
            sys.exit(1)

    # 收集共有图片 ID
    image_ids = sorted([p.stem for p in image_dir.glob("*.jpg")])

    # 筛选同时有 GT 和预测的 ID
    common_ids = []
    for img_id in image_ids:
        gt_path = gt_dir / f"{img_id}.png"
        pred_path = pred_dir / f"{img_id}.png"
        if gt_path.is_file() and pred_path.is_file():
            common_ids.append(img_id)

    if not common_ids:
        print("[ERROR] 没有同时具有原图、GT mask 和预测 mask 的样本。")
        print(f"  原图目录: {image_dir}")
        print(f"  GT 目录:  {gt_dir}")
        print(f"  预测目录: {pred_dir}")
        sys.exit(1)

    # 随机选择
    num_images = min(args.num_images, len(common_ids))
    selected_ids = random.sample(common_ids, num_images)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  FCN 预测可视化")
    print(f"  模型名称:  {args.model_name}")
    print(f"  可用样本:  {len(common_ids)}")
    print(f"  随机选择:  {num_images}")
    print(f"  输出目录:  {output_dir}")
    print("=" * 60)

    # ====================
    # 导入依赖
    # ====================
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端
    import matplotlib.pyplot as plt
    from utils import PALETTE, label_to_image

    # ====================
    # 逐张生成对比图
    # ====================
    print("\n[INFO] 生成对比图...")
    success = 0

    for i, img_id in enumerate(selected_ids):
        try:
            # 读取原图
            image = np.array(Image.open(image_dir / f"{img_id}.jpg").convert("RGB"))

            # 读取 GT mask
            gt_label = np.array(Image.open(gt_dir / f"{img_id}.png"))
            if len(gt_label.shape) == 2:
                gt_label = gt_label[..., None]
            elif gt_label.shape[2] > 1:
                gt_label = gt_label[..., 0:1]
            gt_color = label_to_image(gt_label, palette=PALETTE)

            # 读取预测 mask
            pred_label = np.array(Image.open(pred_dir / f"{img_id}.png"))
            if len(pred_label.shape) == 2:
                pred_label = pred_label[..., None]
            elif pred_label.shape[2] > 1:
                pred_label = pred_label[..., 0:1]
            pred_color = label_to_image(pred_label, palette=PALETTE)

            # 创建三列对比图
            fig, axes = plt.subplots(1, 3, figsize=(args.fig_width, args.fig_width / 3))
            titles = ["Original Image", "Ground Truth", "Prediction"]

            for ax, img_data, title in zip(
                axes,
                [image, gt_color, pred_color],
                titles,
            ):
                ax.imshow(img_data)
                ax.set_title(title, fontsize=12)
                ax.axis("off")

            plt.tight_layout()
            out_path = output_dir / f"{img_id}.png"
            fig.savefig(str(out_path), dpi=args.fig_dpi, bbox_inches="tight")
            plt.close(fig)

            success += 1
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1}/{num_images}] {img_id}.png 已保存")

        except Exception as e:
            print(f"  [ERROR] 处理 {img_id} 失败: {e}")

    print("\n" + "=" * 60)
    print(f"  可视化完成")
    print(f"  成功: {success}/{num_images}")
    print(f"  输出: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
