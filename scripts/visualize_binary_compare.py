#!/usr/bin/env python3
"""
生成传统方法与 FCN 二分类分割结果的横向对比图。

每张图包含：原图、GT binary、Canny、Mean Shift、SLIC、FCN-32s binary、FCN-8s binary。
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


METHOD_COLUMNS = [
    ("GT binary", "gt"),
    ("Canny", "canny"),
    ("Mean Shift", "meanshift"),
    ("SLIC", "slic"),
    ("FCN-32s binary", "fcn32s"),
    ("FCN-8s binary", "fcn8s"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize binary segmentation comparisons.")
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of comparison figures to generate.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/processed/images/val",
        help="Validation image directory.",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="data/processed/masks_binary/val",
        help="Ground-truth binary mask directory.",
    )
    parser.add_argument(
        "--pred-root",
        type=str,
        default="results/pred_binary",
        help="Root directory containing prediction subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visual_compare_binary",
        help="Output directory for comparison figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image selection.",
    )
    parser.add_argument(
        "--fig-dpi",
        type=int,
        default=150,
        help="Saved figure DPI.",
    )
    return parser.parse_args()


def read_binary_mask(mask_path, size):
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, resample=Image.NEAREST)
    arr = np.array(mask)
    if np.any(arr == 255):
        # GT 中的 ignore 区域显示为灰色，其余仍按二分类显示。
        display = np.zeros_like(arr, dtype=np.uint8)
        display[arr > 0] = 255
        display[arr == 255] = 128
        return display
    return ((arr > 0).astype(np.uint8) * 255)


def collect_common_ids(image_dir, gt_dir, pred_root):
    image_ids = sorted(p.stem for p in image_dir.glob("*.jpg"))
    common = []
    for image_id in image_ids:
        if not (gt_dir / f"{image_id}.png").is_file():
            continue
        has_all_predictions = all(
            (pred_root / method / f"{image_id}.png").is_file()
            for _, method in METHOD_COLUMNS
            if method != "gt"
        )
        if has_all_predictions:
            common.append(image_id)
    return common


def main():
    args = parse_args()
    random.seed(args.seed)

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    pred_root = Path(args.pred_root)
    output_dir = Path(args.output_dir)

    for directory, name in [(image_dir, "image"), (gt_dir, "GT"), (pred_root, "prediction root")]:
        if not directory.is_dir():
            print(f"[ERROR] {name} directory not found: {directory}")
            sys.exit(1)

    common_ids = collect_common_ids(image_dir, gt_dir, pred_root)
    if not common_ids:
        print("[ERROR] no image has GT plus all required prediction masks")
        sys.exit(1)

    selected = random.sample(common_ids, min(args.num_images, len(common_ids)))
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for idx, image_id in enumerate(selected, start=1):
        image = Image.open(image_dir / f"{image_id}.jpg").convert("RGB")
        size = image.size

        panels = [np.array(image)]
        titles = ["Original"]
        for title, method in METHOD_COLUMNS:
            mask_path = gt_dir / f"{image_id}.png" if method == "gt" else pred_root / method / f"{image_id}.png"
            panels.append(read_binary_mask(mask_path, size))
            titles.append(title)

        fig, axes = plt.subplots(1, len(panels), figsize=(21, 3.2))
        for ax, panel, title in zip(axes, panels, titles):
            if panel.ndim == 2:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(panel)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.tight_layout(pad=0.4)
        out_path = output_dir / f"{image_id}.png"
        fig.savefig(out_path, dpi=args.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] [{idx}/{len(selected)}] saved {out_path}")

    print(f"[INFO] visual comparisons written to {output_dir}")


if __name__ == "__main__":
    main()
