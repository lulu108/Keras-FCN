#!/usr/bin/env python3
"""
将 FCN 21 类预测 mask 转换为前景/背景二分类 mask。

类别 0 作为背景，类别 1-20 作为前景。输出为单通道 PNG，像素值只包含 0 和 1。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


MODELS = ("fcn32s", "fcn8s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert 21-class FCN prediction masks to binary masks."
    )
    parser.add_argument(
        "--model-name",
        choices=(*MODELS, "all"),
        default="all",
        help="Model prediction directory to convert.",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="results/pred_21class",
        help="Root directory containing 21-class prediction masks.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/pred_binary",
        help="Root directory for converted binary masks.",
    )
    return parser.parse_args()


def selected_models(model_name):
    return list(MODELS) if model_name == "all" else [model_name]


def read_label_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def convert_one_directory(input_dir, output_dir):
    """按 0/非 0 规则转换一个模型目录，保持文件名不变便于后续评估对齐。"""
    if not input_dir.is_dir():
        print(f"[WARN] prediction directory not found, skip: {input_dir}")
        return 0

    mask_files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for mask_path in mask_files:
        label = read_label_mask(mask_path)
        binary = (label > 0).astype(np.uint8)
        Image.fromarray(binary, mode="L").save(output_dir / mask_path.name)
        count += 1
    return count


def main():
    args = parse_args()
    total = 0
    for model_name in selected_models(args.model_name):
        input_dir = Path(args.input_root) / model_name
        output_dir = Path(args.output_root) / model_name
        count = convert_one_directory(input_dir, output_dir)
        total += count
        print(f"[INFO] {model_name}: converted {count} masks -> {output_dir}")

    if total == 0:
        print("[ERROR] no prediction masks were converted")
        sys.exit(1)


if __name__ == "__main__":
    main()
