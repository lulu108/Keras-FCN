#!/usr/bin/env python3
"""
二分类分割指标评估脚本。

以 data/processed/masks_binary/val 作为 GT，只评估预测目录中实际存在的图片；
GT 中像素值为 255 的区域会被忽略。
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


METHODS = ("canny", "meanshift", "slic", "fcn32s", "fcn8s")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate binary segmentation predictions.")
    parser.add_argument(
        "--method",
        choices=(*METHODS, "all"),
        default="all",
        help="Prediction method to evaluate.",
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
        help="Root directory containing binary prediction subdirectories.",
    )
    parser.add_argument(
        "--runtime-csv",
        type=str,
        default="results/metrics/traditional_runtime.csv",
        help="Optional runtime CSV produced by traditional_methods.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/metrics/metrics_binary.csv",
        help="Output CSV for binary metrics.",
    )
    return parser.parse_args()


def selected_methods(method):
    return list(METHODS) if method == "all" else [method]


def read_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def resize_to_shape(mask, shape):
    if mask.shape == shape:
        return mask
    image = Image.fromarray(mask.astype(np.uint8), mode="L")
    resized = image.resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.array(resized)


def evaluate_pair(gt, pred):
    valid = gt != 255
    if not np.any(valid):
        return 0, 0, 0, 0, 0

    gt_fg = (gt > 0) & valid
    pred_fg = (pred > 0) & valid
    pred_bg = ~pred_fg & valid
    gt_bg = ~gt_fg & valid

    tp = int(np.logical_and(pred_fg, gt_fg).sum())
    tn = int(np.logical_and(pred_bg, gt_bg).sum())
    fp = int(np.logical_and(pred_fg, gt_bg).sum())
    fn = int(np.logical_and(pred_bg, gt_fg).sum())
    return tp, tn, fp, fn, int(valid.sum())


def safe_divide(numerator, denominator, empty_value=1.0):
    if denominator == 0:
        return empty_value
    return numerator / denominator


def load_runtime(runtime_csv):
    runtime_csv = Path(runtime_csv)
    if not runtime_csv.is_file():
        return {}

    runtimes = {}
    with open(runtime_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row.get("method", "")
            image_id = row.get("image_id", "")
            try:
                runtime = float(row.get("runtime_sec", ""))
            except ValueError:
                continue
            runtimes.setdefault(method, {}).setdefault(image_id, []).append(runtime)
    return runtimes


def evaluate_method(method, gt_dir, pred_root, runtime_lookup):
    pred_dir = pred_root / method
    if not pred_dir.is_dir():
        print(f"[WARN] prediction directory not found, skip: {pred_dir}")
        return None

    pred_files = sorted(p for p in pred_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not pred_files:
        print(f"[WARN] no prediction masks found, skip: {pred_dir}")
        return None

    tp = tn = fp = fn = valid_pixels = 0
    image_ids = []
    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.is_file():
            continue

        gt = read_mask(gt_path)
        pred = resize_to_shape(read_mask(pred_path), gt.shape)
        p_tp, p_tn, p_fp, p_fn, p_valid = evaluate_pair(gt, pred)
        tp += p_tp
        tn += p_tn
        fp += p_fp
        fn += p_fn
        valid_pixels += p_valid
        image_ids.append(pred_path.stem)

    if not image_ids:
        print(f"[WARN] no predictions matched GT masks, skip: {pred_dir}")
        return None

    pixel_accuracy = safe_divide(tp + tn, valid_pixels, empty_value=0.0)
    foreground_iou = safe_divide(tp, tp + fp + fn)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)

    runtimes = []
    for image_id in image_ids:
        runtimes.extend(runtime_lookup.get(method, {}).get(image_id, []))
    mean_runtime = "" if not runtimes else f"{float(np.mean(runtimes)):.6f}"

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": method,
        "num_images": len(image_ids),
        "valid_pixels": valid_pixels,
        "pixel_accuracy": f"{pixel_accuracy:.6f}",
        "foreground_iou": f"{foreground_iou:.6f}",
        "dice": f"{dice:.6f}",
        "mean_inference_time_sec": mean_runtime,
    }


def write_metrics(output_csv, rows):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "method",
        "num_images",
        "valid_pixels",
        "pixel_accuracy",
        "foreground_iou",
        "dice",
        "mean_inference_time_sec",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    pred_root = Path(args.pred_root)
    if not gt_dir.is_dir():
        print(f"[ERROR] GT directory not found: {gt_dir}")
        sys.exit(1)

    runtime_lookup = load_runtime(args.runtime_csv)
    rows = []
    for method in selected_methods(args.method):
        row = evaluate_method(method, gt_dir, pred_root, runtime_lookup)
        if row is not None:
            rows.append(row)
            print(
                f"[INFO] {method}: PA={row['pixel_accuracy']} "
                f"IoU={row['foreground_iou']} Dice={row['dice']} "
                f"N={row['num_images']}"
            )

    if not rows:
        print("[ERROR] no methods were evaluated")
        sys.exit(1)

    output_csv = Path(args.output_csv)
    write_metrics(output_csv, rows)
    print(f"[INFO] metrics written to {output_csv}")


if __name__ == "__main__":
    main()
