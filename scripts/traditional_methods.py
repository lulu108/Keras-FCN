#!/usr/bin/env python3
"""
传统图像分割方法的二分类预测脚本。

从验证集原图生成前景/背景 mask，用于和 FCN 二分类结果做同一批图片上的对比。
输出 mask 为单通道 PNG，像素值只包含 0 和 1。
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


METHODS = ("canny", "meanshift", "slic")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run traditional binary segmentation methods on validation images."
    )
    parser.add_argument(
        "--method",
        choices=(*METHODS, "all"),
        default="all",
        help="Traditional method to run: canny, meanshift, slic, or all.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of validation images to process. Use 0 for all images.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/processed/images/val",
        help="Validation image directory.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/pred_binary",
        help="Root directory for binary prediction masks.",
    )
    parser.add_argument(
        "--runtime-csv",
        type=str,
        default="results/metrics/traditional_runtime.csv",
        help="CSV file used to append per-image runtime records.",
    )
    return parser.parse_args()


def selected_methods(method):
    return list(METHODS) if method == "all" else [method]


def list_images(image_dir, num_images):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {image_dir}")
    if num_images > 0:
        image_files = image_files[:num_images]
    return image_files


def load_rgb(image_path):
    return np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def try_import_cv2():
    try:
        import cv2

        return cv2
    except ImportError:
        return None


def rgb_to_lab(image):
    cv2 = try_import_cv2()
    if cv2 is not None:
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    from skimage import color

    return color.rgb2lab(image).astype(np.float32)


def save_binary_mask(mask, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary = (mask > 0).astype(np.uint8)
    Image.fromarray(binary, mode="L").save(output_path)


def border_pixels(image):
    top = image[0, :, :]
    bottom = image[-1, :, :]
    left = image[:, 0, :]
    right = image[:, -1, :]
    return np.concatenate([top, bottom, left, right], axis=0)


def clean_binary_mask(mask, min_area_ratio=0.001):
    """对传统方法的粗 mask 做形态学清理，减少边缘碎片对指标的影响。"""
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    kernel_size = max(3, int(round(min(h, w) * 0.015)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    cv2 = try_import_cv2()
    if cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        min_area = max(16, int(h * w * min_area_ratio))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        kept = np.zeros_like(binary)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                kept[labels == label] = 1
        return kept

    from skimage import measure, morphology

    footprint = morphology.disk(max(1, kernel_size // 2))
    binary_bool = morphology.binary_closing(binary.astype(bool), footprint)
    binary_bool = morphology.binary_opening(binary_bool, footprint)
    min_area = max(16, int(h * w * min_area_ratio))
    labels = measure.label(binary_bool, connectivity=2)
    kept = np.zeros_like(binary, dtype=np.uint8)
    for region in measure.regionprops(labels):
        if region.area >= min_area:
            kept[labels == region.label] = 1
    return kept


def fill_holes(mask):
    """用 flood fill 填补闭合边缘内部区域，近似生成前景。"""
    binary = (mask > 0).astype(np.uint8) * 255
    cv2 = try_import_cv2()
    if cv2 is None:
        from scipy import ndimage

        return ndimage.binary_fill_holes(binary > 0).astype(np.uint8)

    h, w = binary.shape
    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return ((binary | holes) > 0).astype(np.uint8)


def background_distance_mask(image, percentile=65.0):
    """用图像边界颜色估计背景，再把颜色差异较大的区域视为候选前景。"""
    lab = rgb_to_lab(image)
    bg_color = np.median(border_pixels(lab), axis=0)
    distance = np.linalg.norm(lab - bg_color, axis=2)
    threshold = max(12.0, float(np.percentile(distance, percentile)))
    return distance > threshold


def predict_canny(image):
    cv2 = try_import_cv2()
    if cv2 is None:
        from skimage import color, feature, morphology

        gray = color.rgb2gray(image)
        edges = feature.canny(gray, sigma=1.4)
        h, w = gray.shape
        radius = max(2, int(round(min(h, w) * 0.025)))
        footprint = morphology.disk(radius)
        closed = morphology.binary_closing(morphology.binary_dilation(edges, footprint), footprint)
        return clean_binary_mask(fill_holes(closed), min_area_ratio=0.001)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)

    h, w = gray.shape
    kernel_size = max(5, int(round(min(h, w) * 0.025)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(gray, dtype=np.uint8)
    min_area = max(32, int(h * w * 0.002))
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
    if not np.any(filled):
        filled = (closed > 0).astype(np.uint8)
    return clean_binary_mask(fill_holes(filled), min_area_ratio=0.001)


def predict_meanshift(image):
    cv2 = try_import_cv2()
    if cv2 is not None:
        # Mean Shift 先平滑颜色块，再用边界背景颜色差异生成二分类前景。
        smoothed = cv2.pyrMeanShiftFiltering(image, sp=16, sr=32)
    else:
        from skimage.segmentation import quickshift

        # 本地缺少 OpenCV 时，用 quickshift 形成颜色区域，再按区域均值近似平滑结果。
        segments = quickshift(image, kernel_size=3, max_dist=12, ratio=0.7)
        smoothed = np.zeros_like(image)
        for label in np.unique(segments):
            region = segments == label
            smoothed[region] = image[region].mean(axis=0)
    mask = background_distance_mask(smoothed, percentile=62.0)
    return clean_binary_mask(mask, min_area_ratio=0.0015)


def predict_slic(image):
    try:
        from skimage.segmentation import slic
    except ImportError as exc:
        raise ImportError(
            "SLIC requires scikit-image. Install it with `pip install scikit-image`."
        ) from exc

    h, w = image.shape[:2]
    n_segments = max(80, min(320, (h * w) // 900))
    segments = slic(
        image,
        n_segments=n_segments,
        compactness=10,
        sigma=1,
        start_label=0,
        channel_axis=-1,
    )

    lab = rgb_to_lab(image)
    bg_color = np.median(border_pixels(lab), axis=0)
    cv2 = try_import_cv2()
    if cv2 is not None:
        edge_map = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 80, 160) > 0
    else:
        from skimage import color, feature

        edge_map = feature.canny(color.rgb2gray(image), sigma=1.2)

    distances = np.zeros(int(segments.max()) + 1, dtype=np.float32)
    edge_density = np.zeros_like(distances)
    for label in range(len(distances)):
        region = segments == label
        mean_color = lab[region].mean(axis=0)
        distances[label] = np.linalg.norm(mean_color - bg_color)
        edge_density[label] = float(edge_map[region].mean())

    color_threshold = max(12.0, float(np.percentile(distances, 58)))
    edge_threshold = max(0.03, float(np.percentile(edge_density, 70)))
    foreground_labels = (distances > color_threshold) | (
        (distances > color_threshold * 0.75) & (edge_density > edge_threshold)
    )
    mask = foreground_labels[segments]
    return clean_binary_mask(mask, min_area_ratio=0.0015)


PREDICTORS = {
    "canny": predict_canny,
    "meanshift": predict_meanshift,
    "slic": predict_slic,
}


def append_runtime_rows(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "method", "image_id", "runtime_sec", "image_path", "output_path"]
    file_exists = csv_path.is_file()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    methods = selected_methods(args.method)

    try:
        image_files = list_images(args.image_dir, args.num_images)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    runtime_rows = []
    print("=" * 72)
    print("Traditional binary segmentation")
    print(f"Image directory: {args.image_dir}")
    print(f"Images: {len(image_files)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Output root: {args.output_root}")
    print("=" * 72)

    for image_path in image_files:
        image = load_rgb(image_path)
        for method in methods:
            output_path = Path(args.output_root) / method / f"{image_path.stem}.png"
            start = time.perf_counter()
            try:
                mask = PREDICTORS[method](image)
            except Exception as exc:
                print(f"[ERROR] {method} failed on {image_path.name}: {exc}")
                continue
            runtime_sec = time.perf_counter() - start
            save_binary_mask(mask, output_path)
            runtime_rows.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "method": method,
                    "image_id": image_path.stem,
                    "runtime_sec": f"{runtime_sec:.6f}",
                    "image_path": str(image_path),
                    "output_path": str(output_path),
                }
            )
        print(f"[INFO] processed {image_path.name}")

    if runtime_rows:
        append_runtime_rows(args.runtime_csv, runtime_rows)
        print(f"[INFO] runtime records appended to {args.runtime_csv}")


if __name__ == "__main__":
    main()
