#!/usr/bin/env python3
"""
VOC 数据集完整性检查脚本
=========================

检查整理后的数据是否满足语义分割实验要求:
    1. images/{split}/ 与 masks_21class/{split}/ 文件名是否一一对应
    2. masks_binary/{split}/ 是否存在对应文件
    3. 随机抽查样本的质量 (RGB、单通道、像素值范围、宽高一致性)
    4. 输出统计信息

如果有严重错误，返回非 0 exit code。

Usage:
    python scripts/check_voc_dataset.py --processed-root data/processed --split train
    python scripts/check_voc_dataset.py --processed-root data/processed --split val
    python scripts/check_voc_dataset.py --processed-root data/processed --split train --sample-count 20
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查 VOC 处理后数据集完整性"
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
        "--sample-count",
        type=int,
        default=10,
        help="随机抽查样本数量，默认 10",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    processed_root = Path(args.processed_root)
    split = args.split
    sample_count = args.sample_count

    img_dir = processed_root / "images" / split
    mask21_dir = processed_root / "masks_21class" / split
    mask_bin_dir = processed_root / "masks_binary" / split

    errors = 0
    warnings = 0

    # ================================================
    # 1. 检查目录是否存在
    # ================================================
    print("=" * 60)
    print(f"  检查 split: {split}")
    print(f"  图像目录:   {img_dir}")
    print(f"  21类mask:   {mask21_dir}")
    print(f"  二分类mask: {mask_bin_dir}")
    print("=" * 60)

    if not img_dir.is_dir():
        print(f"[ERROR] 图像目录不存在: {img_dir}")
        errors += 1
    if not mask21_dir.is_dir():
        print(f"[ERROR] 21类 mask 目录不存在: {mask21_dir}")
        errors += 1

    # ================================================
    # 2. 收集文件列表
    # ================================================
    img_files = sorted(img_dir.glob("*.jpg")) if img_dir.is_dir() else []
    mask21_files = sorted(mask21_dir.glob("*.png")) if mask21_dir.is_dir() else []
    mask_bin_files = sorted(mask_bin_dir.glob("*.png")) if mask_bin_dir.is_dir() else []

    img_names = {f.stem for f in img_files}
    mask21_names = {f.stem for f in mask21_files}
    mask_bin_names = {f.stem for f in mask_bin_files}

    n_images = len(img_files)
    n_masks21 = len(mask21_files)
    n_masks_bin = len(mask_bin_files)

    # ================================================
    # 3. 文件名对应检查
    # ================================================
    print(f"\n[检查] 文件名对应关系:")

    # images 和 masks_21class 是否一一对应
    only_in_images = img_names - mask21_names
    only_in_masks21 = mask21_names - img_names
    common_21 = img_names & mask21_names

    if only_in_images:
        print(f"  [WARNING] 只有图像、缺少 21 类 mask: {len(only_in_images)} 个")
        if len(only_in_images) <= 10:
            for name in sorted(only_in_images):
                print(f"    - {name}")
        warnings += len(only_in_images)
    if only_in_masks21:
        print(f"  [WARNING] 只有 21 类 mask、缺少图像: {len(only_in_masks21)} 个")
        if len(only_in_masks21) <= 10:
            for name in sorted(only_in_masks21):
                print(f"    - {name}")
        warnings += len(only_in_masks21)

    # masks_binary 是否覆盖
    if mask_bin_dir.is_dir():
        only_in_images_for_bin = img_names - mask_bin_names
        if only_in_images_for_bin:
            print(f"  [WARNING] 缺少对应 binary mask: {len(only_in_images_for_bin)} 个")
            warnings += len(only_in_images_for_bin)
    else:
        print(f"  [INFO] 二分类 mask 目录不存在，跳过 binary 对应检查")

    # ================================================
    # 4. 随机抽查
    # ================================================
    print(f"\n[检查] 随机抽查 {min(sample_count, len(common_21))} 个样本:")

    common_list = sorted(common_21)
    if not common_list:
        print("  [WARNING] 没有共同的 image-mask 对可以抽查")
        warnings += 1
        sample_list = []
    else:
        if len(common_list) <= sample_count:
            sample_list = common_list
        else:
            random.seed(42)
            sample_list = random.sample(common_list, sample_count)

    for name in sample_list:
        img_path = img_dir / f"{name}.jpg"
        mask21_path = mask21_dir / f"{name}.png"
        mask_bin_path = mask_bin_dir / f"{name}.png"

        print(f"\n  --- 样本: {name} ---")

        # 检查图像
        try:
            img = Image.open(img_path)
            img_arr = np.array(img)
            if img.mode != "RGB":
                print(f"    [WARNING] 图像模式不是 RGB: {img.mode}")
                warnings += 1
            if len(img_arr.shape) != 3 or img_arr.shape[2] != 3:
                print(f"    [WARNING] 图像 shape 不是 (H, W, 3): {img_arr.shape}")
                warnings += 1
            else:
                print(f"    图像 OK: shape={img_arr.shape}, mode={img.mode}")
            img_h, img_w = img_arr.shape[:2]
        except Exception as e:
            print(f"    [ERROR] 无法读取图像: {e}")
            errors += 1
            continue

        # 检查 21 类 mask
        try:
            mask21 = np.array(Image.open(mask21_path))
            mask21_vals = set(np.unique(mask21))
            valid_21 = set(range(0, 21)) | {255}
            abnormal_21 = mask21_vals - valid_21
            if abnormal_21:
                print(f"    [WARNING] 21类 mask 包含异常像素值: {sorted(abnormal_21)}")
                warnings += 1
            else:
                print(f"    21类 mask OK: shape={mask21.shape}, 像素值={sorted(mask21_vals)}")
            m21_h, m21_w = mask21.shape[:2]
        except Exception as e:
            print(f"    [ERROR] 无法读取 21 类 mask: {e}")
            errors += 1
            continue

        # 检查 binary mask
        if mask_bin_path.is_file():
            try:
                mask_bin = np.array(Image.open(mask_bin_path))
                mask_bin_vals = set(np.unique(mask_bin))
                valid_bin = {0, 1, 255}
                abnormal_bin = mask_bin_vals - valid_bin
                if abnormal_bin:
                    print(f"    [WARNING] binary mask 包含非 {sorted(valid_bin)} 像素值: {sorted(abnormal_bin)}")
                    warnings += 1
                else:
                    print(f"    binary mask OK: shape={mask_bin.shape}, 像素值={sorted(mask_bin_vals)}")
                mbin_h, mbin_w = mask_bin.shape[:2]
            except Exception as e:
                print(f"    [ERROR] 无法读取 binary mask: {e}")
                errors += 1
        else:
            print(f"    [INFO] binary mask 不存在: {mask_bin_path.name}")

        # 检查宽高一致性
        if img_h != m21_h or img_w != m21_w:
            print(f"    [ERROR] 图像和 21 类 mask 尺寸不一致: image=({img_h},{img_w}), mask=({m21_h},{m21_w})")
            errors += 1

    # ================================================
    # 5. 总结
    # ================================================
    print("\n" + "=" * 60)
    print(f"  统计总结:")
    print(f"    图像数量:       {n_images}")
    print(f"    21类 mask 数量: {n_masks21}")
    print(f"    binary mask 数量: {n_masks_bin}")
    print(f"    缺失文件警告:   {warnings}")
    print(f"    异常 mask 数:   {warnings}")
    print(f"    严重错误:       {errors}")
    print("=" * 60)

    if errors > 0:
        print(f"\n[FAIL] 发现 {errors} 个严重错误，请检查数据。")
        sys.exit(1)
    elif warnings > 0:
        print(f"\n[WARN] 发现 {warnings} 个警告，建议检查但不阻塞后续流程。")
        # 警告不阻塞，返回 0
    else:
        print(f"\n[PASS] 所有检查通过！数据集就绪。")

    sys.exit(0)


if __name__ == "__main__":
    main()
