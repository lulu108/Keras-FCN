#!/usr/bin/env python3
"""
VOC 21类 mask → 前景/背景二分类 mask 转换脚本
==============================================

将 Pascal VOC 21 类语义分割 mask 转换成前景/背景二分类 mask。

转换规则:
    - 像素值 0 (背景) → 0
    - 像素值 1~20 (前景物体) → 1
    - 像素值 255 (ignore / boundary) → 255 (保留不变)

输出为单通道 PNG 文件。

Usage:
    python scripts/convert_voc_to_binary.py --mask-root data/processed/masks_21class --out-root data/processed/masks_binary --split train
    python scripts/convert_voc_to_binary.py --mask-root data/processed/masks_21class --out-root data/processed/masks_binary --split val
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 Pascal VOC 21 类 mask 转换为前景/背景二分类 mask"
    )
    parser.add_argument(
        "--mask-root",
        type=str,
        required=True,
        help="21 类 mask 根目录 (如 data/processed/masks_21class)",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="输出根目录 (如 data/processed/masks_binary)",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test", "trainval"],
        help="数据集划分: train / val / test / trainval",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="如果输出文件已存在，是否覆盖 (默认不覆盖)",
    )
    return parser.parse_args()


def convert_21class_to_binary(mask: np.ndarray) -> np.ndarray:
    """
    将 21 类 mask 转换为二分类 mask。

    Args:
        mask: numpy 数组，像素值范围为 [0, 255]

    Returns:
        binary: numpy 数组 (uint8)，像素值只包含 0, 1, 255
    """
    # 先检查是否有异常像素值
    valid_values = set(range(0, 21)) | {255}
    unique_vals = set(np.unique(mask))
    abnormal = unique_vals - valid_values
    if abnormal:
        print(f"  [WARNING] 发现异常像素值: {sorted(abnormal)} (文件将被正常转换，但请检查数据)")

    # 转换: 0→0, 1~20→1, 255→255
    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[(mask >= 1) & (mask <= 20)] = 1
    binary[mask == 255] = 255
    # 背景 (0) 保持 0，已经由 np.zeros_like 默认
    return binary


def main():
    args = parse_args()

    mask_root = Path(args.mask_root)
    out_root = Path(args.out_root)
    split = args.split
    overwrite = args.overwrite

    src_dir = mask_root / split
    dst_dir = out_root / split

    if not src_dir.is_dir():
        print(f"[ERROR] 源目录不存在: {src_dir}")
        sys.exit(1)

    # 收集所有 mask 文件
    mask_files = sorted(src_dir.glob("*.png"))
    if not mask_files:
        print(f"[WARNING] 在 {src_dir} 中未找到任何 .png 文件")
        # 仍然继续，可能是空目录

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 统计
    converted = 0
    skipped = 0
    errors = 0
    all_original_values = set()
    all_binary_values = set()

    for mask_path in mask_files:
        dst_path = dst_dir / mask_path.name

        # 检查是否需要跳过
        if dst_path.is_file() and not overwrite:
            skipped += 1
            # 仍然读取以收集统计信息
            try:
                mask = np.array(Image.open(mask_path))
                all_original_values.update(np.unique(mask))
            except Exception:
                pass
            continue

        try:
            # 读取原始 mask
            mask = np.array(Image.open(mask_path))
            all_original_values.update(np.unique(mask))

            # 转换为二分类
            binary = convert_21class_to_binary(mask)
            all_binary_values.update(np.unique(binary))

            # 保存为单通道 PNG
            img = Image.fromarray(binary, mode="L")
            img.save(dst_path)

            converted += 1
        except Exception as e:
            print(f"[ERROR] 转换失败 {mask_path.name}: {e}")
            errors += 1

    # 如果转换了至少一个文件，还要检查已存在并跳过的文件的 binary values
    if skipped > 0 and overwrite is False:
        # 对跳过的文件也采样检查 binary values (只读模式)
        for mask_path in mask_files:
            dst_path = dst_dir / mask_path.name
            if dst_path.is_file():
                try:
                    binary = np.array(Image.open(dst_path))
                    all_binary_values.update(np.unique(binary))
                except Exception:
                    pass

    # 输出统计
    print("=" * 50)
    print(f"  Split:              {split}")
    print(f"  转换 mask 数量:     {converted}")
    print(f"  已存在并跳过:       {skipped}")
    print(f"  转换失败:           {errors}")
    print(f"  原始像素值集合:     {sorted(all_original_values)}")
    print(f"  二分类像素值集合:   {sorted(all_binary_values)}")
    print(f"  输出路径:           {dst_dir}")
    print("=" * 50)

    if errors > 0:
        print(f"\n[WARNING] {errors} 个文件转换失败。")
    else:
        print(f"\n[INFO] 转换完成，共处理 {len(mask_files)} 个文件。")


if __name__ == "__main__":
    main()
