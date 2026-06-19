#!/usr/bin/env python3
"""
Pascal VOC 数据准备脚本
=======================

从 Pascal VOC 原始目录中根据 train.txt / val.txt 等 split 文件，
将图像和 21 类 mask 整理到 processed/ 目录下。

输出结构:
    data/processed/
    ├── images/{split}/{id}.jpg
    └── masks_21class/{split}/{id}.png

Usage:
    python scripts/prepare_voc.py --voc-root data/VOCdevkit/VOC2012 --out-root data/processed --split train
    python scripts/prepare_voc.py --voc-root data/VOCdevkit/VOC2012 --out-root data/processed --split val
    python scripts/prepare_voc.py --voc-root data/VOCdevkit/VOC2012 --out-root data/processed --split trainval
"""

import argparse
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 Pascal VOC 图像和 mask 按 split 整理到 processed/ 目录"
    )
    parser.add_argument(
        "--voc-root",
        type=str,
        required=True,
        help="Pascal VOC2012 根目录 (包含 JPEGImages/, SegmentationClass/, ImageSets/)",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/processed",
        help="输出根目录，默认为 data/processed",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test", "trainval"],
        help="数据集划分: train / val / test / trainval",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        default="copy",
        choices=["copy", "symlink"],
        help="文件操作方式: copy (复制) 或 symlink (软链接)，默认 copy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="如果目标文件已存在，是否覆盖 (默认不覆盖)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    voc_root = Path(args.voc_root)
    out_root = Path(args.out_root)
    split = args.split
    copy_mode = args.copy_mode
    overwrite = args.overwrite

    # 验证 VOC 根目录
    jpeg_dir = voc_root / "JPEGImages"
    seg_dir = voc_root / "SegmentationClass"
    split_file = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"

    if not voc_root.is_dir():
        print(f"[ERROR] VOC 根目录不存在: {voc_root}")
        sys.exit(1)
    if not jpeg_dir.is_dir():
        print(f"[ERROR] JPEGImages 目录不存在: {jpeg_dir}")
        sys.exit(1)
    if not seg_dir.is_dir():
        print(f"[ERROR] SegmentationClass 目录不存在: {seg_dir}")
        sys.exit(1)
    if not split_file.is_file():
        print(f"[ERROR] Split 文件不存在: {split_file}")
        sys.exit(1)

    # 读取 split 文件中的 image ids
    with open(split_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Split '{split}' 包含 {len(ids)} 个样本 ID")

    # 创建输出目录
    out_img_dir = out_root / "images" / split
    out_mask_dir = out_root / "masks_21class" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # 选择文件操作函数
    if copy_mode == "symlink":
        op_func = _symlink_file
        op_name = "Symlink"
    else:
        op_func = shutil.copy2
        op_name = "Copy"

    # 统计
    stats = {
        "total": len(ids),
        "copied_image": 0,
        "copied_mask": 0,
        "skipped_image": 0,
        "skipped_mask": 0,
        "missing_image": 0,
        "missing_mask": 0,
    }

    for img_id in ids:
        src_img = jpeg_dir / f"{img_id}.jpg"
        src_mask = seg_dir / f"{img_id}.png"
        dst_img = out_img_dir / f"{img_id}.jpg"
        dst_mask = out_mask_dir / f"{img_id}.png"

        # ---- 处理图像 ----
        if not src_img.is_file():
            print(f"[WARNING] 缺失图像: {src_img}")
            stats["missing_image"] += 1
        else:
            if dst_img.is_file() and not overwrite:
                stats["skipped_image"] += 1
            else:
                try:
                    op_func(str(src_img), str(dst_img))
                    stats["copied_image"] += 1
                except OSError as e:
                    print(f"[WARNING] 复制图像失败 {src_img} -> {dst_img}: {e}")

        # ---- 处理 mask ----
        if not src_mask.is_file():
            print(f"[WARNING] 缺失 mask: {src_mask}")
            stats["missing_mask"] += 1
        else:
            if dst_mask.is_file() and not overwrite:
                stats["skipped_mask"] += 1
            else:
                try:
                    op_func(str(src_mask), str(dst_mask))
                    stats["copied_mask"] += 1
                except OSError as e:
                    print(f"[WARNING] 复制 mask 失败 {src_mask} -> {dst_mask}: {e}")

    # 输出统计
    print("=" * 50)
    print(f"  Split:            {split}")
    print(f"  总样本数:         {stats['total']}")
    print(f"  成功{op_name}图像:  {stats['copied_image']}")
    print(f"  成功{op_name}mask:  {stats['copied_mask']}")
    print(f"  已存在跳过(图像):  {stats['skipped_image']}")
    print(f"  已存在跳过(mask):  {stats['skipped_mask']}")
    print(f"  缺失图像:          {stats['missing_image']}")
    print(f"  缺失 mask:         {stats['missing_mask']}")
    print(f"  图像输出路径:      {out_img_dir}")
    print(f"  mask 输出路径:     {out_mask_dir}")
    print("=" * 50)

    # 如果有缺失，给出总结警告
    total_missing = stats["missing_image"] + stats["missing_mask"]
    if total_missing > 0:
        print(f"\n[WARNING] 共 {total_missing} 个文件缺失，请检查 VOC 原始数据完整性。")
    else:
        print(f"\n[INFO] 所有文件处理完毕，无缺失。")


def _symlink_file(src: str, dst: str):
    """创建软链接。如果目标已存在，先删除 (overwrite 模式由调用方控制)。"""
    Path(dst).symlink_to(Path(src).resolve())


if __name__ == "__main__":
    main()
