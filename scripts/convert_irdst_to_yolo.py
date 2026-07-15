#!/usr/bin/env python3
"""
将 datasets/IRDST_real 的 box 标签转换为 YOLO 格式。

源格式: [x, y, w, h] 像素坐标（逗号或空格分隔），可能多行多目标，无类别信息。
目标格式: class_id cx cy w h（归一化到 [0,1]），class_id 统一为 0。

图像通过软链接节省空间。

用法:
  # 直接转换（保留原始 train/test 划分）
  python scripts/convert_irdst_to_yolo.py

  # 抽样 20k 张，12000 训练 + 8000 验证（CST 配置）
  python scripts/convert_irdst_to_yolo.py --sample 20000 --num-train 12000 --num-val 8000

  # 只预览不执行
  python scripts/convert_irdst_to_yolo.py --sample 20000 --dry-run
"""

import argparse
import ast
import os
import random
import re
import sys
from pathlib import Path


def parse_boxes(content: str) -> list[list[int]]:
    """解析 box 标签文件，支持逗号和空格分隔，以及多行多目标。"""
    content = content.strip()
    if not content:
        return []

    boxes = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = line.strip("[]")
        parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
        if len(parts) == 4:
            boxes.append([int(float(p)) for p in parts])

    return boxes


def box_to_yolo(box: list[int], img_w: int, img_h: int) -> str:
    """将像素坐标 box [x, y, w, h] 转为 YOLO 格式字符串。"""
    x, y, w, h = box
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def collect_files(
    src_root: Path, splits: list[str], img_ext: str
) -> list[tuple[str, Path, Path]]:
    """
    收集所有可用的 (stem, img_path, box_path) 三元组。
    跨所有 split 汇总，按 stem 去重（同名 stem 只保留第一次出现）。
    """
    seen: set[str] = set()
    files = []
    for split in splits:
        img_dir = src_root / "images" / split
        box_dir = src_root / "boxes" / split
        if not img_dir.exists() or not box_dir.exists():
            print(f"[SKIP] {split}: 缺少 images/ 或 boxes/ 目录")
            continue
        for box_file in sorted(box_dir.iterdir()):
            if not box_file.suffix == ".txt":
                continue
            stem = box_file.stem
            if stem in seen:
                continue
            img_file = img_dir / f"{stem}{img_ext}"
            if img_file.exists():
                seen.add(stem)
                files.append((stem, img_file, box_file))
    return files


def process_file(
    stem: str,
    img_src: Path,
    box_src: Path,
    dst_img_dir: Path,
    dst_label_dir: Path,
    img_w: int,
    img_h: int,
    dry_run: bool = False,
) -> int:
    """
    处理单个文件：写入 YOLO 标签 + 创建图像软链接。
    返回目标数量。
    """
    with open(box_src, "r") as f:
        content = f.read()
    boxes = parse_boxes(content)

    label_path = dst_label_dir / f"{stem}.txt"
    yolo_lines = [box_to_yolo(b, img_w, img_h) for b in boxes]

    if not dry_run:
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        dst_img = dst_img_dir / f"{stem}{img_src.suffix}"
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        dst_img.symlink_to(os.path.relpath(img_src, dst_img_dir))

    return len(boxes)


def convert_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    img_w: int,
    img_h: int,
    img_ext: str = ".png",
    dry_run: bool = False,
) -> tuple[int, int]:
    """转换一个 split（train/test）。返回 (文件数, 目标数)。"""
    src_box_dir = src_root / "boxes" / split
    src_img_dir = src_root / "images" / split
    dst_label_dir = dst_root / "labels" / split
    dst_img_dir = dst_root / "images" / split

    if not src_box_dir.exists():
        print(f"[SKIP] 源目录不存在: {src_box_dir}")
        return 0, 0

    if not dry_run:
        dst_label_dir.mkdir(parents=True, exist_ok=True)
        dst_img_dir.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_objs = 0

    for box_file in sorted(src_box_dir.iterdir()):
        if not box_file.suffix == ".txt":
            continue
        stem = box_file.stem
        img_file = src_img_dir / f"{stem}{img_ext}"
        if not img_file.exists():
            continue

        n_objs += process_file(
            stem, img_file, box_file, dst_img_dir, dst_label_dir,
            img_w, img_h, dry_run,
        )
        n_files += 1

    return n_files, n_objs


def main():
    parser = argparse.ArgumentParser(
        description="将 IRDST_real box 标签转为 YOLO 格式，图像使用软链接"
    )
    parser.add_argument(
        "--src", default="datasets/IRDST_real",
        help="源数据集目录 (默认: datasets/IRDST_real)",
    )
    parser.add_argument(
        "--dst", default="datasets/IRDST_real_yolo",
        help="目标 YOLO 格式目录 (默认: datasets/IRDST_real_yolo)",
    )
    parser.add_argument("--img-width", type=int, default=720, help="图像宽度 (默认: 720)")
    parser.add_argument("--img-height", type=int, default=480, help="图像高度 (默认: 480)")
    parser.add_argument("--img-ext", default=".png", help="图像文件扩展名 (默认: .png)")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"],
        help="源 splits，用于直接转换模式 (默认: train test)",
    )

    # 抽样模式
    parser.add_argument(
        "--sample", type=int, default=None,
        help="抽样模式: 从所有源文件中随机抽取 N 张图像 (e.g. 20000)",
    )
    parser.add_argument(
        "--num-train", type=int, default=12000,
        help="抽样模式下训练集数量 (默认: 12000)",
    )
    parser.add_argument(
        "--num-val", type=int, default=8000,
        help="抽样模式下验证集数量 (默认: 8000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认: 42)",
    )

    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际执行")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"[ERROR] 源目录不存在: {src_root}")
        sys.exit(1)

    print(f"源目录: {src_root.absolute()}")
    print(f"目标目录: {dst_root.absolute()}")
    print(f"图像尺寸: {args.img_width}x{args.img_height}")
    if args.dry_run:
        print("[DRY RUN] 仅预览，不实际执行")

    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    if args.sample:
        # ===== 抽样模式 =====
        random.seed(args.seed)

        # 从所有 splits 收集文件
        all_files = collect_files(src_root, args.splits, args.img_ext)
        print(f"\n可用文件总数: {len(all_files)} (来自 {args.splits})")

        # 随机抽样
        n_needed = args.num_train + args.num_val
        if args.sample > 0:
            n_needed = min(args.sample, len(all_files))
        else:
            n_needed = min(n_needed, len(all_files))

        if n_needed > len(all_files):
            print(f"[WARNING] 需要 {n_needed} 张但只有 {len(all_files)} 张可用，将使用全部")
            n_needed = len(all_files)

        sampled = random.sample(all_files, n_needed)
        print(f"抽样: {n_needed} 张")

        # 划分 train/val
        random.shuffle(sampled)
        n_train = min(args.num_train, n_needed)
        n_val = n_needed - n_train

        files_train = sampled[:n_train]
        files_val = sampled[n_train:]

        print(f"划分:  训练集 {len(files_train)} 张, 验证集 {len(files_val)} 张\n")

        total_files = 0
        total_objs = 0

        for split_name, split_files in [("train", files_train), ("val", files_val)]:
            dst_img_dir = dst_root / "images" / split_name
            dst_label_dir = dst_root / "labels" / split_name
            if not args.dry_run:
                dst_img_dir.mkdir(parents=True, exist_ok=True)
                dst_label_dir.mkdir(parents=True, exist_ok=True)

            n_files = 0
            n_objs = 0
            for stem, img_path, box_path in split_files:
                n_objs += process_file(
                    stem, img_path, box_path, dst_img_dir, dst_label_dir,
                    args.img_width, args.img_height, args.dry_run,
                )
                n_files += 1

            total_files += n_files
            total_objs += n_objs
            print(f"  [{split_name}] {n_files} 个文件, {n_objs} 个目标")

    else:
        # ===== 直接转换模式 =====
        print(f"Splits: {args.splits}\n")

        total_files = 0
        total_objs = 0

        for split in args.splits:
            n_files, n_objs = convert_split(
                src_root=src_root, dst_root=dst_root, split=split,
                img_w=args.img_width, img_h=args.img_height,
                img_ext=args.img_ext, dry_run=args.dry_run,
            )
            total_files += n_files
            total_objs += n_objs
            print(f"  [{split}] {n_files} 个文件, {n_objs} 个目标")

    print(f"\n总计: {total_files} 个文件, {total_objs} 个目标")
    if not args.dry_run:
        print(f"YOLO 数据集已生成: {dst_root.absolute()}")
        print(f"  结构: images/{{split}}/  (软链接) + labels/{{split}}/  (YOLO 标签)")


if __name__ == "__main__":
    main()
