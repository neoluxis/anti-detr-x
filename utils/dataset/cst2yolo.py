#!/usr/bin/env python3
# utils/dataset/cst2yolo.py
# neolux_lee <neolux_lee@outlook.com>

import os
import argparse
import re
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CST format to YOLO format.")

    parser.add_argument(
        "--cst_dir",
        type=str,
        default="datasets/CST_AntiUAV/CST-AntiUAV",
        help="CST 数据集的路径，包含 train、val、test 文件夹。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/CST_AntiUAV/yolo",
        help="输出 YOLO 格式数据集的路径。",
    )
    parser.add_argument(
        "--type",
        type=str,
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
        help="要转换的数据集类型，可传多个（默认: train）。",
    )
    sample_size = parser.add_mutually_exclusive_group()
    sample_size.add_argument(
        "--number",
        type=int,
        nargs="+",
        default=[0],
        help="要转换的图像数量，可传多个；单个值会广播到所有 type（默认: 0，表示转换所有图像）。",
    )
    sample_size.add_argument(
        "--ratio",
        type=int,
        nargs="+",
        default=None,
        help="转换图像的比例，范围 1-100；可传多个；单个值会广播到所有 type。",
    )
    parser.add_argument(
        "--selection",
        type=str,
        nargs="+",
        default=["global"],
        help="图像选择方式：'global' 全局随机抽样；'seq-X' 按随机场景连续抽取至多 X 帧。可传多个；单个值会广播到所有 type。",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="随机种子，用于随机选择图像（默认: 42）。"
    )
    parser.add_argument(
        "--id",
        type=int,
        nargs="+",
        default=[0],
        help="类别 ID，可传多个；单个值会广播到所有 type。默认 0。",
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="如果 output_dir 非空，则不交互直接清空后覆盖。",
    )
    overwrite_group.add_argument(
        "--noforce",
        "-nf",
        action="store_true",
        help="如果 output_dir 非空，则不交互直接退出。",
    )

    args = parser.parse_args()

    # 验证输入路径
    if not os.path.exists(args.cst_dir):
        raise FileNotFoundError(f"CST 数据集路径不存在: {args.cst_dir}")

    def normalize_per_type(values, name):
        if values is None:
            return None
        if len(values) == 1:
            return values * len(args.type)
        if len(values) != len(args.type):
            raise ValueError(f"{name} 参数数量必须为 1 或与 type 数量一致。")
        return values

    args.number = normalize_per_type(args.number, "number")
    args.ratio = normalize_per_type(args.ratio, "ratio")
    args.selection = normalize_per_type(args.selection, "selection")
    args.id = normalize_per_type(args.id, "id")

    args.jobs = []
    for split, number, ratio, selection, class_id in zip(
        args.type, args.number, args.ratio or [None] * len(args.type), args.selection, args.id
    ):
        if number < 0:
            raise ValueError("number 参数必须为非负整数。")
        if ratio is not None and not (0 < ratio <= 100):
            raise ValueError("ratio 参数必须在 1 到 100 之间。")

        sequence_length = None
        if selection != "global":
            match = re.fullmatch(r"seq-(\d+)", selection)
            if not match or int(match.group(1)) <= 0:
                raise ValueError("selection 必须为 'global' 或 'seq-X'，其中 X 为正整数。")
            sequence_length = int(match.group(1))

        args.jobs.append(
            {
                "type": split,
                "number": number,
                "ratio": ratio,
                "selection": selection,
                "sequence_length": sequence_length,
                "id": class_id,
            }
        )

    return args


def ensure_output_dir(output_dir, force=False, noforce=False):
    output_path = os.path.abspath(output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        return output_path

    if not os.path.isdir(output_path):
        raise NotADirectoryError(f"output_dir 不是目录: {output_path}")

    if not os.listdir(output_path):
        return output_path

    if noforce:
        print(f"输出目录非空，已退出: {output_path}")
        raise SystemExit(1)

    if not force:
        reply = input(
            f"输出目录非空，是否清空后继续？[y/N]: {output_path} "
        ).strip().lower()
        if reply not in {"y", "yes"}:
            print("已取消，未清空输出目录。")
            raise SystemExit(1)

    for entry in os.listdir(output_path):
        entry_path = os.path.join(output_path, entry)
        if os.path.isdir(entry_path) and not os.path.islink(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.unlink(entry_path)

    return output_path


if __name__ == "__main__":
    import json
    import random

    args = parse_args()
    args.output_dir = ensure_output_dir(
        args.output_dir, force=args.force, noforce=args.noforce
    )
    output_root = Path(args.output_dir).resolve()

    def try_load_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # fallback: try to read as lines of json
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if len(lines) == 1:
                    return json.loads(lines[0])
                return [json.loads(l) for l in lines]
            except Exception:
                return None

    def get_image_size(path):
        """Get image size using PIL/cv2, matching visualize_dataset.py approach."""
        try:
            from PIL import Image
            with Image.open(path) as im:
                return im.width, im.height
        except Exception:
            pass
        
        try:
            import cv2
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError("无法读取图像")
            h, w = img.shape[:2]
            return w, h
        except Exception as e:
            raise RuntimeError(f"无法读取图像尺寸: {path} -> {e}")

    def write_dataset_yaml(class_ids):
        split_entries = {}
        for split in ("train", "val", "test"):
            split_images_dir = output_root / split / "images"
            if split_images_dir.is_dir():
                split_entries[split] = f"{split}/images"

        yaml_path = output_root / "data.yaml"
        lines = [
            "# Auto-generated by util/dataset/cst2yolo.py",
            "# Neolux Lee <neolux_lee@outlook.com>",
            "",
            f"path: {output_root.as_posix()}",
        ]
        for split in ("train", "val", "test"):
            if split in split_entries:
                lines.append(f"{split}: {split_entries[split]}")
        lines.extend(
            [
                "",
                "# Classes",
                "names:",
            ]
        )
        for class_id in sorted(set(class_ids)):
            lines.append(f"  {class_id}: UAV")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return yaml_path

    def collect_candidates(src_type_dir):
        candidates = []
        for scene in sorted(os.listdir(src_type_dir)):
            scene_dir = os.path.join(src_type_dir, scene)
            if not os.path.isdir(scene_dir):
                continue

            imgs = sorted(
                [
                    p
                    for p in os.listdir(scene_dir)
                    if p.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            if not imgs:
                continue

            exist_path = os.path.join(scene_dir, "exist.txt")
            gt_path = os.path.join(scene_dir, "gt.txt")
            exist_data = try_load_json(exist_path) if os.path.exists(exist_path) else None
            gt_data = try_load_json(gt_path) if os.path.exists(gt_path) else None

            if os.path.exists(exist_path) and exist_data is None:
                try:
                    with open(exist_path, "r", encoding="utf-8") as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    exist_data = [int(ln) for ln in lines]
                except Exception:
                    exist_data = None

            if os.path.exists(gt_path) and gt_data is None:
                try:
                    with open(gt_path, "r", encoding="utf-8") as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    parsed = []
                    for ln in lines:
                        if "," in ln:
                            parts = [p.strip() for p in ln.split(",") if p.strip()]
                        else:
                            parts = [p.strip() for p in ln.split() if p.strip()]
                        parsed.append([float(p) for p in parts])
                    gt_data = parsed
                except Exception:
                    gt_data = None

            exist_list = exist_data["exist"] if isinstance(exist_data, dict) and "exist" in exist_data else exist_data
            gt_list = gt_data["gt"] if isinstance(gt_data, dict) and "gt" in gt_data else gt_data

            if isinstance(gt_list, dict):
                ordered = []
                for img_name in imgs:
                    key = img_name
                    idx = os.path.splitext(img_name)[0]
                    ordered.append(gt_list.get(key, gt_list.get(idx, [0, 0, 0, 0])))
                gt_list = ordered

            num_imgs = len(imgs)
            if gt_list and len(gt_list) != num_imgs:
                gt_list = gt_list[:num_imgs] if len(gt_list) > num_imgs else gt_list + [[0, 0, 0, 0]] * (num_imgs - len(gt_list))
            if exist_list and len(exist_list) != num_imgs:
                exist_list = exist_list[:num_imgs] if len(exist_list) > num_imgs else exist_list + [0] * (num_imgs - len(exist_list))

            for i, img_name in enumerate(imgs):
                img_path = os.path.join(scene_dir, img_name)
                exists = None
                bbox = None
                if exist_list is not None:
                    try:
                        exists = int(exist_list[i])
                    except Exception:
                        exists = 1 if exist_list[i] else 0
                if gt_list is not None:
                    try:
                        bbox = gt_list[i]
                    except Exception:
                        bbox = None

                if isinstance(bbox, dict):
                    bbox = [bbox.get(k, 0) for k in ("x", "y", "w", "h")]

                valid_bbox = None
                if exists != 0 and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    try:
                        parsed_bbox = [float(value) for value in bbox[:4]]
                        if parsed_bbox[2] > 0 and parsed_bbox[3] > 0:
                            valid_bbox = parsed_bbox
                    except (TypeError, ValueError):
                        pass

                candidates.append((scene, i, img_path, valid_bbox))
        return candidates

    def select_candidates(candidates, job, rng):
        total = len(candidates)
        if job["number"] > 0:
            k = min(job["number"], total)
        elif job["ratio"] is not None:
            k = max(1, int(total * job["ratio"] / 100.0))
        else:
            k = total

        if job["selection"] == "global":
            return rng.sample(candidates, k)

        candidates_by_scene = {}
        for candidate in candidates:
            candidates_by_scene.setdefault(candidate[0], []).append(candidate)

        selected = []
        available = {
            scene: {candidate[1]: candidate for candidate in scene_candidates}
            for scene, scene_candidates in candidates_by_scene.items()
        }
        previous_scene = None
        while len(selected) < k:
            runs_by_scene = {}
            for scene, frames in available.items():
                frame_indices = sorted(frames)
                if not frame_indices:
                    continue
                runs = []
                run = [frame_indices[0]]
                for frame_index in frame_indices[1:]:
                    if frame_index == run[-1] + 1:
                        run.append(frame_index)
                    else:
                        runs.append(run)
                        run = [frame_index]
                runs.append(run)
                runs_by_scene[scene] = runs

            if not runs_by_scene:
                break

            max_chunk = min(job["sequence_length"], k - len(selected))
            scenes = [
                scene
                for scene, runs in runs_by_scene.items()
                if any(len(run) >= max_chunk for run in runs)
            ]
            scenes = scenes or list(runs_by_scene)
            if len(scenes) > 1 and previous_scene in scenes:
                scenes.remove(previous_scene)
            scene = rng.choice(scenes)
            previous_scene = scene

            usable_runs = [
                run for run in runs_by_scene[scene] if len(run) >= max_chunk
            ]
            run = rng.choice(usable_runs or runs_by_scene[scene])
            chunk_size = min(max_chunk, len(run))
            start = rng.randint(0, len(run) - chunk_size)
            chunk_indices = run[start : start + chunk_size]
            selected.extend(available[scene].pop(frame_index) for frame_index in chunk_indices)

        selected.sort(key=lambda candidate: (candidate[0], candidate[1]))
        return selected

    def write_selected(job, selected):
        import sys
        import time

        out_images_dir = os.path.join(args.output_dir, job["type"], "images")
        out_labels_dir = os.path.join(args.output_dir, job["type"], "labels")
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_labels_dir, exist_ok=True)

        scene_size_cache = {}
        total_selected = len(selected)
        start_time = time.time()
        for idx, (scene, _, img_path, bbox) in enumerate(selected, start=1):
            label_line = ""
            if bbox is not None:
                try:
                    scene_key = os.path.dirname(img_path)
                    if scene_key not in scene_size_cache:
                        scene_imgs = sorted(
                            [
                                p
                                for p in os.listdir(scene_key)
                                if p.lower().endswith((".jpg", ".jpeg", ".png"))
                            ]
                        )
                        if not scene_imgs:
                            raise RuntimeError(f"场景中没有图片: {scene_key}")
                        scene_size_cache[scene_key] = get_image_size(
                            os.path.join(scene_key, scene_imgs[0])
                        )
                    w, h = scene_size_cache[scene_key]
                except Exception as e:
                    print(f"跳过图像 (无法读取尺寸): {img_path} -> {e}")
                    continue

                x, y, bw, bh = bbox
                x_center_n = max(0.0, min(1.0, (x + bw / 2.0) / w))
                y_center_n = max(0.0, min(1.0, (y + bh / 2.0) / h))
                bw_n = max(0.0, min(1.0, bw / w))
                bh_n = max(0.0, min(1.0, bh / h))
                label_line = (
                    f"{job['id']} {x_center_n:.6f} {y_center_n:.6f} "
                    f"{bw_n:.6f} {bh_n:.6f}\n"
                )

            img_basename = os.path.basename(img_path)
            if job["selection"] == "global":
                output_name = f"{scene}_{img_basename}"
                out_img = os.path.join(out_images_dir, output_name)
                out_lbl = os.path.join(
                    out_labels_dir, os.path.splitext(output_name)[0] + ".txt"
                )
            else:
                out_img = os.path.join(out_images_dir, scene, img_basename)
                out_lbl = os.path.join(
                    out_labels_dir, scene, os.path.splitext(img_basename)[0] + ".txt"
                )
                os.makedirs(os.path.dirname(out_img), exist_ok=True)
                os.makedirs(os.path.dirname(out_lbl), exist_ok=True)

            try:
                if not os.path.lexists(out_img):
                    os.symlink(os.path.abspath(img_path), out_img)
            except Exception:
                try:
                    if not os.path.lexists(out_img):
                        shutil.copy2(img_path, out_img)
                except Exception:
                    print(f"无法链接或复制图像: {img_path}")
                    continue

            with open(out_lbl, "w", encoding="utf-8") as f:
                f.write(label_line)

            if idx % 50 == 0 or idx == total_selected:
                elapsed = time.time() - start_time
                avg = elapsed / idx
                remaining = avg * (total_selected - idx)
                pct = idx / total_selected * 100
                sys.stdout.write(
                    f"\r[{job['type']}] 已处理 {idx}/{total_selected} ({pct:.1f}%)，耗时 {elapsed:.1f}s，预计剩余 {remaining:.1f}s"
                )
                sys.stdout.flush()
        print()

    used_class_ids = []
    for offset, job in enumerate(args.jobs):
        rng = random.Random(args.seed + offset)
        src_type_dir = os.path.join(args.cst_dir, job["type"])
        if not os.path.isdir(src_type_dir):
            raise FileNotFoundError(f"CST 子目录不存在: {src_type_dir}")

        print(f"[{job['type']}] 开始收集图像帧...")
        candidates = collect_candidates(src_type_dir)
        if not candidates:
            print(f"[{job['type']}] 没有找到任何图像帧。请检查 CST 数据集结构。")
            raise SystemExit(1)

        print(f"[{job['type']}] 找到 {len(candidates)} 个图像帧。")
        selected = select_candidates(candidates, job, rng)
        write_selected(job, selected)
        print(
            f"[{job['type']}] 完成：已生成 {len(selected)} 个样本，输出目录: {os.path.join(args.output_dir, job['type'])}"
        )
        used_class_ids.append(job["id"])

    yaml_path = write_dataset_yaml(used_class_ids)
    print(f"数据集 YAML: {yaml_path}")
