import os
import argparse
import json
import random
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CST format to Frhybrid YOLO format (Scene-level sampling).")

    parser.add_argument(
        "--cst_dir",
        type=str,
        default="datasets/CST_AntiUAV/CST-AntiUAV",
        help="CST 数据集的路径，包含 train、val、test 文件夹。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/CST_AntiUAV/frhybrid",
        help="输出 Frhybrid YOLO 格式数据集的路径。",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="要转换的数据集类型（默认: train）。",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=5,
        help="要转换的场景数量（默认: 0，表示转换所有场景）。",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=100,
        help="要转换的场景比例（默认: 100，表示转换所有场景）。",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="随机种子（仅在 number/ratio < 100 时生效）。"
    )
    parser.add_argument(
        "--id", 
        type=int, 
        default=0, 
        help="类别 ID，默认为 0。"
    )

    args = parser.parse_args()

    if not os.path.exists(args.cst_dir):
        raise FileNotFoundError(f"CST 数据集路径不存在: {args.cst_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if not (0 < args.ratio <= 100):
        raise ValueError("ratio 参数必须在 1 到 100 之间。")

    return args


def try_load_json_or_txt(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("[") or content.startswith("{"):
            return json.loads(content)
        else:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if "gt.txt" in path.lower():
                return [[float(x) for x in line.replace(",", " ").split()] for line in lines]
            else:
                return [int(x) for x in lines]
    except Exception:
        return None


def get_image_size(img_path):
    try:
        with Image.open(img_path) as im:
            return im.width, im.height
    except Exception:
        import cv2
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
        raise RuntimeError(f"无法读取图像: {img_path}")


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    src_type_dir = os.path.join(args.cst_dir, args.type)
    if not os.path.isdir(src_type_dir):
        raise FileNotFoundError(f"目录不存在: {src_type_dir}")

    out_images_root = os.path.join(args.output_dir, args.type, "images")
    out_labels_root = os.path.join(args.output_dir, args.type, "labels")
    os.makedirs(out_images_root, exist_ok=True)
    os.makedirs(out_labels_root, exist_ok=True)

    # 获取所有场景
    all_scenes = sorted([s for s in os.listdir(src_type_dir) if os.path.isdir(os.path.join(src_type_dir, s))])
    print(f"共发现 {len(all_scenes)} 个场景。")

    # 根据 number / ratio 选择场景
    if args.number > 0:
        num_scenes = min(args.number, len(all_scenes))
        selected_scenes = random.sample(all_scenes, num_scenes)
    elif args.ratio < 100:
        num_scenes = max(1, int(len(all_scenes) * args.ratio / 100))
        selected_scenes = random.sample(all_scenes, num_scenes)
    else:
        selected_scenes = all_scenes

    print(f"将转换 {len(selected_scenes)} 个场景（全量图片导出）。\n")

    outer_pbar = tqdm(selected_scenes, desc="Scenes", unit="scene")
    total_images = 0

    for scene in outer_pbar:
        scene_dir = os.path.join(src_type_dir, scene)
        imgs = sorted([p for p in os.listdir(scene_dir) if p.lower().endswith((".jpg", ".jpeg", ".png"))])
        if not imgs:
            continue

        exist_data = try_load_json_or_txt(os.path.join(scene_dir, "exist.txt"))
        gt_data = try_load_json_or_txt(os.path.join(scene_dir, "gt.txt"))

        # 创建输出目录
        scene_img_dir = os.path.join(out_images_root, scene)
        scene_lbl_dir = os.path.join(out_labels_root, scene)
        os.makedirs(scene_img_dir, exist_ok=True)
        os.makedirs(scene_lbl_dir, exist_ok=True)

        inner_pbar = tqdm(imgs, desc=f"  └─ {scene}", unit="img", leave=False)
        scene_size = None

        for img_name in inner_pbar:
            img_path = os.path.join(scene_dir, img_name)
            idx = imgs.index(img_name)

            # 检查是否存在
            if exist_data is not None:
                try:
                    exists = int(exist_data[idx]) if isinstance(exist_data, list) else 1
                    if exists == 0:
                        continue
                except:
                    pass

            # 获取 bbox
            bbox = None
            if gt_data and idx < len(gt_data):
                bbox = gt_data[idx]
                if isinstance(bbox, dict):
                    bbox = [bbox.get(k, 0) for k in ("x", "y", "w", "h")]

            if not bbox or not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                continue

            try:
                if scene_size is None:
                    scene_size = get_image_size(img_path)
                w, h = scene_size

                # 转为 YOLO 格式
                x, y, bw, bh = [float(v) for v in bbox[:4]]
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw_n = bw / w
                bh_n = bh / h

                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                bw_n = max(0.0, min(1.0, bw_n))
                bh_n = max(0.0, min(1.0, bh_n))

                # 输出路径
                out_img = os.path.join(scene_img_dir, img_name)
                out_lbl = os.path.join(scene_lbl_dir, os.path.splitext(img_name)[0] + ".txt")

                # 仅使用软链接
                if not os.path.exists(out_img):
                    os.symlink(os.path.abspath(img_path), out_img)

                # 写入标签
                with open(out_lbl, "w", encoding="utf-8") as f:
                    f.write(f"{args.id} {x_center:.6f} {y_center:.6f} {bw_n:.6f} {bh_n:.6f}\n")

                total_images += 1

            except Exception as e:
                inner_pbar.write(f"[Warn] 失败 {img_name}: {e}")

    print("\n\n[Done] 转换完成！")
    print(f"输出路径 : {os.path.join(args.output_dir, args.type)}")
    print(f"共转换 {len(selected_scenes)} 个场景，{total_images} 张图像（全部使用软链接）。")