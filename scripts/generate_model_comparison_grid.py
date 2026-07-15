#!/usr/bin/env python3
"""
Run inference for 14 models on 3 test images, then assemble a comparison grid.

Layout: two stacked blocks (YOLO family top, RT-DETR/B5 bottom).
Each block: 3 rows (images) × N columns (models).
Row labels on the left, model names on top.

Inference results are cached to disk so layout tweaks don't require re-inference.
"""

import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure the modified ultralytics fork is importable
_project_root = Path(__file__).resolve().parent.parent
_ugit = _project_root / "ultralytics-git"
if str(_ugit) not in sys.path:
    sys.path.insert(0, str(_ugit))

from ultralytics import YOLO, RTDETR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = _project_root

YOLO_MODELS: list[dict] = [
    {"name": "YOLOv5s",      "cls": "yolo",  "ckpt": "runs/detect/cst-sample-5k1k1k-s100/yolov5s/weights/best.pt"},
    {"name": "YOLOv5s\n(P2)", "cls": "yolo",  "ckpt": "runs/detect/cst-sample-5k1k1k-s100/yolov5s-p2/weights/best.pt"},
    {"name": "YOLOv8s",      "cls": "yolo",  "ckpt": "runs/detect/cst-sample-5k1k1k-s100/yolov8s/weights/best.pt"},
    {"name": "YOLOv8s\n(P2)", "cls": "yolo",  "ckpt": "runs/detect/cst-sample-5k1k1k-s100/yolov8s-p2/weights/best.pt"},
    {"name": "YOLO11s",      "cls": "yolo",  "ckpt": "runs/detect/cst-sample-5k1k1k-s100/yolo11s/weights/best.pt"},
    {"name": "YOLO11s\n(P2)", "cls": "yolo",  "ckpt": "runs/detect/cst-sample-5k1k1k-s100/yolo11s-p2/weights/best.pt"},
    {"name": "YOLO26s",      "cls": "yolo",  "ckpt": "runs/detect/cst-sample-5k1k1k-s100/yolo26s/weights/best.pt"},
    {"name": "YOLO26s\n(P2)", "cls": "yolo",  "ckpt": "runs/detect/cst-sample-5k1k1k-s100/yolo26s-p2/weights/best.pt"},
]

RTDETR_MODELS: list[dict] = [
    {"name": "RT-DETR\n-L",       "cls": "rtdetr", "ckpt": "runs/detect/cst-sample-5k1k1k-s100/rtdetr-l/weights/best.pt"},
    {"name": "RT-DETR\n-R18",     "cls": "rtdetr", "ckpt": "runs/detect/cst-sample-5k1k1k-s100/rtdetr-r18-init/weights/best.pt"},
    {"name": "RT-DETR\n-R18(P2)", "cls": "rtdetr", "ckpt": "runs/detect/cst-sample-5k1k1k-s100/rtdetr-r18-p2/weights/best.pt"},
    {"name": "RT-DETR\n-R50",     "cls": "rtdetr", "ckpt": "runs/detect/cst-sample-5k1k1k-s100/rtdetr-r50/weights/best.pt"},
    {"name": "RT-DETR\n-R101",    "cls": "rtdetr", "ckpt": "runs/detect/cst-sample-5k1k1k-s100/rtdetr-r101/weights/best.pt"},
    {"name": "0705/B5",           "cls": "rtdetr", "ckpt": "runs/detect/0705/b5-3/weights/best.pt"},
]

IMG_DIR = ROOT / "datasets" / "CST_AntiUAV" / "graphiz"
OUTPUT = ROOT / "assets" / "figures" / "model_comparison_grid.png"
CACHE_DIR = ROOT / "assets" / "figures" / ".inference_cache"

IMGSZ = 640
CONF = 0.25
DEVICE = "0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_images(img_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return paths


def cache_key(entry: dict, img_stem: str) -> str:
    """Unique key for a (model, image) pair."""
    ckpt_stem = Path(entry["ckpt"]).parent.parent.name  # experiment dir name
    return f"{ckpt_stem}__{img_stem}"


def load_from_cache(entry: dict, img_stem: str) -> np.ndarray | None:
    key = cache_key(entry, img_stem)
    cache_file = CACHE_DIR / f"{key}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    return None


def save_to_cache(entry: dict, img_stem: str, arr: np.ndarray):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(entry, img_stem)
    cache_file = CACHE_DIR / f"{key}.npy"
    np.save(cache_file, arr)


def run_inference(model, img_path: Path) -> np.ndarray:
    results = model(str(img_path), imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
    annotated = results[0].plot(conf=True, labels=True, boxes=True)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def build_block(models: list[dict], images: list[Path], block_title: str,
                 thumb_h: int = 450) -> tuple[list, int]:
    """
    Run inference (with cache) for one block of models and return:
      - grid: list[list[np.ndarray]]  [row][col]
      - cell_w: thumbnail width
    """
    n_cols = len(models)
    n_rows = len(images)
    grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    sample_h, sample_w = 640, 640

    for j, entry in enumerate(models):
        name_flat = entry['name'].replace('\n', ' ')
        print(f"  [{j+1}/{n_cols}] {name_flat} ...")

        # Load model (once per entry, only if at least one image needs inference)
        all_cached = all(
            load_from_cache(entry, img.stem) is not None for img in images
        )
        model_obj = None if all_cached else load_model(entry)

        for i, img_path in enumerate(images):
            stem = img_path.stem

            cached = load_from_cache(entry, stem)
            if cached is not None:
                grid[i][j] = cached
                if sample_h == 640:
                    sample_h, sample_w = cached.shape[:2]
                continue

            print(f"    {stem} inferring...")
            try:
                annotated = run_inference(model_obj, img_path)
                grid[i][j] = annotated
                save_to_cache(entry, stem, annotated)
                if sample_h == 640:
                    sample_h, sample_w = annotated.shape[:2]
            except Exception as e:
                print(f"    ERROR: {e}")
                grid[i][j] = None

    if grid[0][0] is not None:
        sample_h, sample_w = grid[0][0].shape[:2]

    cell_w = int(sample_w * thumb_h / sample_h)
    return grid, cell_w


def load_model(entry: dict):
    ckpt = ROOT / entry["ckpt"]
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if entry["cls"] == "yolo":
        return YOLO(str(ckpt))
    else:
        return RTDETR(str(ckpt))


def draw_block(gs_slice, models: list[dict], images: list[Path],
               grid: list[list[np.ndarray]], cell_w: int, thumb_h: int,
               row_offset: int):
    """Draw one block of the comparison grid into the given GridSpec slice."""
    n_cols = len(models)
    n_rows = len(images)

    gap = 4
    label_w = 100

    # Inner GridSpec for this block
    inner_gs = gs_slice.subgridspec(
        nrows=1 + n_rows, ncols=1 + n_cols,
        height_ratios=[0.08] + [1] * n_rows,
        width_ratios=[label_w / (label_w + n_cols * cell_w)] + [1] * n_cols,
        hspace=0.02, wspace=0.02,
    )

    # Column headers
    for j, entry in enumerate(models):
        ax = plt.subplot(inner_gs[0, 1 + j])
        ax.axis("off")
        ax.text(0.5, 0.5, entry["name"], transform=ax.transAxes,
                ha="center", va="center", fontsize=6.5, fontweight="bold",
                linespacing=1.1)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#aaaaaa")
            spine.set_linewidth(0.5)

    # Row labels
    for i, img_path in enumerate(images):
        ax = plt.subplot(inner_gs[1 + i, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, img_path.name, transform=ax.transAxes,
                ha="center", va="center", fontsize=7, fontstyle="italic",
                rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#aaaaaa")
            spine.set_linewidth(0.5)

    # Image cells
    for i in range(n_rows):
        for j in range(n_cols):
            ax = plt.subplot(inner_gs[1 + i, 1 + j])
            ax.axis("off")
            img = grid[i][j]
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "ERROR", transform=ax.transAxes,
                        ha="center", va="center", fontsize=5, color="red")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    images = resolve_images(IMG_DIR)
    print(f"Found {len(images)} test images:")
    for p in images:
        print(f"  {p.name} -> {p.resolve()}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    thumb_h = 450  # pixels per cell

    # Combine all models into a flat list, no category separation
    ALL_MODELS = YOLO_MODELS + RTDETR_MODELS  # 14 models total

    # Run inference (will use cache for all since we just cached them)
    print("\n===== Loading cached inference results =====")
    all_grid, cell_w = build_block(ALL_MODELS, images, "All Models", thumb_h)

    # ---- Split into two rows of 7 ----
    COLS_PER_ROW = 7
    row1_models = ALL_MODELS[:COLS_PER_ROW]
    row2_models = ALL_MODELS[COLS_PER_ROW:]
    row1_grid = [row[:COLS_PER_ROW] for row in all_grid]
    row2_grid = [row[COLS_PER_ROW:] for row in all_grid]

    n_img_rows = len(images)
    label_w = 100
    dpi = 150

    # Figure width: label + 7 cells per row
    fig_w_px = label_w + COLS_PER_ROW * cell_w
    header_ratio = 0.08
    img_ratio = 1.0
    spacer_ratio = 0.05

    fig_h_px = (2 * header_ratio + 2 * n_img_rows * img_ratio + spacer_ratio) * thumb_h

    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)

    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[header_ratio + n_img_rows * img_ratio] * 2,
        hspace=spacer_ratio / (2 * header_ratio + 2 * n_img_rows * img_ratio) * 2,
        top=0.98, bottom=0.02, left=0.03, right=0.99,
    )

    # Row 1: models 1-7
    draw_block(gs[0], row1_models, images, row1_grid, cell_w, thumb_h, 0)

    # Row 2: models 8-14
    draw_block(gs[1], row2_models, images, row2_grid, cell_w, thumb_h, 0)

    # Main title
    fig.suptitle("Model Comparison — CST-AntiUAV Test Images",
                 fontsize=13, fontweight="bold", y=0.998)

    fig.savefig(OUTPUT, dpi=dpi, bbox_inches="tight", facecolor="white",
                edgecolor="none", pad_inches=0.3)
    plt.close(fig)
    print(f"\nDone → {OUTPUT}")


if __name__ == "__main__":
    main()
