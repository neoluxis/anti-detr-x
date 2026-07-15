"""Compare feature response heatmaps across multiple model variants.

This script calls `visualize_det_heatmap.py` for each model variant and
stitches the generated panels side-by-side for easy visual comparison.

Example:
  python scripts/compare_feature_responses.py --models runs/detect/0705/b5-3 runs/detect/0705/b5-2 \
    --source data/images --outdir runs/heatmap_comparisons --cam-method eigencam
"""
import argparse
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="Model weights or directories (space-separated)")
    p.add_argument("--source", required=True, help="Image file or directory")
    p.add_argument("--outdir", default="runs/heatmap_comparisons", help="Output directory")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--cam-method", default="eigencam", choices=("eigencam", "gradcam"))
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--crop-scale", type=float, default=3.0)
    p.add_argument("--crop-pad", type=float, default=1.4)
    return p.parse_args()


def collect_images(source: Path, max_images: int):
    IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")
    images = [p for p in sorted(source.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    return images[:max_images]


def model_outdir(base_out: Path, model_path: str):
    name = Path(model_path).name
    return base_out / name


def run_visualize_for_model(model: str, source: str, outdir: Path, args):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/visualize_det_heatmap.py", "--model", model, "--source", source, "--outdir", str(outdir), "--imgsz", str(args.imgsz), "--cam-method", args.cam_method, "--max-images", str(args.max_images), "--alpha", str(args.alpha), "--crop-scale", str(args.crop_scale), "--crop-pad", str(args.crop_pad), "--conf", str(args.conf), "--rank", str(args.rank)]
    if args.device:
        cmd += ["--device", args.device]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def stitch_panels(models: list[str], model_outs: list[Path], images: list[Path], outdir: Path):
    cmp_dir = outdir / "comparisons"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        stem = f"{img.parent.name}__{img.stem}"
        panels = []
        names = []
        for model, m_out in zip(models, model_outs):
            panel_path = m_out / f"{stem}_heatmap_panel.jpg"
            if not panel_path.exists():
                print(f"warning: missing panel for {model} -> {panel_path}")
                break
            im = cv2.imread(str(panel_path))
            if im is None:
                print(f"warning: unreadable panel {panel_path}")
                break
            panels.append(im)
            names.append(Path(model).name)

        if not panels or len(panels) != len(models):
            continue

        # Normalize heights and stitch horizontally
        heights = [p.shape[0] for p in panels]
        target_h = max(heights)
        resized = [cv2.resize(p, (int(p.shape[1] * target_h / p.shape[0]), target_h), interpolation=cv2.INTER_CUBIC) for p in panels]
        spacer = 10
        total_w = sum(p.shape[1] for p in resized) + spacer * (len(resized) - 1)
        canvas = np.full((target_h + 40, total_w, 3), 255, dtype=np.uint8)
        x = 0
        for name, p in zip(names, resized):
            canvas[0: p.shape[0], x : x + p.shape[1]] = p
            # put label below each panel
            cv2.putText(canvas, name, (x + 8, target_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            x += p.shape[1] + spacer

        out_path = cmp_dir / f"{stem}_comparison.jpg"
        cv2.imwrite(str(out_path), canvas)
        print(f"wrote: {out_path}")


def main():
    args = parse_args()
    models = args.models
    src = Path(args.source)
    out = Path(args.outdir)
    images = collect_images(src, args.max_images)
    if not images:
        print("No images found.")
        return

    model_outs = []
    for m in models:
        mo = model_outdir(out, m)
        model_outs.append(mo)
        run_visualize_for_model(m, str(src), mo, args)

    stitch_panels(models, model_outs, images, out)


if __name__ == "__main__":
    main()
