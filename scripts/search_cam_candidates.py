"""Batch-search CAM candidates across target layers and target scalar modes.

Example:
  PYTHONPATH=ultralytics-git /home/neolux/.conda/envs/yolo/bin/python scripts/search_cam_candidates.py \
    --model runs/detect/0705/b5-3/weights/best.pt \
    --source ../datasets/CST_AntiUAV/CST-AntiUAV/val/building_41/000757.jpg
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CANDIDATES = [
    ("01_p3_mdhifi_eigencam", "eigencam", "score", [19]),
    ("02_p3_mdhifi_gradcam_score", "gradcam", "score", [19]),
    ("03_p3c_gradcam_score", "gradcam", "score", [22]),
    ("04_p3c_gradcam_score_box_area", "gradcam", "score_box_area", [22]),
    ("05_p3out_gradcam_score", "gradcam", "score", [38]),
    ("06_p4out_gradcam_score", "gradcam", "score", [41]),
    ("07_p2p3p4_eigencam", "eigencam", "score", [35, 38, 41]),
    ("08_p2p3p4_gradcam_score_box_area", "gradcam", "score_box_area", [35, 38, 41]),
]


def parse_args():
    p = argparse.ArgumentParser(description="Search for CAM candidates with different target layers/scalars.")
    p.add_argument("--model", required=True, help="Model weights or run dir.")
    p.add_argument("--source", required=True, help="Image path.")
    p.add_argument("--outdir", default="runs/cam_candidate_search", help="Output root.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.34)
    p.add_argument("--base-render", default="gray", choices=("gray", "color"))
    p.add_argument("--colormap", default="turbo", choices=("jet", "turbo"))
    p.add_argument("--crop-scale", type=float, default=3.0)
    p.add_argument("--crop-pad", type=float, default=1.4)
    p.add_argument("--box-source", default="gt", choices=("auto", "gt", "pred"))
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_lines = []

    for name, cam_method, scalar_mode, layer_indices in DEFAULT_CANDIDATES:
        candidate_out = outdir / name
        cmd = [
            sys.executable,
            "scripts/visualize_det_heatmap.py",
            "--model",
            args.model,
            "--source",
            args.source,
            "--outdir",
            str(candidate_out),
            "--imgsz",
            str(args.imgsz),
            "--cam-method",
            cam_method,
            "--target-scalar-mode",
            scalar_mode,
            "--max-images",
            "1",
            "--alpha",
            str(args.alpha),
            "--crop-scale",
            str(args.crop_scale),
            "--crop-pad",
            str(args.crop_pad),
            "--conf",
            str(args.conf),
            "--rank",
            str(args.rank),
            "--base-render",
            args.base_render,
            "--colormap",
            args.colormap,
            "--box-source",
            args.box_source,
            "--layer-indices",
            *[str(i) for i in layer_indices],
        ]
        if args.device:
            cmd += ["--device", args.device]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)
        manifest_lines.append(
            f"{name}\tcam={cam_method}\ttarget={scalar_mode}\tlayers={','.join(map(str, layer_indices))}"
        )

    (outdir / "manifest.tsv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"saved candidate manifest -> {outdir / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
