import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    "1": {
        "name": "freeze-last3-heatmap",
        "model": "exp_cfg/detr/rtdetr-resnet50-last3-pretrained.yaml",
        "freeze": 4,
    },
    "2": {
        "name": "freeze-last3-ema-heatmap",
        "model": "exp_cfg/detr/rtdetr-resnet50-last3-pretrained-ema.yaml",
        "freeze": 7,
    },
    "3": {
        "name": "finetune-last3-heatmap",
        "model": "exp_cfg/detr/rtdetr-resnet50-last3-pretrained.yaml",
        "freeze": 0,
    },
    "4": {
        "name": "finetune-last3-ema-heatmap",
        "model": "exp_cfg/detr/rtdetr-resnet50-last3-pretrained-ema.yaml",
        "freeze": 0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the four requested ResNet50 RT-DETR experiments.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset YAML.")
    parser.add_argument("--project", type=str, default="runs/detr_resnet50", help="Ultralytics project directory.")
    parser.add_argument("--device", type=str, default=None, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--variants", nargs="*", default=["1", "2", "3", "4"], choices=sorted(EXPERIMENTS), help="Subset of experiment ids to run.")
    parser.add_argument("--heatmap-source", type=str, default=None, help="Optional image file or directory for post-train heatmaps.")
    parser.add_argument("--heatmap-max-images", type=int, default=50, help="Maximum images for each heatmap run.")
    parser.add_argument("--heatmap-rank", type=int, default=0, help="Detection rank used by the heatmap script.")
    parser.add_argument("--exist-ok", action="store_true", default=False, help="Allow reuse of existing run directories.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print commands without executing them.")
    return parser.parse_args()


def build_train_command(args, exp_id: str, cfg: dict) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train_detr.py",
        "--model",
        cfg["model"],
        "--dataset_path",
        args.dataset_path,
        "--project",
        args.project,
        "--name",
        f"exp{exp_id}-{cfg['name']}",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--imgsz",
        str(args.imgsz),
        "--workers",
        str(args.workers),
        "--freeze",
        str(cfg["freeze"]),
    ]
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.exist_ok:
        cmd.append("--exist-ok")
    return cmd


def build_heatmap_command(args, exp_id: str, cfg: dict, best_pt: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/visualize_det_heatmap.py",
        "--model",
        str(best_pt),
        "--source",
        args.heatmap_source,
        "--arch",
        "detr",
        "--imgsz",
        str(args.imgsz),
        "--rank",
        str(args.heatmap_rank),
        "--max-images",
        str(args.heatmap_max_images),
        "--outdir",
        str(Path(args.project) / f"exp{exp_id}-{cfg['name']}" / "heatmap"),
    ]


def main():
    args = parse_args()
    for exp_id in args.variants:
        cfg = EXPERIMENTS[exp_id]
        train_cmd = build_train_command(args, exp_id, cfg)
        print(" ".join(train_cmd))
        if args.dry_run:
            continue
        subprocess.run(train_cmd, check=True)

        if not args.heatmap_source:
            continue
        best_pt = Path(args.project) / f"exp{exp_id}-{cfg['name']}" / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"Best checkpoint not found for experiment {exp_id}: {best_pt}")
        heatmap_cmd = build_heatmap_command(args, exp_id, cfg, best_pt)
        print(" ".join(heatmap_cmd))
        subprocess.run(heatmap_cmd, check=True)


if __name__ == "__main__":
    main()
