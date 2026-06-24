import argparse
import subprocess
import sys


EXPERIMENTS = {
    "tiny": {
        "plain": {
            "name": "rtdetr-swinv2-tiny-unfreeze-last3-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained.yaml",
            "freeze": 0,
        },
        "ema": {
            "name": "rtdetr-swinv2-tiny-unfreeze-last3-ema-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema.yaml",
            "freeze": 0,
        },
        "b1-aifi-mdhifi": {
            "name": "rtdetr-swinv2-tiny-unfreeze-last3-ema-b1-aifi-mdhifi-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b1-aifi-mdhifi.yaml",
            "freeze": 0,
        },
        "b2-p3-mdhifi": {
            "name": "rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi.yaml",
            "freeze": 0,
        },
        "b2-p3-mdhifi-direct-repc3-p3-aux-p4-p4mdhifi": {
            "name": "rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-direct-repc3-p3-aux-p4-p4mdhifi-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi-direct-repc3-p3-aux-p4-p4mdhifi.yaml",
            "freeze": 0,
        },
        "b3-p3-hifi": {
            "name": "rtdetr-swinv2-tiny-unfreeze-last3-ema-b3-p3-hifi-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b3-p3-hifi.yaml",
            "freeze": 0,
        },
        "b3-p3-aifi": {
            "name": "rtdetr-swinv2-tiny-unfreeze-last3-ema-b3-p3-aifi-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b3-p3-aifi.yaml",
            "freeze": 0,
        },
        "ema-wtconv-biagcau-mdhifi": {
            "name": "rtdetr-swinv2-tiny-unfreeze-last3-ema-wtconv-biagcau-mdhifi-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-wtconv-biagcau-mdhifi.yaml",
            "freeze": 0,
        },
        "0619-b1-biagcau-p3": {
            "name": "rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-0619-b1-biagcau-p3-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi-0619-b1-biagcau-p3.yaml",
            "freeze": 0,
        },
        "0619-b2-biagcau-topdown": {
            "name": "rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-0619-b2-biagcau-topdown-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi-0619-b2-biagcau-topdown.yaml",
            "freeze": 0,
        },
        "0619-b3-biagcau-full": {
            "name": "rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-0619-b3-biagcau-full-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi-0619-b3-biagcau-full.yaml",
            "freeze": 0,
        },
    },
    "small": {
        "plain": {
            "name": "rtdetr-swinv2-small-unfreeze-last3-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-small-last3-pretrained.yaml",
            "freeze": 0,
        },
        "ema": {
            "name": "rtdetr-swinv2-small-unfreeze-last3-ema-12k4k4k-e100",
            "model": "exp_cfg/detr/rtdetr-swinv2-small-last3-pretrained-ema.yaml",
            "freeze": 0,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Swin Transformer V2 RT-DETR experiments.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset YAML.")
    parser.add_argument("--project", type=str, required=True, help="Ultralytics project directory.")
    parser.add_argument("--device", type=str, default=None, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience in epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["tiny/ema", "tiny/b1-aifi-mdhifi", "tiny/b2-p3-mdhifi", "tiny/b3-p3-hifi"],
        choices=[
            "tiny/plain",
            "tiny/ema",
            "tiny/b1-aifi-mdhifi",
            "tiny/b2-p3-mdhifi",
            "tiny/b2-p3-mdhifi-direct-repc3-p3-aux-p4-p4mdhifi",
            "tiny/b3-p3-hifi",
            "tiny/b3-p3-aifi",
            "tiny/ema-wtconv-biagcau-mdhifi",
            "tiny/0619-b1-biagcau-p3",
            "tiny/0619-b2-biagcau-topdown",
            "tiny/0619-b3-biagcau-full",
            "small/plain",
            "small/ema",
        ],
        help="Subset of experiment variants to run.",
    )
    parser.add_argument("--exist-ok", action="store_true", default=False, help="Allow reuse of existing run directories.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print commands without executing them.")
    return parser.parse_args()


def build_train_command(args, size: str, flavor: str, cfg: dict) -> list[str]:
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
        cfg["name"],
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
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


def main():
    args = parse_args()
    for variant in args.variants:
        size, flavor = variant.split("/", 1)
        cfg = EXPERIMENTS[size][flavor]
        train_cmd = build_train_command(args, size, flavor, cfg)
        print(" ".join(train_cmd))
        if not args.dry_run:
            subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
