from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import yaml

import matplotlib.pyplot as plt

from ultralytics import RTDETR, YOLO
from ultralytics.models.rtdetr import RTDETRValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import smooth

import colorsys
import random


def gencolors(n):
    """
    通过在 HSL 空间引入多维随机偏移，生成视觉可分辨的色盘
    """
    colors = []

    # 将色相环等分为 n 份
    hue_step = 1.0 / n

    for i in range(n):
        # 1. 色相(H)：在每个区间内进行大范围偏移，但保证大基调差异
        base_hue = i * hue_step
        hue = (base_hue + random.uniform(0, hue_step * 0.8)) % 1.0

        # 2. 饱和度(S)：在 [0.5, 0.9] 范围内随机，避开灰暗颜色
        saturation = random.uniform(0.3, 0.9)

        # 3. 亮度(L)：在 [0.4, 0.7] 范围内随机，避开极亮（白）和极暗（黑）
        # 这样确保了所有颜色在显示器上都有足够的对比度
        lightness = random.uniform(0.3, 0.8)

        # 转换并存储
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    # 彻底打乱空间分布
    random.shuffle(colors)

    results = []
    for color in colors:
        # 将 RGB 元组转换为十六进制字符串格式
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        results.append(hex_color)

    return results


# CURVE_COLORS = [
#     "#1f77b4",
#     "#d62728",
#     "#2ca02c",
#     "#ff7f0e",
#     "#9467bd",
#     "#8c564b",
#     "#e377c2",
#     "#17becf",
#     "#b3b349",
#     "#7f7f7f",
#     "#393b79",
#     "#637939",
#     "#8c6d31",
#     "#843c39",
#     "#7b4173",
#     "#fff200",
#     "#dd875d",
#     "#63a176",
#     "#BFED4A",
# ]

# <display name>: <run name>
DEFAULT_MODELS = {
    # "yolo26s-p2": "yolo26s-p2-init",
    # "yolo11s-p2": "yolo11s-p2-init",
    # "yolov8s-p2": "yolov8s-p2-init",
    # "yolov5s-p2": "yolov5s-p2-init",
    # "yolo26s": "yolo26s-pretrained",
    # "yolo11s": "yolo11s-pretrained",
    # "yolov8s": "yolov8s-pretrained",
    # "yolov5s": "yolov5s-pretrained-2",
    # "yolo11s-ema-p2": "yolo11s-ema-p2-init-3",
    # "rtdetr-l": "redetr-l-init-2",
    # "rtdetr-p2": "rtdetr-p2-init",
    # "rtdetr-r18": "rtdetr-r18-init",
    # "rtdetr-HIFI": "rtdetr-hifi-init-2",
    # "rtdetr-WTConv": "rtdetr-wtconv-init",
    # "rtdetr-MRFPN": "rtdetr-mrfpn-init",
    # "anti-detr-3o": "rtdetr-HIFI-MRFPN-WTConv-init",
    # "rtdetr-dgwrn": "exp1-dgwrn",
    # "rtdetr-dgwrn-biagcau": "exp2-dgwrn-biagcau",
    # "rtdetr-dgwrn-biagcau-mdhifi": "exp3-dgwrn-biagcau-mdhifi",
    # "rtdetr-ema-p2-init": "rtdetr-ema-p2-init",
    # "rtdetr-resnet101": "rtdetr-resnet101",
    # "rtdetr-resnet50": "rtdetr-resnet50",
    # "rtdetr-exp3-ema-biagcau-mdhifi": "rtdetr-exp3-ema-biagcau-mdhifi",
    # "rtdetr-exp2-ema-biagcau": "rtdetr-exp2-ema-biagcau",
    # "rtdetr-exp1-ema": "rtdetr-exp1-ema",
    # "rtdetr-resnet50-unfreeze-last3-ema": "rtdetr-resnet50-unfreeze-last3-ema",
    # "rtdetr-resnet50-unfreeze-last3": "rtdetr-resnet50-unfreeze-last3",
    # "rtdetr-resnet50-freeze-last3-ema": "rtdetr-resnet50-freeze-last3-ema",
    # "rtdetr-resnet50-freeze-last3": "rtdetr-resnet50-freeze-last3",
    "rtdetr-resnet101-unfreeze-last3-ema-12k4k4k": "rtdetr-resnet101-unfreeze-last3-ema-12k4k4k",
    "rtdetr-resnet50-unfreeze-last3-ema-12k4k4k": "rtdetr-resnet50-unfreeze-last3-ema-12k4k4k",
    "rtdetr-resnet50-unfreeze-last3-12k4k4k": "rtdetr-resnet50-unfreeze-last3-12k4k4k",
    "rtdetr-resnet101-unfreeze-last3-12k4k4k": "rtdetr-resnet101-unfreeze-last3-12k4k4k-2",
    "rtdetr-resnet101-unfreeze-last3-ema-12k4k4k-e100": "rtdetr-resnet101-unfreeze-last3-ema-12k4k4k-e100",
    "rtdetr-resnet50-unfreeze-last3-ema-12k4k4k-e100": "rtdetr-resnet50-unfreeze-last3-ema-12k4k4k-e100",
    "rtdetr-swinv2-small-unfreeze-last3-12k4k4k-e100": "rtdetr-swinv2-small-unfreeze-last3-12k4k4k-e100",
    "rtdetr-swinv2-small-unfreeze-last3-ema-12k4k4k-e100": "rtdetr-swinv2-small-unfreeze-last3-ema-12k4k4k-e100",
    "rtdetr-swinv2-tiny-unfreeze-last3-12k4k4k-e100": "rtdetr-swinv2-tiny-unfreeze-last3-12k4k4k-e100-2",
    "rtdetr-swinv2-tiny-unfreeze-last3-12k4k4k-ema-e100": "rtdetr-swinv2-tiny-unfreeze-last3-ema-12k4k4k-e100-2",
    "rtdetr-resnet101-unfreeze-last3-ema-wtconv-biagcau-mdhifi-12k4k4k-e100": "rtdetr-resnet101-unfreeze-last3-ema-wtconv-biagcau-mdhifi-12k4k4k-e100",
    "rtdetr-swinv2-tiny-unfreeze-last3-ema-wtconv-biagcau-mdhifi-12k4k4k-e100": "rtdetr-swinv2-tiny-unfreeze-last3-ema-wtconv-biagcau-mdhifi-12k4k4k-e100",
}

CURVE_COLORS = gencolors(len(DEFAULT_MODELS))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare best.pt validation curves across multiple runs."
    )
    parser.add_argument(
        "--project-path",
        type=Path,
        default=Path("runs/detect/cst-sample-12k4k4k-s100"),
        help="Project folder that contains multiple run directories.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="DISPLAY=RUN_NAME",
        help="Add or override one mapping entry. Can be passed multiple times.",
    )
    parser.add_argument(
        "--clear-defaults",
        action="store_true",
        help="Ignore DEFAULT_MODELS. If no --model is provided, auto-scan all runs under --project-path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="runs/plots/rtdetrs-12k4k4ks100",
        help="Directory for merged figures and report. Defaults to <project-path>/best_curve_report.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override validation batch size for all models.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override validation image size for all models.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Override validation device, e.g. 0 or cpu.",
    )
    return parser.parse_args()


def parse_model_overrides(raw_items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --model '{item}', expected DISPLAY=RUN_NAME")
        display_name, run_name = item.split("=", 1)
        display_name = display_name.strip()
        run_name = run_name.strip()
        if not display_name or not run_name:
            raise ValueError(
                f"Invalid --model '{item}', DISPLAY and RUN_NAME must be non-empty"
            )
        parsed[display_name] = run_name
    return parsed


def scan_project_runs(project_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for run_dir in sorted(project_path.iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "weights" / "best.pt").exists():
            mapping[run_dir.name] = run_dir.name
    return mapping


def build_model_mapping(args: argparse.Namespace, project_path: Path) -> dict[str, str]:
    mapping = {} if args.clear_defaults else dict(DEFAULT_MODELS)
    mapping.update(parse_model_overrides(args.model))
    if not mapping:
        mapping = scan_project_runs(project_path)
    if not mapping:
        raise ValueError(f"No runs with weights/best.pt found under {project_path}")
    return mapping


def load_run_args(run_dir: Path) -> dict:
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing args.yaml: {args_path}")
    with args_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_model_channels(model) -> int:
    yaml_channels = getattr(getattr(model, "model", None), "yaml", {}).get("channels")
    if yaml_channels is not None:
        return int(yaml_channels)

    module = getattr(model, "model", model)
    for child in module.modules():
        conv = getattr(child, "conv", None)
        if conv is not None and hasattr(conv, "in_channels"):
            return int(conv.in_channels)
        if hasattr(child, "in_channels"):
            return int(child.in_channels)
    return 3


def build_compatible_data_path(
    data_path: str | os.PathLike, expected_channels: int, save_dir: Path
) -> tuple[str, bool]:
    data_path = Path(data_path)
    if data_path.suffix not in {".yaml", ".yml"}:
        return str(data_path), False

    with data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    current_channels = int(data_cfg.get("channels", 3))
    if current_channels == expected_channels:
        return str(data_path), False

    compat_dir = save_dir / "compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    compat_path = (
        compat_dir / f"{data_path.stem}.channels-{expected_channels}{data_path.suffix}"
    )
    compat_cfg = dict(data_cfg)
    compat_cfg["channels"] = int(expected_channels)
    with compat_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(compat_cfg, f, sort_keys=False, allow_unicode=True)
    return str(compat_path), True


def pick_model_and_validator(best_pt: Path, train_args: dict):
    model_hint = str(train_args.get("model", "")).lower()
    if "rtdetr" in model_hint or "rtdetr" in best_pt.name.lower():
        return RTDETR(str(best_pt)), RTDETRValidator
    return YOLO(str(best_pt)), DetectionValidator


def collect_curves(
    display_name: str,
    run_name: str,
    project_path: Path,
    output_dir: Path,
    batch_override: int | None,
    imgsz_override: int | None,
    device_override: str | None,
) -> dict:
    run_dir = project_path / run_name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {best_pt}")

    train_args = load_run_args(run_dir)
    model, validator_cls = pick_model_and_validator(best_pt, train_args)
    model_channels = infer_model_channels(model)

    batch = (
        batch_override
        if batch_override is not None
        else max(int(train_args.get("batch", 1)) * 2, 1)
    )
    imgsz = (
        imgsz_override
        if imgsz_override is not None
        else int(train_args.get("imgsz", 640))
    )
    device = (
        device_override
        if device_override is not None
        else (train_args.get("device") or "")
    )
    split = train_args.get("split", "val")

    save_dir = output_dir / run_name
    data_path, patched_channels = build_compatible_data_path(
        train_args["data"], model_channels, save_dir
    )
    validator_args = {
        "model": str(best_pt),
        "data": data_path,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "split": split,
        "plots": True,
        "save_json": False,
        "save_txt": False,
        "save_conf": False,
        "save_crop": False,
        "rect": False,
        "cache": train_args.get("cache", False),
        "workers": train_args.get("workers", 8),
        "project": str(output_dir),
        "name": run_name,
        "exist_ok": True,
        "mode": "val",
    }
    validator = validator_cls(save_dir=save_dir, args=validator_args)
    stats = validator(model=model.model)

    plot_data = {}
    for plot_info in validator.plots.values():
        data = plot_info.get("data")
        if data and "type" in data:
            plot_data[data["type"]] = data

    required = {"pr_curve", "precision_curve", "recall_curve"}
    missing = required - plot_data.keys()
    if missing:
        raise RuntimeError(f"Missing curve data for {display_name}: {sorted(missing)}")

    pr_y = np.asarray(plot_data["pr_curve"]["y"], dtype=float)
    p_y = np.asarray(plot_data["precision_curve"]["y"], dtype=float)
    r_y = np.asarray(plot_data["recall_curve"]["y"], dtype=float)

    return {
        "display_name": display_name,
        "run_name": run_name,
        "run_dir": run_dir,
        "save_dir": save_dir,
        "metrics": stats,
        "data_path": data_path,
        "patched_channels": patched_channels,
        "model_channels": model_channels,
        "pr_x": np.asarray(plot_data["pr_curve"]["x"], dtype=float),
        "pr_y": pr_y.mean(axis=0),
        "p_x": np.asarray(plot_data["precision_curve"]["x"], dtype=float),
        "p_y": smooth(p_y.mean(axis=0), 0.1),
        "r_x": np.asarray(plot_data["recall_curve"]["x"], dtype=float),
        "r_y": smooth(r_y.mean(axis=0), 0.1),
    }


def plot_combined_curve(
    records: list[dict],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    for idx, record in enumerate(records):
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]
        ax.plot(
            record[x_key],
            record[y_key],
            linewidth=2,
            color=color,
            label=record["display_name"],
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def write_report(records: list[dict], project_path: Path, output_dir: Path):
    report_path = output_dir / "report.md"
    lines = [
        "# Best.pt Curve Report",
        "",
        f"- Project path: `{project_path}`",
        f"- Output dir: `{output_dir}`",
        "",
        "## Models",
        "",
        "| Display Name | Run Name | Precision | Recall | mAP50 | mAP50-95 | Run Dir |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        metrics = record["metrics"]
        lines.append(
            "| "
            f"{record['display_name']} | {record['run_name']} | "
            f"{metrics['metrics/precision(B)']:.4f} | {metrics['metrics/recall(B)']:.4f} | "
            f"{metrics['metrics/mAP50(B)']:.4f} | {metrics['metrics/mAP50-95(B)']:.4f} | "
            f"`{record['run_dir']}` |"
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "### Precision-Confidence",
            "",
            "![P curve](compare_p_curve.png)",
            "",
            "### Recall-Confidence",
            "",
            "![R curve](compare_r_curve.png)",
            "",
            "### Precision-Recall",
            "",
            "![PR curve](compare_pr_curve.png)",
            "",
            "## Validator Outputs",
            "",
        ]
    )

    for record in records:
        lines.extend(
            [
                f"### {record['display_name']}",
                "",
                f"- Run name: `{record['run_name']}`",
                f"- Validation output: `{record['save_dir']}`",
                f"- Validator data: `{record['data_path']}`",
                f"- Model channels: `{record['model_channels']}`",
                f"- Channel compat override: `{'yes' if record['patched_channels'] else 'no'}`",
                "",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    project_path = args.project_path.resolve()
    mapping = build_model_mapping(args, project_path)
    output_dir = (args.output_dir or (project_path / "best_curve_report")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for display_name, run_name in mapping.items():
        print(f"[INFO] validating {display_name} <- {run_name}")
        records.append(
            collect_curves(
                display_name=display_name,
                run_name=run_name,
                project_path=project_path,
                output_dir=output_dir,
                batch_override=args.batch,
                imgsz_override=args.imgsz,
                device_override=args.device,
            )
        )

    plot_combined_curve(
        records,
        x_key="p_x",
        y_key="p_y",
        title="Precision-Confidence Curve",
        xlabel="Confidence",
        ylabel="Precision",
        output_path=output_dir / "compare_p_curve.png",
    )
    plot_combined_curve(
        records,
        x_key="r_x",
        y_key="r_y",
        title="Recall-Confidence Curve",
        xlabel="Confidence",
        ylabel="Recall",
        output_path=output_dir / "compare_r_curve.png",
    )
    plot_combined_curve(
        records,
        x_key="pr_x",
        y_key="pr_y",
        title="Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        output_path=output_dir / "compare_pr_curve.png",
    )
    write_report(records, project_path, output_dir)

    print(f"[OK] wrote report to {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
