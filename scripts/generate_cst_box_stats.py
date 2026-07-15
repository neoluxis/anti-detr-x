from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from PIL import Image


DATASET_ROOT = Path("datasets/CST_AntiUAV/cst-sample_train-12000_val-4000_test-4000_seq-100_id-0")
OUTDIR = Path("assets/detr0705/cst_box_stats")
SPLITS = ("train", "val", "test")


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def summarize(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {
        "min": values[0],
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "mean": sum(values) / len(values),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.9),
        "p95": percentile(values, 0.95),
        "max": values[-1],
    }


def read_dataset() -> tuple[dict[str, dict[str, list[float]]], list[dict[str, object]]]:
    split_values: dict[str, dict[str, list[float]]] = {
        split: {"width_px": [], "height_px": [], "area_px": []} for split in SPLITS
    }
    seq_rows: list[dict[str, object]] = []

    for split in SPLITS:
        by_seq: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "images": 0,
                "labeled_images": 0,
                "objects": 0,
                "width_px": [],
                "height_px": [],
                "area_px": [],
            }
        )
        for label_path in sorted((DATASET_ROOT / split / "labels").rglob("*.txt")):
            seq = label_path.parent.name
            image_path = DATASET_ROOT / split / "images" / seq / f"{label_path.stem}.jpg"
            if not image_path.exists():
                candidates = list((DATASET_ROOT / split / "images" / seq).glob(f"{label_path.stem}.*"))
                if not candidates:
                    continue
                image_path = candidates[0]
            with Image.open(image_path) as image:
                image_w, image_h = image.size

            seq_data = by_seq[seq]
            seq_data["images"] += 1

            lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
            if lines:
                seq_data["labeled_images"] += 1
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, _, _, w, h = parts[:5]
                width_px = float(w) * image_w
                height_px = float(h) * image_h
                area_px = width_px * height_px
                split_values[split]["width_px"].append(width_px)
                split_values[split]["height_px"].append(height_px)
                split_values[split]["area_px"].append(area_px)
                seq_data["objects"] += 1
                seq_data["width_px"].append(width_px)
                seq_data["height_px"].append(height_px)
                seq_data["area_px"].append(area_px)

        for seq, seq_data in sorted(by_seq.items()):
            if seq_data["objects"] == 0:
                row = {
                    "split": split,
                    "sequence": seq,
                    "images": seq_data["images"],
                    "labeled_images": seq_data["labeled_images"],
                    "empty_images": seq_data["images"] - seq_data["labeled_images"],
                    "objects": 0,
                    "width_mean_px": 0.0,
                    "width_median_px": 0.0,
                    "width_p90_px": 0.0,
                    "height_mean_px": 0.0,
                    "height_median_px": 0.0,
                    "height_p90_px": 0.0,
                    "area_mean_px2": 0.0,
                    "area_median_px2": 0.0,
                    "area_p90_px2": 0.0,
                    "small_lt_32_ratio": 0.0,
                }
            else:
                width_stats = summarize(seq_data["width_px"])
                height_stats = summarize(seq_data["height_px"])
                area_stats = summarize(seq_data["area_px"])
                small_lt_32 = sum(1 for area in seq_data["area_px"] if area < 32 * 32)
                row = {
                    "split": split,
                    "sequence": seq,
                    "images": seq_data["images"],
                    "labeled_images": seq_data["labeled_images"],
                    "empty_images": seq_data["images"] - seq_data["labeled_images"],
                    "objects": seq_data["objects"],
                    "width_mean_px": width_stats["mean"],
                    "width_median_px": width_stats["median"],
                    "width_p90_px": width_stats["p90"],
                    "height_mean_px": height_stats["mean"],
                    "height_median_px": height_stats["median"],
                    "height_p90_px": height_stats["p90"],
                    "area_mean_px2": area_stats["mean"],
                    "area_median_px2": area_stats["median"],
                    "area_p90_px2": area_stats["p90"],
                    "small_lt_32_ratio": small_lt_32 / seq_data["objects"],
                }
            seq_rows.append(row)

    return split_values, seq_rows


def plot_histograms(split_values: dict[str, dict[str, list[float]]]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    metrics = [
        ("width_px", "Box Width (px)", 40),
        ("height_px", "Box Height (px)", 30),
        ("area_px", "Box Area (px^2)", 40),
    ]
    split_colors = {"train": "#4E79A7", "val": "#F28E2B", "test": "#59A14F"}

    for col, (metric_key, metric_label, bins) in enumerate(metrics):
        for row, split in enumerate(SPLITS):
            ax = axes[row][col]
            values = split_values[split][metric_key]
            ax.hist(values, bins=bins, color=split_colors[split], alpha=0.9, edgecolor="white")
            ax.set_title(f"{split.upper()} {metric_label}")
            ax.set_xlabel(metric_label)
            ax.set_ylabel("Count")
            ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle("CST Box Size Distribution by Split", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTDIR / "cst_box_size_histograms.png", dpi=200)
    plt.close(fig)


def plot_boxplots(split_values: dict[str, dict[str, list[float]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metrics = [
        ("width_px", "Box Width (px)"),
        ("height_px", "Box Height (px)"),
        ("area_px", "Box Area (px^2)"),
    ]
    split_colors = ["#4E79A7", "#F28E2B", "#59A14F"]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        data = [split_values[split][metric_key] for split in SPLITS]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 2},
            boxprops={"linewidth": 1.5},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
        )
        ax.set_xticks(range(1, len(SPLITS) + 1), [split.upper() for split in SPLITS])
        for patch, color in zip(bp["boxes"], split_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(metric_label)
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle("CST Box Size Boxplots by Split", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTDIR / "cst_box_size_boxplots.png", dpi=200)
    plt.close(fig)


def write_sequence_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "split",
        "sequence",
        "images",
        "labeled_images",
        "empty_images",
        "objects",
        "width_mean_px",
        "width_median_px",
        "width_p90_px",
        "height_mean_px",
        "height_median_px",
        "height_p90_px",
        "area_mean_px2",
        "area_median_px2",
        "area_p90_px2",
        "small_lt_32_ratio",
    ]
    with (OUTDIR / "cst_sequence_box_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_summary(rows: list[dict[str, object]]) -> None:
    top_objects = sorted(rows, key=lambda row: (-int(row["objects"]), str(row["split"]), str(row["sequence"])))[:20]
    top_large = sorted(rows, key=lambda row: (-float(row["area_median_px2"]), -int(row["objects"])))[:20]
    lines = [
        "# CST Sequence Box Stats",
        "",
        "## Files",
        "",
        "- `cst_box_size_histograms.png`: train/val/test width, height, area histograms",
        "- `cst_box_size_boxplots.png`: train/val/test width, height, area boxplots",
        "- `cst_sequence_box_stats.csv`: per-sequence box statistics",
        "",
        "## Top 20 Sequences by Object Count",
        "",
        "| split | sequence | images | labeled | empty | objects | width_median_px | height_median_px | area_median_px2 | <32x32 ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_objects:
        lines.append(
            f"| {row['split']} | {row['sequence']} | {row['images']} | {row['labeled_images']} | {row['empty_images']} | "
            f"{row['objects']} | {float(row['width_median_px']):.2f} | {float(row['height_median_px']):.2f} | "
            f"{float(row['area_median_px2']):.2f} | {float(row['small_lt_32_ratio']) * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Top 20 Sequences by Median Area",
            "",
            "| split | sequence | objects | width_median_px | height_median_px | area_median_px2 | area_p90_px2 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_large:
        lines.append(
            f"| {row['split']} | {row['sequence']} | {row['objects']} | {float(row['width_median_px']):.2f} | "
            f"{float(row['height_median_px']):.2f} | {float(row['area_median_px2']):.2f} | {float(row['area_p90_px2']):.2f} |"
        )

    (OUTDIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    split_values, rows = read_dataset()
    plot_histograms(split_values)
    plot_boxplots(split_values)
    write_sequence_csv(rows)
    write_markdown_summary(rows)
    print(OUTDIR / "cst_box_size_histograms.png")
    print(OUTDIR / "cst_box_size_boxplots.png")
    print(OUTDIR / "cst_sequence_box_stats.csv")
    print(OUTDIR / "README.md")


if __name__ == "__main__":
    main()
