from __future__ import annotations

import argparse
import csv
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

RUN_COLS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map5095": "metrics/mAP50-95(B)",
}


@dataclass(frozen=True)
class ExperimentSpec:
    code: str
    label: str
    run_name: str


@dataclass(frozen=True)
class ExperimentGroup:
    slug: str
    title: str
    docs: tuple[str, ...]
    experiments: tuple[ExperimentSpec, ...]
    note: str = ""


GROUPS = (
    ExperimentGroup(
        slug="0616_mdhifi_positions",
        title="0616 MDHIFI insertion study",
        docs=("0616实验.docx", "0616实验(1).docx"),
        experiments=(
            ExperimentSpec(
                code="B0",
                label="B0 baseline",
                run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-12k4k4k-e100-2",
            ),
            ExperimentSpec(
                code="B1",
                label="B1 AIFI+MDHIFI",
                run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b1-aifi-mdhifi-12k4k4k-e100",
            ),
            ExperimentSpec(
                code="B2",
                label="B2 P3+MDHIFI",
                run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-12k4k4k-e100-3",
            ),
            ExperimentSpec(
                code="B3",
                label="B3 P3+HIFI",
                run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b3-p3-hifi-12k4k4k-e100",
            ),
        ),
        note=(
            "B3 follows the table wording 'HIFI vs MDHIFI'. "
            "The alternate P3+AIFI run is not used in this figure."
        ),
    ),
    ExperimentGroup(
        slug="0616_full_module_vs_baseline",
        title="0616 baseline vs full WTConv+BiAGCAU+MDHIFI model",
        docs=("0616实验.docx", "0616实验(1).docx"),
        experiments=(
            ExperimentSpec(
                code="Base",
                label="Baseline",
                run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-12k4k4k-e100-2",
            ),
            ExperimentSpec(
                code="Full",
                label="WTConv+BiAGCAU+MDHIFI",
                run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-wtconv-biagcau-mdhifi-12k4k4k-e100",
            ),
        ),
        note="Summary metrics are derived from the best mAP50 epoch in results.csv.",
    ),
    ExperimentGroup(
        slug="0619_biagcau_positions",
        title="0619 BiAGCAU insertion study",
        docs=("0619实验.docx",),
        experiments=(
            ExperimentSpec(
                code="B0",
                label="B0 P3+MDHIFI baseline",
                run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-12k4k4k-e100-3",
            ),
            ExperimentSpec(
                code="B1",
                label="B1 P3 BiAGCAU",
                run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-0619-b1-biagcau-p3-12k4k4k-e100",
            ),
            ExperimentSpec(
                code="B2",
                label="B2 top-down BiAGCAU",
                run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-0619-b2-biagcau-topdown-12k4k4k-e100",
            ),
            ExperimentSpec(
                code="B3",
                label="B3 full BiAGCAU",
                run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-0619-b3-biagcau-full-12k4k4k-e100",
            ),
        ),
        note=(
            "The doc table is unfilled, so summary metrics are derived from the "
            "best mAP50 epoch in results.csv."
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment summaries from the 0616/0619 docx notes.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("runs/detect/cst-sample-12k4k4k-s100"),
        help="Directory containing the experiment run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/plots/docx_experiments"),
        help="Directory for generated plots and tables.",
    )
    return parser.parse_args()


def read_docx_lines(docx_path: Path) -> list[str]:
    with ZipFile(docx_path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))

    lines: list[str] = []
    for paragraph in root.findall(".//w:p", NS):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", NS)).strip()
        if text:
            lines.append(text)
    return lines


def extract_doc_meta(lines: list[str]) -> tuple[str, str]:
    title = next((line for line in lines if line.startswith("实验：")), "实验：unknown")
    date = next((line for line in lines if line.startswith("日期：")), "日期：unknown")
    return title.removeprefix("实验：").strip(), date.removeprefix("日期：").strip()


def parse_doc_metric_table(lines: list[str]) -> dict[str, dict[str, float | int | str]]:
    headers = ["Model", "Pretrain", "Dataset", "Epochs", "Precision", "Recall", "mAP50", "mAP50-95", "F1 Score"]
    header_start = None
    for idx in range(len(lines) - len(headers) + 1):
        if lines[idx : idx + len(headers)] == headers:
            header_start = idx + len(headers)
            break
    if header_start is None:
        return {}

    rows: dict[str, dict[str, float | int | str]] = {}
    idx = header_start
    while idx < len(lines):
        if not re.fullmatch(r"B\d+", lines[idx]):
            idx += 1
            continue

        code = lines[idx]
        values = [code]
        cursor = idx + 1
        while cursor < len(lines) and not re.fullmatch(r"B\d+", lines[cursor]):
            line = lines[cursor]
            if line.startswith("与仅引入") or line.startswith("Curves"):
                break
            values.append(line)
            cursor += 1
            if len(values) >= len(headers):
                break

        if len(values) == len(headers):
            try:
                rows[code] = {
                    "source": "doc",
                    "pretrain": values[1],
                    "dataset": values[2],
                    "epoch": int(values[3]),
                    "precision": float(values[4]),
                    "recall": float(values[5]),
                    "map50": float(values[6]),
                    "map5095": float(values[7]),
                    "f1": float(values[8]),
                }
            except ValueError:
                pass

        idx = max(cursor, idx + 1)

    return rows


def load_run_history(run_dir: Path) -> list[dict[str, float]]:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.csv: {results_path}")

    history: list[dict[str, float]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            precision = float(raw[RUN_COLS["precision"]])
            recall = float(raw[RUN_COLS["recall"]])
            history.append(
                {
                    "epoch": float(raw["epoch"]),
                    "precision": precision,
                    "recall": recall,
                    "map50": float(raw[RUN_COLS["map50"]]),
                    "map5095": float(raw[RUN_COLS["map5095"]]),
                    "f1": f1_score(precision, recall),
                }
            )
    return history


def f1_score(precision: float, recall: float) -> float:
    denom = precision + recall
    return 0.0 if denom == 0 else 2.0 * precision * recall / denom


def best_epoch_row(history: list[dict[str, float]], metric: str = "map50") -> dict[str, float]:
    return max(history, key=lambda row: row[metric])


def collect_group_rows(
    group: ExperimentGroup,
    project_dir: Path,
    repo_root: Path,
) -> tuple[list[dict[str, float | int | str]], dict[str, list[dict[str, float]]], list[str], list[str]]:
    doc_metric_rows: dict[str, dict[str, float | int | str]] = {}
    doc_titles: list[str] = []
    doc_dates: list[str] = []
    for doc_name in group.docs:
        lines = read_docx_lines(repo_root / doc_name)
        title, date = extract_doc_meta(lines)
        doc_titles.append(f"{doc_name}: {title}")
        doc_dates.append(f"{doc_name}: {date}")
        doc_metric_rows.update(parse_doc_metric_table(lines))

    summary_rows: list[dict[str, float | int | str]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for exp in group.experiments:
        run_dir = project_dir / exp.run_name
        history = load_run_history(run_dir)
        histories[exp.code] = history
        if exp.code in doc_metric_rows:
            row = dict(doc_metric_rows[exp.code])
        else:
            best = best_epoch_row(history, "map50")
            row = {
                "source": "run_best_map50",
                "epoch": int(best["epoch"]),
                "precision": best["precision"],
                "recall": best["recall"],
                "map50": best["map50"],
                "map5095": best["map5095"],
                "f1": best["f1"],
                "dataset": "12k4k4k",
                "pretrain": "true",
            }

        row["code"] = exp.code
        row["label"] = exp.label
        row["run_name"] = exp.run_name
        summary_rows.append(row)

    return summary_rows, histories, doc_titles, doc_dates


def pct_formatter(value: float, _position: float) -> str:
    return f"{value:.0f}%"


def plot_summary(rows: list[dict[str, float | int | str]], title: str, output_path: Path, note: str) -> None:
    metric_specs = (
        ("precision", "Precision", "#1f4e79"),
        ("recall", "Recall", "#3d9970"),
        ("f1", "F1", "#ff851b"),
        ("map50", "mAP50", "#b10dc9"),
        ("map5095", "mAP50-95", "#c0392b"),
    )

    labels = [str(row["code"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 6.8))
    for idx, (key, display, color) in enumerate(metric_specs):
        offsets = x + (idx - 2) * width
        values = [float(row[key]) * 100.0 for row in rows]
        bars = ax.bar(offsets, values, width=width, label=display, color=color, alpha=0.9)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.yaxis.set_major_formatter(pct_formatter)
    ax.set_ylim(0, max(float(row["precision"]) for row in rows) * 100.0 + 12.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))

    label_lines = [f"{row['code']}: {row['label']}" for row in rows]
    source_lines = [f"{row['code']}={row['source']}" for row in rows]
    fig.text(0.02, 0.02, " | ".join(label_lines), fontsize=9)
    fig.text(0.02, 0.005, f"metric source: {' | '.join(source_lines)}; {note}", fontsize=8.5)

    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_curves(
    rows: list[dict[str, float | int | str]],
    histories: dict[str, list[dict[str, float]]],
    title: str,
    output_path: Path,
) -> None:
    metric_specs = (
        ("precision", "Precision", "#1f4e79"),
        ("recall", "Recall", "#3d9970"),
        ("map50", "mAP50", "#b10dc9"),
        ("map5095", "mAP50-95", "#c0392b"),
    )
    line_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#17becf"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()

    for ax, (metric_key, metric_name, highlight) in zip(axes, metric_specs):
        for idx, row in enumerate(rows):
            code = str(row["code"])
            history = histories[code]
            epochs = [item["epoch"] for item in history]
            values = [item[metric_key] * 100.0 for item in history]
            ax.plot(
                epochs,
                values,
                linewidth=2.0,
                color=line_colors[idx % len(line_colors)],
                label=f"{code}: {row['label']}",
            )
            best = max(history, key=lambda item: item[metric_key])
            ax.scatter(
                [best["epoch"]],
                [best[metric_key] * 100.0],
                color=highlight,
                s=28,
                zorder=5,
            )

        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score (%)")
        ax.yaxis.set_major_formatter(pct_formatter)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_group_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    fieldnames = ["code", "label", "run_name", "source", "epoch", "precision", "recall", "map50", "map5095", "f1", "dataset", "pretrain"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readme(output_dir: Path, sections: list[dict[str, object]]) -> None:
    lines = [
        "# Docx experiment plots",
        "",
        "Generated from the 0616/0619 experiment notes and the corresponding run folders.",
        "",
    ]

    for section in sections:
        lines.append(f"## {section['title']}")
        lines.append("")
        lines.append(f"- Docs: {', '.join(section['docs'])}")
        lines.append(f"- Plot summary: `{section['summary_png']}`")
        lines.append(f"- Plot curves: `{section['curves_png']}`")
        lines.append(f"- Table: `{section['csv']}`")
        lines.append(f"- Note: {section['note']}")
        lines.append("")
        lines.append("| Code | Label | Source | Epoch | Precision | Recall | mAP50 | mAP50-95 | F1 |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in section["rows"]:
            lines.append(
                "| {code} | {label} | {source} | {epoch} | {precision:.4f} | {recall:.4f} | {map50:.4f} | {map5095:.4f} | {f1:.4f} |".format(
                    code=row["code"],
                    label=row["label"],
                    source=row["source"],
                    epoch=int(row["epoch"]),
                    precision=float(row["precision"]),
                    recall=float(row["recall"]),
                    map50=float(row["map50"]),
                    map5095=float(row["map5095"]),
                    f1=float(row["f1"]),
                )
            )
        lines.append("")

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    sections: list[dict[str, object]] = []
    for group in GROUPS:
        rows, histories, doc_titles, doc_dates = collect_group_rows(group, args.project_dir, repo_root)
        csv_path = output_dir / f"{group.slug}.csv"
        summary_path = output_dir / f"{group.slug}_summary.png"
        curves_path = output_dir / f"{group.slug}_curves.png"

        write_group_csv(rows, csv_path)
        plot_summary(rows, group.title, summary_path, group.note)
        plot_curves(rows, histories, f"{group.title} training curves", curves_path)

        sections.append(
            {
                "title": group.title,
                "docs": [*doc_titles, *doc_dates],
                "summary_png": summary_path.name,
                "curves_png": curves_path.name,
                "csv": csv_path.name,
                "note": group.note,
                "rows": rows,
            }
        )

    write_readme(output_dir, sections)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
