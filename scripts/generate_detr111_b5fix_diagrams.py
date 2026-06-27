from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_swinv2_b2_variant_principles import (
    BG,
    BLUE,
    BLUE2,
    BLUE3,
    ORANGE,
    PANEL,
    PEACH,
    PINK,
    PURPLE,
    RED,
    TAN,
    Box,
    Edge,
    Variant,
    drawio_cell_box,
    drawio_edge,
    drawio_title,
    svg_box,
    svg_edge,
)


OUTDIR = Path("assets/figures")
DRAWIO_PATH = OUTDIR / "rtdetr_swinv2_tiny_detr111_b5fix_principles.drawio"


@dataclass(frozen=True)
class Detr111Variant:
    title: str
    page_name: str
    svg_name: str
    run_name: str
    yaml_name: str
    note: str
    p4_mdhifi: bool
    p3_upsample: bool


VARIANTS = [
    Detr111Variant(
        title="DETR111-05 B5 Stepwise Lateral P4-MDHIFI",
        page_name="detr111-05-b5-stepwise-lateral-p4mdhifi",
        svg_name="rtdetr_swinv2_tiny_detr111_05_b5_stepwise_lateral_p4mdhifi.svg",
        run_name="rtdetr-swinv2-tiny-detr111-05-b5-stepwise-lateral-p4mdhifi-12k4k4k-e100",
        yaml_name="exp_cfg/detr111/05-b5-stepwise-lateral-p4mdhifi.yaml",
        note="Full corrected B5: P4 input_proj.1 is refined by MDHIFI, P3 uses Up(Y4)+P3m, A4 injects into P4, and A5 from P4m injects into P5.",
        p4_mdhifi=True,
        p3_upsample=True,
    ),
    Detr111Variant(
        title="DETR111-06 B5 No P3 Upsample",
        page_name="detr111-06-b5-no-p3-upsample",
        svg_name="rtdetr_swinv2_tiny_detr111_06_b5_no_p3_upsample.svg",
        run_name="rtdetr-swinv2-tiny-detr111-06-b5-no-p3-upsample-12k4k4k-e100",
        yaml_name="exp_cfg/detr111/06-b5-stepwise-lateral-p4mdhifi-no-p3-upsample.yaml",
        note="Same corrected B5 lateral path as 05, but P3_out is produced directly from P3m without Up(Y4).",
        p4_mdhifi=True,
        p3_upsample=False,
    ),
    Detr111Variant(
        title="DETR111-07 B5 No P4-MDHIFI",
        page_name="detr111-07-b5-no-p4mdhifi",
        svg_name="rtdetr_swinv2_tiny_detr111_07_b5_no_p4mdhifi.svg",
        run_name="rtdetr-swinv2-tiny-detr111-07-b5-no-p4mdhifi-12k4k4k-e100",
        yaml_name="exp_cfg/detr111/07-b5-stepwise-lateral-no-p4mdhifi.yaml",
        note="P4 branch falls back to plain top-down fusion; A5 is generated from Y4 instead of P4m.",
        p4_mdhifi=False,
        p3_upsample=True,
    ),
]


def build_boxes(v: Detr111Variant) -> dict[str, Box]:
    p4_td_label = "Concat Up(Y5)+P4m" if v.p4_mdhifi else "Concat Up(Y5)+P4"
    p3_label = "Concat Up(Y4)+P3m\nRepC3 -> P3_out [80]" if v.p3_upsample else "P3m -> RepC3\nP3_out [80]"
    p3_w = 240 if v.p3_upsample else 190
    p4_aux_source = "P4m" if v.p4_mdhifi else "Y4"
    p5_concat = f"Concat Down(P4_out)+Y5+A5\nRepC3 -> P5_out [20]"

    items = [
        Box("input", "Input image\n640 x 640", 70, 145, 220, 52, ORANGE),
        Box("backbone", "TorchVision SwinV2-Tiny\nStage2-4 outputs", 70, 230, 220, 72, "#9ec8c0"),
        Box("p3", "P3/8 feature [80 x 80]\nIndex + Permute + EMA", 70, 360, 220, 72, BLUE),
        Box("p4", "P4/16 feature [40 x 40]\nIndex + Permute + EMA", 70, 490, 220, 72, BLUE2),
        Box("p5", "P5/32 feature [20 x 20]\nIndex + Permute + EMA", 70, 620, 220, 72, BLUE3),
        Box("p5_proj", "P5 1x1 Conv\n[20 x 20]", 420, 145, 150, 54, TAN),
        Box("aifi", "AIFI encoder\n[20 x 20]", 620, 145, 165, 54, PURPLE),
        Box("y5", "Y5\n[20 x 20]", 850, 145, 155, 54, PEACH),
        Box("p4_proj", "P4 1x1 Conv\n[40 x 40]", 420, 275, 150, 54, TAN),
        Box("p4_midhifi_or_td", p4_td_label, 620, 268, 210, 68, TAN),
        Box("y4_or_p4m", "P4 MDHIFI\nP4m [40 x 40]" if v.p4_mdhifi else "RepC3 -> Y4\n[40 x 40]", 890, 272, 190, 62, PINK if v.p4_mdhifi else PEACH, stroke="#8f2742" if v.p4_mdhifi else "#263746"),
        Box("y4", "Y4\n[40 x 40]" if v.p4_mdhifi else "Y4 already above", 1115, 272, 120, 62, PEACH if v.p4_mdhifi else BG, stroke="#263746" if v.p4_mdhifi else BG),
        Box("p3_proj", "P3 1x1 Conv\n[80 x 80]", 420, 405, 150, 54, TAN),
        Box("p3m", "P3 MDHIFI\nP3m [80 x 80]", 620, 398, 165, 68, PINK, stroke="#8f2742"),
        Box("p3_out", p3_label, 850, 392, p3_w, 78 if v.p3_upsample else 68, PEACH),
        Box("a4", "A4: Down(P3m)\nConv s=2\n80 -> 40", 620, 565, 190, 70, "#ffe4dc", stroke=RED),
        Box("p4_out", "Concat A4 + Y4\nRepC3 -> P4_out [40]", 875, 560, 220, 74, PEACH),
        Box("a5", f"A5: Down({p4_aux_source})\nConv s=2\n40 -> 20", 405, 710, 170, 78, "#ffe4dc", stroke=RED),
        Box("p5_down", "Down(P4_out)\nConv s=2\n40 -> 20", 620, 720, 190, 70, TAN),
        Box("p5_out", p5_concat, 875, 713, 240, 78, PEACH),
        Box("multi", "Multi-scale features\nP3 80 / P4 40 / P5 20", 1230, 495, 190, 100, "#f8eee3", stroke="#bba58b"),
        Box("decoder", "RT-DETR decoder", 1500, 485, 170, 120, "#f8b39b", font_size=18),
    ]
    return {box.key: box for box in items}


def build_edges(v: Detr111Variant) -> list[Edge]:
    edges = [
        Edge("input", "backbone"),
        Edge("backbone", "p3"),
        Edge("p3", "p4"),
        Edge("p4", "p5"),
        Edge("p5", "p5_proj", points=((340, 656), (340, 172))),
        Edge("p4", "p4_proj", points=((350, 526), (350, 302))),
        Edge("p3", "p3_proj"),
        Edge("p5_proj", "aifi"),
        Edge("aifi", "y5"),
        Edge("y5", "p4_midhifi_or_td", points=((1120, 172), (1120, 235), (725, 235))),
        Edge("p4_proj", "p4_midhifi_or_td"),
        Edge("p3_proj", "p3m"),
        Edge("p3m", "a4", color=RED, dashed=True, points=((690, 500), (690, 535))),
        Edge("a4", "p4_out", color=RED, dashed=True),
        Edge("p4_out", "p5_down"),
        Edge("p5_down", "p5_out"),
        Edge("a5", "p5_out", color=RED, dashed=True),
        Edge("y5", "p5_out", points=((1135, 172), (1135, 700), (1000, 700))),
        Edge("p3_out", "multi"),
        Edge("p4_out", "multi"),
        Edge("p5_out", "multi"),
        Edge("multi", "decoder"),
    ]

    if v.p4_mdhifi:
        edges.extend(
            [
                Edge("p4_midhifi_or_td", "y4_or_p4m"),
                Edge("y4_or_p4m", "y4"),
                Edge("y4_or_p4m", "a5", color=RED, dashed=True, points=((980, 360), (520, 360), (520, 690))),
            ]
        )
        if v.p3_upsample:
            edges.extend(
                [
                    Edge("y4", "p3_out", points=((1175, 303), (1175, 370), (980, 370))),
                    Edge("y4", "p4_out", points=((1175, 303), (1175, 535), (1000, 535))),
                ]
            )
        else:
            edges.extend(
                [
                    Edge("p3m", "p3_out"),
                    Edge("y4", "p4_out", points=((1175, 303), (1175, 535), (1000, 535))),
                ]
            )
    else:
        edges.extend(
            [
                Edge("p4_midhifi_or_td", "y4_or_p4m"),
                Edge("y4_or_p4m", "p3_out", points=((1090, 303), (1090, 370), (970, 370))),
                Edge("y4_or_p4m", "p4_out", points=((1090, 303), (1090, 535), (1000, 535))),
                Edge("y4_or_p4m", "a5", color=RED, dashed=True, points=((980, 360), (520, 360), (520, 690))),
                Edge("p3m", "p3_out"),
            ]
        )

    if v.p3_upsample and v.p4_mdhifi:
        edges.append(Edge("p3m", "p3_out"))
    return edges


def svg_text(parts: list[str], x: float, y: float, text: str, size: int, fill: str = "#16222d", weight: str = "400") -> None:
    lines = text.split("\n")
    line_h = size + 4
    start = y - (len(lines) - 1) * line_h / 2 + size / 3
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x:.1f}" y="{start + i * line_h:.1f}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(line)}</text>'
        )


def write_svg(v: Detr111Variant) -> Path:
    boxes = build_boxes(v)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1760" height="960" viewBox="0 0 1760 960">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M0,0 L12,6 L0,12 z" fill="#41576b"/>',
        "</marker>",
        "</defs>",
        f'<rect x="0" y="0" width="1760" height="960" fill="{BG}"/>',
        f'<text x="880" y="42" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="26" font-weight="700" fill="#16222d">{escape(v.title)}</text>',
        f'<text x="880" y="68" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#3b4a57">Run: {escape(v.run_name)}</text>',
        f'<text x="880" y="86" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#3b4a57">YAML: {escape(v.yaml_name)}</text>',
        f'<rect x="40" y="115" width="290" height="610" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
        f'<rect x="380" y="115" width="1080" height="740" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
        '<text x="185" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Backbone</text>',
        '<text x="920" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">DETR111 B5-Fix Encoder and Neck</text>',
    ]
    for edge in build_edges(v):
        svg_edge(parts, edge, boxes)
    for box in boxes.values():
        if box.key == "y4" and not v.p4_mdhifi:
            continue
        svg_box(parts, box)
    svg_text(parts, 880, 915, v.note, 15, "#344756")
    parts.append("</svg>")
    path = OUTDIR / v.svg_name
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_drawio() -> Path:
    pages: list[str] = []
    for i, v in enumerate(VARIANTS, start=1):
        page_id = f"detr111_b5fix_{i}"
        cells = [
            f'<mxCell id="{page_id}_0"/>',
            f'<mxCell id="{page_id}_root" parent="{page_id}_0"/>',
            *drawio_title(Variant(v.title, v.page_name, v.svg_name, v.run_name, v.yaml_name, v.note), page_id),
        ]
        cells.extend(drawio_edge(edge, page_id) for edge in build_edges(v))
        for box in build_boxes(v).values():
            if box.key == "y4" and not v.p4_mdhifi:
                continue
            cells.append(drawio_cell_box(box, page_id))
        root = "\n".join(cells)
        pages.append(
            f'<diagram id="{page_id}" name="{v.page_name}">\n'
            '<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
            'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1760" pageHeight="960" math="0" shadow="0">\n'
            f"<root>\n{root}\n</root>\n"
            "</mxGraphModel>\n"
            "</diagram>"
        )
    DRAWIO_PATH.write_text(f'<mxfile host="codex" pages="{len(pages)}">\n' + "\n".join(pages) + "\n</mxfile>\n", encoding="utf-8")
    return DRAWIO_PATH


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(write_drawio())
    for v in VARIANTS:
        print(write_svg(v))


if __name__ == "__main__":
    main()
