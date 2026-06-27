from __future__ import annotations

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
SVG_PATH = OUTDIR / "rtdetr_swinv2_tiny_b2_p3_mdhifi_0621_b4_stepwise_lateral_plain.svg"
DRAWIO_PATH = OUTDIR / "rtdetr_swinv2_tiny_b2_p3_mdhifi_0621_b4_stepwise_lateral_plain.drawio"

VARIANT = Variant(
    title="B2 P3-MDHIFI 0621 B4 Stepwise Lateral Plain",
    page_name="0621-b4-stepwise-lateral-plain",
    svg_name=SVG_PATH.name,
    run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-0621-b4-stepwise-lateral-plain-12k4k4k-e100",
    yaml_name="rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi-0621-b4-stepwise-lateral-plain.yaml",
    note="B4 keeps the B2 P3 top-down path, then injects A4 from P3m into P4 and A5 from Y4 into P5.",
)


def boxes() -> dict[str, Box]:
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
        Box("td_p4", "Upsample Y5\n+ Concat(P4)", 620, 268, 190, 68, TAN),
        Box("y4", "RepC3 -> Y4\n[40 x 40]", 875, 272, 190, 62, PEACH),
        Box("p3_proj", "P3 1x1 Conv\n[80 x 80]", 420, 405, 150, 54, TAN),
        Box("p3m", "P3 MDHIFI\nP3m [80 x 80]", 620, 398, 165, 68, PINK, stroke="#8f2742"),
        Box("p3_out", "Concat Up(Y4)+P3m\nRepC3 -> P3_out [80]", 850, 392, 235, 78, PEACH),
        Box("a4", "A4: Down(P3m)\nConv s=2\n80 -> 40", 620, 565, 190, 70, "#ffe4dc", stroke=RED),
        Box("p4_out", "Concat A4 + Y4\nRepC3 -> P4_out [40]", 875, 560, 220, 74, PEACH),
        Box("a5", "A5: Down(Y4)\nConv s=2\n40 -> 20", 405, 710, 170, 78, "#ffe4dc", stroke=RED),
        Box("p5_down", "Down(P4_out)\nConv s=2\n40 -> 20", 620, 720, 190, 70, TAN),
        Box("p5_out", "Concat Down(P4_out)+Y5+A5\nRepC3 -> P5_out [20]", 875, 713, 240, 78, PEACH),
        Box("multi", "Multi-scale features\nP3 80 / P4 40 / P5 20", 1230, 495, 190, 100, "#f8eee3", stroke="#bba58b"),
        Box("decoder", "RT-DETR decoder", 1500, 485, 170, 120, "#f8b39b", font_size=18),
    ]
    return {box.key: box for box in items}


def edges() -> list[Edge]:
    return [
        Edge("input", "backbone"),
        Edge("backbone", "p3"),
        Edge("p3", "p4"),
        Edge("p4", "p5"),
        Edge("p5", "p5_proj", points=((340, 656), (340, 172))),
        Edge("p4", "p4_proj", points=((350, 526), (350, 302))),
        Edge("p3", "p3_proj"),
        Edge("p5_proj", "aifi"),
        Edge("aifi", "y5"),
        Edge("y5", "td_p4", points=((1120, 172), (1120, 235), (715, 235))),
        Edge("p4_proj", "td_p4"),
        Edge("td_p4", "y4"),
        Edge("y4", "p3_out", points=((1130, 303), (1130, 370), (970, 370))),
        Edge("p3_proj", "p3m"),
        Edge("p3m", "p3_out"),
        Edge("p3m", "a4", color=RED, dashed=True, points=((690, 500), (690, 535))),
        Edge("a4", "p4_out", color=RED, dashed=True),
        Edge("y4", "p4_out", points=((1130, 303), (1130, 535), (1000, 535))),
        Edge("y4", "a5", color=RED, dashed=True, points=((970, 360), (520, 360), (520, 690))),
        Edge("p4_out", "p5_down"),
        Edge("p5_down", "p5_out"),
        Edge("a5", "p5_out", color=RED, dashed=True),
        Edge("y5", "p5_out", points=((1135, 172), (1135, 700), (1000, 700))),
        Edge("p3_out", "multi"),
        Edge("p4_out", "multi"),
        Edge("p5_out", "multi"),
        Edge("multi", "decoder"),
    ]


def svg_text(parts: list[str], x: float, y: float, text: str, size: int, fill: str = "#16222d", weight: str = "400") -> None:
    lines = text.split("\n")
    line_h = size + 4
    start = y - (len(lines) - 1) * line_h / 2 + size / 3
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x:.1f}" y="{start + i * line_h:.1f}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(line)}</text>'
        )


def write_svg() -> Path:
    box_map = boxes()
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1740" height="940" viewBox="0 0 1740 940">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M0,0 L12,6 L0,12 z" fill="#41576b"/>',
        "</marker>",
        "</defs>",
        f'<rect x="0" y="0" width="1740" height="940" fill="{BG}"/>',
        f'<text x="870" y="42" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="26" font-weight="700" fill="#16222d">{escape(VARIANT.title)}</text>',
        f'<text x="870" y="68" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#3b4a57">Run: {escape(VARIANT.run_name)}</text>',
        f'<text x="870" y="86" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#3b4a57">YAML: {escape(VARIANT.yaml_name)}</text>',
        f'<rect x="40" y="115" width="290" height="610" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
        f'<rect x="380" y="115" width="1060" height="720" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
        '<text x="185" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Backbone</text>',
        '<text x="910" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Hybrid Encoder and Stepwise Lateral Neck</text>',
    ]
    for edge in edges():
        svg_edge(parts, edge, box_map)
    for box in box_map.values():
        svg_box(parts, box)
    svg_text(parts, 870, 895, VARIANT.note, 15, "#344756")
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")
    return SVG_PATH


def write_drawio() -> Path:
    page_id = "b4_stepwise_lateral_plain"
    cells = [
        f'<mxCell id="{page_id}_0"/>',
        f'<mxCell id="{page_id}_root" parent="{page_id}_0"/>',
        *drawio_title(VARIANT, page_id),
    ]
    cells.extend(drawio_edge(edge, page_id) for edge in edges())
    cells.extend(drawio_cell_box(box, page_id) for box in boxes().values())
    root = "\n".join(cells)
    content = (
        '<mxfile host="codex" pages="1">\n'
        f'<diagram id="{page_id}" name="{VARIANT.page_name}">\n'
        '<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
        'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1740" pageHeight="940" math="0" shadow="0">\n'
        f"<root>\n{root}\n</root>\n"
        "</mxGraphModel>\n"
        "</diagram>\n"
        "</mxfile>\n"
    )
    DRAWIO_PATH.write_text(content, encoding="utf-8")
    return DRAWIO_PATH


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(write_drawio())
    print(write_svg())


if __name__ == "__main__":
    main()
