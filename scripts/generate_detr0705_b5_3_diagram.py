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
    GREEN,
    ORANGE,
    PANEL,
    PEACH,
    PINK,
    PURPLE,
    TAN,
    Box,
    Edge,
    drawio_cell_box,
    drawio_edge,
    svg_box,
    svg_edge,
)


OUTDIR = Path("assets/detr0705")
DRAWIO_PATH = OUTDIR / "b5-3_principle.drawio"
SVG_PATH = OUTDIR / "b5-3_principle.svg"
WIDTH = 2060
HEIGHT = 1040
TITLE = "DETR0705 B5-3 Principle Diagram"
RUN_NAME = "runs/detect/0705/b5-3"
YAML_NAME = "exp_cfg/0705/b5.yaml"
NOTE = (
    "P5 detection head is removed. P5 is kept as the AIFI semantic source and as an HFSCC "
    "calibration source, while the RT-DETR decoder consumes only P2_out, P3_out, and P4_out."
)


def build_boxes() -> dict[str, Box]:
    items = [
        Box("input", "Input image\n640 x 640", 70, 160, 220, 52, ORANGE),
        Box("backbone", "TorchVision SwinV2-Tiny\nsplit stage outputs", 70, 245, 220, 72, "#9ec8c0"),
        Box("p2", "P2/4 feature [160 x 160]\nIndex + Permute + EMA", 70, 360, 220, 72, "#cfe8d8"),
        Box("p3", "P3/8 feature [80 x 80]\nIndex + Permute + EMA", 70, 480, 220, 72, BLUE),
        Box("p4", "P4/16 feature [40 x 40]\nIndex + Permute + EMA", 70, 600, 220, 72, BLUE2),
        Box("p5", "P5/32 feature [20 x 20]\nIndex + Permute + EMA", 70, 720, 220, 72, BLUE3),
        Box("p5_proj", "P5 1x1 Conv\n[20 x 20]", 420, 160, 150, 54, TAN),
        Box("aifi", "AIFI encoder\n[20 x 20]", 620, 160, 170, 54, PURPLE),
        Box("y5", "Y5\n[20 x 20]", 850, 160, 155, 54, PEACH),
        Box("p4_proj", "P4 1x1 Conv\n[40 x 40]", 420, 315, 150, 54, TAN),
        Box("p4m", "P4 MDHIFI\nP4m [40 x 40]", 620, 306, 185, 72, PINK, stroke="#8f2742"),
        Box("p3_proj", "P3 1x1 Conv\n[80 x 80]", 420, 470, 150, 54, TAN),
        Box("p3m", "P3 MDHIFI\nP3m [80 x 80]", 620, 461, 185, 72, PINK, stroke="#8f2742"),
        Box("p2_proj", "P2 1x1 Conv\n[160 x 160]", 420, 625, 150, 54, TAN),
        Box("hfscc", "HFSCC\nP3/P4/P5 shift align\ncross-scale calibration", 900, 290, 255, 200, GREEN, stroke="#587d3b"),
        Box("p3c", "P3c\n[80 x 80]", 1235, 470, 130, 58, PEACH),
        Box("p4c", "P4c\n[40 x 40]", 1235, 360, 130, 58, PEACH),
        Box("p5c", "P5c\n[20 x 20]", 1235, 250, 130, 58, PEACH),
        Box("up_p5c", "Up(P5c)\n20 -> 40", 1435, 250, 145, 54, TAN),
        Box("y4", "Concat Up(P5c)+P4c\nRepC3 -> Y4 [40]", 1630, 235, 255, 78, PEACH),
        Box("y4_proj", "Y4 proj\n1x1 Conv", 1630, 350, 185, 58, TAN),
        Box("up_y4", "Up(Y4 proj)\n40 -> 80", 1435, 470, 145, 54, TAN),
        Box("y3", "Concat Up(Y4)+P3c\nRepC3 -> Y3 [80]", 1630, 455, 255, 78, PEACH),
        Box("y3_proj", "Y3 proj\n1x1 Conv", 1630, 570, 185, 58, TAN),
        Box("up_y3", "Up(Y3 proj)\n80 -> 160", 1435, 655, 155, 54, TAN),
        Box("p2_out", "Concat Up(Y3)+P2 proj\nRepC3 -> P2_out [160]", 1630, 640, 270, 78, PEACH),
        Box("down_p2", "Down(P2_out)\nConv s=2\n160 -> 80", 1435, 775, 175, 70, TAN),
        Box("p3_out", "Concat Down(P2)+Y3 proj+P3c\nRepC3 -> P3_out [80]", 1630, 765, 310, 78, PEACH),
        Box("down_p3", "Down(P3_out)\nConv s=2\n80 -> 40", 1435, 895, 175, 70, TAN),
        Box("p4_out", "Concat Down(P3)+Y4 proj+P4c\nRepC3 -> P4_out [40]", 1630, 885, 310, 78, PEACH),
        Box("multi", "Multi-scale features\nP2/P3/P4 only", 1215, 890, 190, 96, "#f8eee3", stroke="#bba58b"),
        Box("decoder", "RT-DETR decoder", 1000, 888, 175, 96, "#f8b39b", font_size=18),
        Box("removed_p5", "Removed P5 head\nno P5_out / no decoder input", 1630, 120, 240, 70, "#f9e3e0", stroke="#b94747"),
    ]
    return {box.key: box for box in items}


def build_edges() -> list[Edge]:
    return [
        Edge("input", "backbone"),
        Edge("backbone", "p2"),
        Edge("p2", "p3"),
        Edge("p3", "p4"),
        Edge("p4", "p5"),
        Edge("p5", "p5_proj", points=((345, 749), (345, 187))),
        Edge("p5_proj", "aifi"),
        Edge("aifi", "y5"),
        Edge("p4", "p4_proj", points=((350, 636), (350, 342))),
        Edge("p4_proj", "p4m"),
        Edge("p3", "p3_proj", points=((350, 516), (350, 497))),
        Edge("p3_proj", "p3m"),
        Edge("p2", "p2_proj", points=((345, 396), (345, 652))),
        Edge("y5", "hfscc", points=((1030, 187), (1030, 240), (970, 240))),
        Edge("p4m", "hfscc", points=((805, 342), (860, 342), (860, 345), (900, 345))),
        Edge("p3m", "hfscc", points=((805, 497), (860, 497), (860, 440), (900, 440))),
        Edge("hfscc", "p5c"),
        Edge("hfscc", "p4c"),
        Edge("hfscc", "p3c"),
        Edge("p5c", "up_p5c"),
        Edge("up_p5c", "y4"),
        Edge("p4c", "y4", points=((1365, 389), (1505, 389), (1505, 274), (1630, 274))),
        Edge("y4", "y4_proj"),
        Edge("y4_proj", "up_y4"),
        Edge("up_y4", "y3"),
        Edge("p3c", "y3", points=((1365, 499), (1500, 499), (1500, 494), (1630, 494))),
        Edge("y3", "y3_proj"),
        Edge("y3_proj", "up_y3"),
        Edge("up_y3", "p2_out"),
        Edge("p2_proj", "p2_out", points=((570, 652), (1385, 652), (1385, 679), (1630, 679))),
        Edge("p2_out", "down_p2"),
        Edge("down_p2", "p3_out"),
        Edge("y3_proj", "p3_out", points=((1815, 599), (1815, 748), (1775, 748))),
        Edge("p3c", "p3_out", points=((1365, 499), (1395, 499), (1395, 804), (1630, 804))),
        Edge("p3_out", "down_p3"),
        Edge("down_p3", "p4_out"),
        Edge("y4_proj", "p4_out", points=((1815, 379), (1815, 870), (1775, 870))),
        Edge("p4c", "p4_out", points=((1365, 389), (1395, 389), (1395, 924), (1630, 924))),
        Edge("p2_out", "multi"),
        Edge("p3_out", "multi"),
        Edge("p4_out", "multi"),
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
    boxes = build_boxes()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M0,0 L12,6 L0,12 z" fill="#41576b"/>',
        "</marker>",
        "</defs>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>',
        f'<text x="{WIDTH / 2:.1f}" y="42" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="26" font-weight="700" fill="#16222d">{escape(TITLE)}</text>',
        f'<text x="{WIDTH / 2:.1f}" y="68" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#3b4a57">Run: {escape(RUN_NAME)}</text>',
        f'<text x="{WIDTH / 2:.1f}" y="86" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#3b4a57">YAML: {escape(YAML_NAME)}</text>',
        f'<rect x="40" y="125" width="290" height="700" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
        f'<rect x="380" y="125" width="1590" height="860" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
        '<text x="185" y="148" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Backbone</text>',
        '<text x="1175" y="148" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">DETR0705 B5 Encoder and Neck</text>',
    ]
    for edge in build_edges():
        svg_edge(parts, edge, boxes)
    for box in boxes.values():
        svg_box(parts, box)
    svg_text(parts, WIDTH / 2, HEIGHT - 26, NOTE, 15, "#344756")
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")
    return SVG_PATH


def escape_label(label: str) -> str:
    return escape(label).replace("\n", "&#xa;")


def drawio_title(page_id: str) -> list[str]:
    subtitle = f"Run: {RUN_NAME}&#xa;YAML: {YAML_NAME}"
    return [
        f'<mxCell id="{page_id}_title" value="{escape_label(TITLE)}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#16222d;fontSize=26;fontStyle=1;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="40" y="18" width="{WIDTH - 80}" height="34" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_subtitle" value="{subtitle}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#3b4a57;fontSize=13;fontStyle=0;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="80" y="56" width="{WIDTH - 160}" height="42" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_backbone_panel" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="40" y="125" width="290" height="700" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_neck_panel" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="380" y="125" width="1590" height="860" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_note" value="{escape_label(NOTE)}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#344756;fontSize=15;fontStyle=0;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="240" y="{HEIGHT - 44}" width="{WIDTH - 480}" height="32" as="geometry"/>\n'
        "</mxCell>",
    ]


def write_drawio() -> Path:
    page_id = "detr0705_b5_3"
    cells = [
        f'<mxCell id="{page_id}_0"/>',
        f'<mxCell id="{page_id}_root" parent="{page_id}_0"/>',
        *drawio_title(page_id),
    ]
    cells.extend(drawio_edge(edge, page_id) for edge in build_edges())
    cells.extend(drawio_cell_box(box, page_id) for box in build_boxes().values())
    root = "\n".join(cells)
    content = (
        '<mxfile host="codex" pages="1">\n'
        f'<diagram id="{page_id}" name="b5-3-principle">\n'
        '<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
        f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{WIDTH}" pageHeight="{HEIGHT}" math="0" shadow="0">\n'
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
