from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_detr111_b5fix_diagrams import (
    OUTDIR,
    Detr111Variant,
    drawio_cell_box,
    drawio_edge,
    drawio_title,
    svg_box,
    svg_edge,
    BG,
    BLUE,
    BLUE2,
    BLUE3,
    PANEL,
    ORANGE,
    PEACH,
    PINK,
    PURPLE,
    RED,
    TAN,
    Box,
    Edge,
    Variant,
)


DRAWIO_PATH = OUTDIR / "rtdetr_swinv2_tiny_detr111_b5fix_upbefore_principles.drawio"

VARIANTS = [
    Detr111Variant(
        title="DETR111-05 B5 Stepwise Lateral P4-MDHIFI UpBefore",
        page_name="detr111-05-b5-stepwise-lateral-p4mdhifi-upbefore",
        svg_name="rtdetr_swinv2_tiny_detr111_05_b5_stepwise_lateral_p4mdhifi_upbefore.svg",
        run_name="rtdetr-swinv2-tiny-detr111-05-b5-stepwise-lateral-p4mdhifi-upbefore-12k4k4k-e100",
        yaml_name="exp_cfg/detr111/05-b5-stepwise-lateral-p4mdhifi.yaml",
        note="UpBefore run for DETR111-05. Topology matches YAML 05: P4 uses MDHIFI, P3 uses Up(Y4)+P3m, and A5 comes from P4m.",
        p4_mdhifi=True,
        p3_upsample=True,
    ),
    Detr111Variant(
        title="DETR111-06 B5 No P3 Upsample UpBefore",
        page_name="detr111-06-b5-no-p3-upsample-upbefore",
        svg_name="rtdetr_swinv2_tiny_detr111_06_b5_no_p3_upsample_upbefore.svg",
        run_name="rtdetr-swinv2-tiny-detr111-06-b5-no-p3-upsample-upbefore-12k4k4k-e100",
        yaml_name="exp_cfg/detr111/06-b5-stepwise-lateral-p4mdhifi-no-p3-upsample.yaml",
        note="UpBefore run for DETR111-06. Topology matches YAML 06: P3_out comes directly from P3m, while P4-MDHIFI and A5 from P4m stay enabled.",
        p4_mdhifi=True,
        p3_upsample=False,
    ),
]


def build_boxes(v: Detr111Variant) -> dict[str, Box]:
    p3_label = "Concat Up(Y4)+P3m\nRepC3 -> P3_out [80]" if v.p3_upsample else "P3m -> RepC3\nP3_out [80]"
    p3_w = 240 if v.p3_upsample else 190

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
        Box("p4m", "P4 MDHIFI\nP4m [40 x 40]", 620, 268, 180, 68, PINK, stroke="#8f2742"),
        Box("up_p4m", "Up(P4m)\n40 -> 20", 855, 275, 145, 54, TAN),
        Box("p4_td", "Concat Y5 + Up(P4m)\nRepC3 -> top-down P4", 1070, 262, 230, 78, PEACH),
        Box("y4", "Y4\n[40 x 40]", 1360, 272, 120, 62, PEACH),
        Box("up_y4", "Up(Y4)\n40 -> 80", 1175, 388, 145, 54, TAN) if v.p3_upsample else None,
        Box("p3_proj", "P3 1x1 Conv\n[80 x 80]", 420, 405, 150, 54, TAN),
        Box("p3m", "P3 MDHIFI\nP3m [80 x 80]", 620, 398, 165, 68, PINK, stroke="#8f2742"),
        Box("p3_out", p3_label, 850, 392, p3_w, 78 if v.p3_upsample else 68, PEACH),
        Box("a4", "A4: Down(P3m)\nConv s=2\n80 -> 40", 620, 565, 190, 70, "#ffe4dc", stroke=RED),
        Box("p4_out", "Concat A4 + Y4\nRepC3 -> P4_out [40]", 1030, 560, 220, 74, PEACH),
        Box("a5", "A5: Down(P4m)\nConv s=2\n40 -> 20", 405, 710, 170, 78, "#ffe4dc", stroke=RED),
        Box("p5_down", "Down(P4_out)\nConv s=2\n40 -> 20", 700, 720, 190, 70, TAN),
        Box("p5_out", "Concat Down(P4_out)+Y5+A5\nRepC3 -> P5_out [20]", 1030, 713, 250, 78, PEACH),
        Box("multi", "Multi-scale features\nP3 80 / P4 40 / P5 20", 1335, 495, 190, 100, "#f8eee3", stroke="#bba58b"),
        Box("decoder", "RT-DETR decoder", 1580, 485, 140, 120, "#f8b39b", font_size=18),
    ]
    return {box.key: box for box in items if box is not None}


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
        Edge("p4_proj", "p4m"),
        Edge("p4m", "up_p4m"),
        Edge("y5", "p4_td", points=((1010, 172), (1010, 235), (1185, 235))),
        Edge("up_p4m", "p4_td"),
        Edge("p4_td", "y4"),
        Edge("p3_proj", "p3m"),
        Edge("p3m", "a4", color=RED, dashed=True, points=((702, 500), (702, 535))),
        Edge("a4", "p4_out", color=RED, dashed=True),
        Edge("y4", "p4_out", points=((1420, 303), (1420, 535), (1140, 535))),
        Edge("p4_out", "p5_down"),
        Edge("p5_down", "p5_out"),
        Edge("a5", "p5_out", color=RED, dashed=True),
        Edge("y5", "p5_out", points=((930, 172), (930, 700), (1155, 700))),
        Edge("p4m", "a5", color=RED, dashed=True, points=((710, 360), (490, 360), (490, 690))),
        Edge("p3_out", "multi"),
        Edge("p4_out", "multi"),
        Edge("p5_out", "multi"),
        Edge("multi", "decoder"),
    ]

    if v.p3_upsample:
        edges.extend(
            [
                Edge("y4", "up_y4", points=((1420, 303), (1420, 415), (1250, 415))),
                Edge("up_y4", "p3_out"),
                Edge("p3m", "p3_out"),
            ]
        )
    else:
        edges.append(Edge("p3m", "p3_out"))

    return edges


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
        f'<text x="880" y="42" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="26" font-weight="700" fill="#16222d">{v.title}</text>',
        f'<text x="880" y="68" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#3b4a57">Run: {v.run_name}</text>',
        f'<text x="880" y="86" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#3b4a57">YAML: {v.yaml_name}</text>',
        f'<rect x="40" y="115" width="290" height="610" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
        f'<rect x="380" y="115" width="1080" height="740" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
        '<text x="185" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Backbone</text>',
        '<text x="920" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">DETR111 B5-Fix UpBefore Encoder and Neck</text>',
    ]
    for edge in build_edges(v):
        svg_edge(parts, edge, boxes)
    for box in boxes.values():
        if box.key == "y4" and not v.p4_mdhifi:
            continue
        svg_box(parts, box)
    parts.append(
        f'<text x="880" y="915" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#344756">{v.note}</text>'
    )
    parts.append("</svg>")
    path = OUTDIR / v.svg_name
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_drawio() -> Path:
    pages: list[str] = []
    for i, v in enumerate(VARIANTS, start=1):
        page_id = f"detr111_b5fix_upbefore_{i}"
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
