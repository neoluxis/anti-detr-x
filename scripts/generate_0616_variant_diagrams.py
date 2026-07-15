from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import sys
from textwrap import wrap

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
    RED,
    TAN,
    Box,
    Edge,
    drawio_cell_box,
    drawio_edge,
    svg_box,
    svg_edge,
)


OUTDIR = Path("assets/figures/0616")
COMBINED_DRAWIO_PATH = OUTDIR / "rtdetr_swinv2_tiny_0616_variants.drawio"
WIDTH = 1880
HEIGHT = 940


@dataclass(frozen=True)
class Variant0616:
    code: str
    title: str
    page_name: str
    drawio_name: str
    svg_name: str
    run_name: str
    yaml_name: str
    note: str
    p5_module: str | None = None
    p3_module: str | None = None


VARIANTS = [
    Variant0616(
        code="b0",
        title="0616 B0 EMA Baseline",
        page_name="0616-b0-ema-baseline",
        drawio_name="0616_b0_ema_baseline.drawio",
        svg_name="0616_b0_ema_baseline.svg",
        run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-12k4k4k-e100-2",
        yaml_name="exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema.yaml",
        note="Baseline neck: AIFI on P5, plain top-down FPN, then standard bottom-up PAN.",
    ),
    Variant0616(
        code="b1",
        title="0616 B1 AIFI + MDHIFI",
        page_name="0616-b1-aifi-mdhifi",
        drawio_name="0616_b1_aifi_mdhifi.drawio",
        svg_name="0616_b1_aifi_mdhifi.svg",
        run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b1-aifi-mdhifi-12k4k4k-e100",
        yaml_name="exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b1-aifi-mdhifi.yaml",
        note="B1 inserts MDHIFI after the P5 AIFI encoder, then projects to Y5 for the rest of the neck.",
        p5_module="MDHIFI",
    ),
    Variant0616(
        code="b2",
        title="0616 B2 P3 MDHIFI",
        page_name="0616-b2-p3-mdhifi",
        drawio_name="0616_b2_p3_mdhifi.drawio",
        svg_name="0616_b2_p3_mdhifi.svg",
        run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-12k4k4k-e100-3",
        yaml_name="exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi.yaml",
        note="B2 inserts MDHIFI after the P3 lateral 1x1 Conv before top-down P3 fusion.",
        p3_module="MDHIFI",
    ),
    Variant0616(
        code="b3a",
        title="0616 B3 P3 AIFI",
        page_name="0616-b3-p3-aifi",
        drawio_name="0616_b3_p3_aifi.drawio",
        svg_name="0616_b3_p3_aifi.svg",
        run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b3-p3-aifi-12k4k4k-e100",
        yaml_name="exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b3-p3-aifi.yaml",
        note="This B3 sibling swaps the P3 insertion block to AIFI before Concat Up(Y4)+P3.",
        p3_module="AIFI",
    ),
    Variant0616(
        code="b3h",
        title="0616 B3 P3 HIFI",
        page_name="0616-b3-p3-hifi",
        drawio_name="0616_b3_p3_hifi.drawio",
        svg_name="0616_b3_p3_hifi.svg",
        run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b3-p3-hifi-12k4k4k-e100",
        yaml_name="exp_cfg/detr/rtdetr-swinv2-tiny-last3-pretrained-ema-b3-p3-hifi.yaml",
        note="Document-style B3 compares HIFI at the same P3 lateral insertion point used by B2.",
        p3_module="HIFI",
    ),
]


def subtitle_lines(v: Variant0616) -> list[str]:
    return wrap(f"Run: {v.run_name} | YAML: {v.yaml_name}", width=160)


def module_box(v: Variant0616) -> Box | None:
    if v.p3_module is None:
        return None
    if v.p3_module == "MDHIFI":
        label = "P3 MDHIFI\nP3m [80 x 80]"
        fill = PINK
        stroke = "#8f2742"
    elif v.p3_module == "AIFI":
        label = "P3 AIFI\nP3a [80 x 80]"
        fill = PURPLE
        stroke = "#5a4d9c"
    elif v.p3_module == "HIFI":
        label = "P3 HIFI\nP3h [80 x 80]"
        fill = GREEN
        stroke = "#587d3b"
    else:
        raise ValueError(v.p3_module)
    return Box("p3_special", label, 620, 398, 175, 68, fill, stroke=stroke)


def p5_module_box(v: Variant0616) -> Box | None:
    if v.p5_module is None:
        return None
    if v.p5_module == "MDHIFI":
        return Box("p5_special", "P5 MDHIFI\nP5m [20 x 20]", 835, 138, 170, 68, PINK, stroke="#8f2742")
    raise ValueError(v.p5_module)


def build_boxes(v: Variant0616) -> dict[str, Box]:
    p3_ref = {
        None: "P3",
        "MDHIFI": "P3m",
        "AIFI": "P3a",
        "HIFI": "P3h",
    }[v.p3_module]
    y5_x = 1060 if v.p5_module else 850
    td_x = 1240 if v.p5_module else 1070
    y4_x = 1520 if v.p5_module else 1360

    items = [
        Box("input", "Input image\n640 x 640", 70, 145, 220, 52, ORANGE),
        Box("backbone", "TorchVision SwinV2-Tiny\nStage2-4 outputs", 70, 230, 220, 72, "#9ec8c0"),
        Box("p3", "P3/8 feature [80 x 80]\nIndex + Permute + EMA", 70, 360, 220, 72, BLUE),
        Box("p4", "P4/16 feature [40 x 40]\nIndex + Permute + EMA", 70, 490, 220, 72, BLUE2),
        Box("p5", "P5/32 feature [20 x 20]\nIndex + Permute + EMA", 70, 620, 220, 72, BLUE3),
        Box("p5_proj", "P5 1x1 Conv\n[20 x 20]", 420, 145, 150, 54, TAN),
        Box("aifi", "AIFI encoder\n[20 x 20]", 620, 145, 165, 54, PURPLE),
        p5_module_box(v),
        Box("y5", "Y5\n[20 x 20]", y5_x, 145, 150, 54, PEACH),
        Box("p4_proj", "P4 1x1 Conv\n[40 x 40]", 420, 275, 150, 54, TAN),
        Box("td_p4", "Concat Up(Y5)+P4\nRepC3 -> top-down P4", td_x, 262, 230, 78, PEACH),
        Box("y4", "Y4\n[40 x 40]", y4_x, 272, 120, 62, PEACH),
        Box("p3_proj", "P3 1x1 Conv\n[80 x 80]", 420, 405, 150, 54, TAN),
        module_box(v),
        Box("p3_out", f"Concat Up(Y4)+{p3_ref}\nRepC3 -> P3_out [80]", 850, 392, 245, 78, PEACH),
        Box("p4_down", "Down(P3_out)\nConv s=2\n80 -> 40", 620, 565, 190, 70, TAN),
        Box("p4_out", "Concat Down(P3_out)+Y4\nRepC3 -> P4_out [40]", 1030, 558, 240, 76, PEACH),
        Box("p5_down", "Down(P4_out)\nConv s=2\n40 -> 20", 620, 720, 190, 70, TAN),
        Box("p5_out", "Concat Down(P4_out)+Y5\nRepC3 -> P5_out [20]", 1030, 713, 240, 76, PEACH),
        Box("multi", "Multi-scale features\nP3 80 / P4 40 / P5 20", 1390, 495, 190, 100, "#f8eee3", stroke="#bba58b"),
        Box("decoder", "RT-DETR decoder", 1660, 485, 170, 120, "#f8b39b", font_size=18),
    ]
    return {box.key: box for box in items if box is not None}


def build_edges(v: Variant0616) -> list[Edge]:
    boxes = build_boxes(v)
    y5_box = boxes["y5"]
    td_box = boxes["td_p4"]
    y4_box = boxes["y4"]
    p3_out_box = boxes["p3_out"]
    p4_out_box = boxes["p4_out"]
    p5_out_box = boxes["p5_out"]
    y5_route = (
        (y5_box.x + y5_box.w + 40, y5_box.y + y5_box.h / 2),
        (y5_box.x + y5_box.w + 40, td_box.y - 25),
        (td_box.x + 80, td_box.y - 25),
    )
    y4_p3_route = (
        (y4_box.x + 60, y4_box.y + y4_box.h + 35),
        (p3_out_box.x + 165, y4_box.y + y4_box.h + 35),
    )
    y4_p4_route = (
        (y4_box.x + 60, p4_out_box.y - 25),
        (p4_out_box.x + 120, p4_out_box.y - 25),
    )
    y5_p5_route = (
        (y5_box.x + y5_box.w + 55, y5_box.y + y5_box.h / 2),
        (y5_box.x + y5_box.w + 55, p5_out_box.y - 12),
        (p5_out_box.x + 110, p5_out_box.y - 12),
    )

    edges = [
        Edge("input", "backbone"),
        Edge("backbone", "p3"),
        Edge("p3", "p4"),
        Edge("p4", "p5"),
        Edge("p5", "p5_proj", points=((340, 656), (340, 172))),
        Edge("p4", "p4_proj", points=((350, 526), (350, 302))),
        Edge("p3", "p3_proj"),
        Edge("p5_proj", "aifi"),
        Edge("y5", "td_p4", points=y5_route),
        Edge("p4_proj", "td_p4"),
        Edge("td_p4", "y4"),
        Edge("y4", "p3_out", points=y4_p3_route),
        Edge("p4_down", "p4_out"),
        Edge("y4", "p4_out", points=y4_p4_route),
        Edge("p3_out", "p4_down"),
        Edge("p4_out", "p5_down"),
        Edge("p5_down", "p5_out"),
        Edge("y5", "p5_out", points=y5_p5_route),
        Edge("p3_out", "multi"),
        Edge("p4_out", "multi"),
        Edge("p5_out", "multi"),
        Edge("multi", "decoder"),
    ]

    if v.p5_module is None:
        edges.insert(8, Edge("aifi", "y5"))
    else:
        edges.insert(8, Edge("aifi", "p5_special"))
        edges.insert(9, Edge("p5_special", "y5"))

    if v.p3_module is None:
        edges.insert(13, Edge("p3_proj", "p3_out"))
    else:
        edges.insert(13, Edge("p3_proj", "p3_special"))
        edges.insert(14, Edge("p3_special", "p3_out"))

    return edges


def title_cells(v: Variant0616, page_id: str) -> list[str]:
    subtitle = "\n".join(subtitle_lines(v))
    return [
        f'<mxCell id="{page_id}_title" value="{escape(v.title)}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#16222d;fontSize=26;fontStyle=1;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="40" y="18" width="{WIDTH - 80}" height="34" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_subtitle" value="{escape(subtitle).replace(chr(10), "&#xa;")}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#3b4a57;fontSize=13;fontStyle=0;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="90" y="56" width="{WIDTH - 180}" height="42" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_backbone_panel" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="40" y="115" width="290" height="610" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_neck_panel" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="380" y="115" width="1450" height="720" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_note" value="{escape(v.note)}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#344756;fontSize=15;fontStyle=0;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="250" y="880" width="{WIDTH - 500}" height="28" as="geometry"/>\n'
        "</mxCell>",
    ]


def drawio_page(v: Variant0616, idx: int) -> str:
    page_id = f"v0616_{idx}"
    cells = [
        f'<mxCell id="{page_id}_0"/>',
        f'<mxCell id="{page_id}_root" parent="{page_id}_0"/>',
        *title_cells(v, page_id),
    ]
    edges = build_edges(v)
    boxes = build_boxes(v)
    cells.extend(drawio_edge(edge, page_id) for edge in edges)
    cells.extend(drawio_cell_box(box, page_id) for box in boxes.values())
    root = "\n".join(cells)
    return (
        f'<diagram id="{page_id}" name="{escape(v.page_name)}">\n'
        f'<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{WIDTH}" pageHeight="{HEIGHT}" math="0" shadow="0">\n'
        f"<root>\n{root}\n</root>\n"
        "</mxGraphModel>\n"
        "</diagram>"
    )


def write_combined_drawio() -> Path:
    pages = "\n".join(drawio_page(v, i) for i, v in enumerate(VARIANTS, start=1))
    COMBINED_DRAWIO_PATH.write_text(f'<mxfile host="codex" pages="{len(VARIANTS)}">\n{pages}\n</mxfile>\n', encoding="utf-8")
    return COMBINED_DRAWIO_PATH


def write_single_drawio(v: Variant0616, idx: int) -> Path:
    path = OUTDIR / v.drawio_name
    path.write_text(f'<mxfile host="codex" pages="1">\n{drawio_page(v, idx)}\n</mxfile>\n', encoding="utf-8")
    return path


def write_svg(v: Variant0616) -> Path:
    boxes = build_boxes(v)
    edges = build_edges(v)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M0,0 L12,6 L0,12 z" fill="#41576b"/>',
        "</marker>",
        "</defs>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>',
        f'<text x="{WIDTH / 2:.0f}" y="42" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="26" font-weight="700" fill="#16222d">{escape(v.title)}</text>',
    ]
    for i, line in enumerate(subtitle_lines(v)):
        parts.append(
            f'<text x="{WIDTH / 2:.0f}" y="{68 + i * 18}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#3b4a57">{escape(line)}</text>'
        )
    parts.extend(
        [
            f'<rect x="40" y="115" width="290" height="610" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
            f'<rect x="380" y="115" width="1450" height="720" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
            '<text x="185" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Backbone</text>',
            '<text x="1105" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Hybrid Encoder and Neck</text>',
        ]
    )
    for edge in edges:
        svg_edge(parts, edge, boxes)
    for box in boxes.values():
        svg_box(parts, box)
    parts.append(
        f'<text x="{WIDTH / 2:.0f}" y="895" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#344756">{escape(v.note)}</text>'
    )
    parts.append("</svg>")
    path = OUTDIR / v.svg_name
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(write_combined_drawio())
    for idx, variant in enumerate(VARIANTS, start=1):
        print(write_single_drawio(variant, idx))
        print(write_svg(variant))


if __name__ == "__main__":
    main()
