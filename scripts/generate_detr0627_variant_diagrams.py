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
    drawio_cell_box,
    drawio_edge,
    svg_box,
    svg_edge,
)


OUTDIR = Path("assets/figures/detr0627")
COMBINED_DRAWIO_PATH = OUTDIR / "detr0627_variants.drawio"
WIDTH = 1900
HEIGHT = 960


@dataclass(frozen=True)
class Variant0627:
    slug: str
    title: str
    run_name: str
    yaml_name: str
    note: str
    p4_mdhifi: bool
    p3_upsample: bool
    a4_used: bool
    a5_from_p4m: bool

    @property
    def drawio_name(self) -> str:
        return f"{self.slug}_principle.drawio"

    @property
    def svg_name(self) -> str:
        return f"{self.slug}_principle.svg"


VARIANTS = [
    Variant0627(
        slug="test00001",
        title="DETR0627 Test00001 Principle",
        run_name="runs/detect/detr0627/test00001",
        yaml_name="exp_cfg/detr0627/test00001.yaml",
        note="P4 lateral and P3 lateral both use MDHIFI. P4m joins Up(Y5) before top-down P4 fusion; A4 from P3m and A5 from P4m feed the two bottom-up fusion stages.",
        p4_mdhifi=True,
        p3_upsample=True,
        a4_used=True,
        a5_from_p4m=True,
    ),
    Variant0627(
        slug="test00002",
        title="DETR0627 Test00002 Principle",
        run_name="runs/detect/detr0627/test00002",
        yaml_name="exp_cfg/detr0627/test00002.yaml",
        note="This variant keeps P4-MDHIFI and P3-MDHIFI, but removes Up(Y4) from the P3 fusion stage so P3_out is produced directly from P3m.",
        p4_mdhifi=True,
        p3_upsample=False,
        a4_used=True,
        a5_from_p4m=True,
    ),
    Variant0627(
        slug="test00003",
        title="DETR0627 Test00003 Principle",
        run_name="runs/detect/detr0627/test00003",
        yaml_name="exp_cfg/detr0627/test00003.yaml",
        note="P4 branch falls back to plain P4 lateral fusion and A5 comes from the plain P4 projection; P3 still uses MDHIFI and bypasses Up(Y4).",
        p4_mdhifi=False,
        p3_upsample=False,
        a4_used=True,
        a5_from_p4m=False,
    ),
    Variant0627(
        slug="test00004",
        title="DETR0627 Test00004 Principle",
        run_name="runs/detect/detr0627/test00004",
        yaml_name="exp_cfg/detr0627/test00004.yaml",
        note="Literal YAML topology: P4 uses plain lateral fusion, P3 uses Up(Y4)+P3m, A5 comes from the plain P4 projection, and the A4 branch is generated from P3m but not consumed by P4_out because the P4 concat is single-input.",
        p4_mdhifi=False,
        p3_upsample=True,
        a4_used=False,
        a5_from_p4m=False,
    ),
]


def subtitle_lines(v: Variant0627) -> list[str]:
    return [f"Run: {v.run_name}", f"YAML: {v.yaml_name}"]


def build_boxes(v: Variant0627) -> dict[str, Box]:
    p4_td_label = "Concat Up(Y5)+P4m\nRepC3 -> top-down P4" if v.p4_mdhifi else "Concat Up(Y5)+P4\nRepC3 -> top-down P4"
    p3_out_label = "Concat Up(Y4)+P3m\nRepC3 -> P3_out [80]" if v.p3_upsample else "P3m -> RepC3\nP3_out [80]"
    p4_out_label = "Concat A4 + Y4\nRepC3 -> P4_out [40]" if v.a4_used else "Y4 only via single-input Concat\nRepC3 -> P4_out [40]"
    a4_label = "A4: Down(P3m)\nConv s=2\n80 -> 40" if v.a4_used else "A4: Down(P3m)\nGenerated but unused\nin literal YAML"
    a5_label = "A5: Down(P4m)\nConv s=2\n40 -> 20" if v.a5_from_p4m else "A5: Down(P4 proj)\nConv s=2\n40 -> 20"

    items = [
        Box("input", "Input image\n640 x 640", 70, 145, 220, 52, ORANGE),
        Box("backbone", "TorchVision SwinV2-Tiny\nStage2-4 outputs", 70, 230, 220, 72, "#9ec8c0"),
        Box("p3", "P3/8 feature [80 x 80]\nIndex + Permute + EMA", 70, 360, 220, 72, BLUE),
        Box("p4", "P4/16 feature [40 x 40]\nIndex + Permute + EMA", 70, 490, 220, 72, BLUE2),
        Box("p5", "P5/32 feature [20 x 20]\nIndex + Permute + EMA", 70, 620, 220, 72, BLUE3),
        Box("p5_proj", "P5 1x1 Conv\n[20 x 20]", 420, 145, 150, 54, TAN),
        Box("aifi", "AIFI encoder\n[20 x 20]", 620, 145, 165, 54, PURPLE),
        Box("y5", "Y5\n[20 x 20]", 850, 145, 155, 54, PEACH),
        Box("up_y5", "Up(Y5)\n20 -> 40", 1060, 145, 145, 54, TAN),
        Box("p4_proj", "P4 1x1 Conv\n[40 x 40]", 420, 275, 150, 54, TAN),
        Box("p4m", "P4 MDHIFI\nP4m [40 x 40]", 620, 268, 180, 68, PINK, stroke="#8f2742") if v.p4_mdhifi else None,
        Box("p4_td", p4_td_label, 1060, 262, 245, 78, PEACH),
        Box("y4", "Y4\n[40 x 40]", 1365, 272, 120, 62, PEACH),
        Box("up_y4", "Up(Y4)\n40 -> 80", 1060, 388, 145, 54, TAN) if v.p3_upsample else None,
        Box("p3_proj", "P3 1x1 Conv\n[80 x 80]", 420, 405, 150, 54, TAN),
        Box("p3m", "P3 MDHIFI\nP3m [80 x 80]", 620, 398, 165, 68, PINK, stroke="#8f2742"),
        Box("p3_out", p3_out_label, 850, 392, 240, 78 if v.p3_upsample else 68, PEACH),
        Box("a4", a4_label, 620, 565, 190, 76 if not v.a4_used else 70, "#ffe4dc", stroke=RED),
        Box("p4_out", p4_out_label, 1030, 560, 230, 76, PEACH),
        Box("a5", a5_label, 405, 710, 180, 78, "#ffe4dc", stroke=RED),
        Box("p5_down", "Down(P4_out)\nConv s=2\n40 -> 20", 700, 720, 190, 70, TAN),
        Box("p5_out", "Concat Down(P4_out)+Y5+A5\nRepC3 -> P5_out [20]", 1030, 713, 250, 78, PEACH),
        Box("multi", "Multi-scale features\nP3 80 / P4 40 / P5 20", 1385, 495, 190, 100, "#f8eee3", stroke="#bba58b"),
        Box("decoder", "RT-DETR decoder", 1655, 485, 170, 120, "#f8b39b", font_size=18),
    ]
    return {box.key: box for box in items if box is not None}


def build_edges(v: Variant0627) -> list[Edge]:
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
        Edge("y5", "up_y5"),
        Edge("up_y5", "p4_td"),
        Edge("p3_proj", "p3m"),
        Edge("p3m", "a4", color=RED, dashed=True, points=((700, 500), (700, 535))),
        Edge("p5_down", "p5_out"),
        Edge("a5", "p5_out", color=RED, dashed=True),
        Edge("y5", "p5_out", points=((930, 172), (930, 700), (1155, 700))),
        Edge("p3_out", "multi"),
        Edge("p4_out", "multi"),
        Edge("p5_out", "multi"),
        Edge("multi", "decoder"),
    ]

    if v.p4_mdhifi:
        edges.extend(
            [
                Edge("p4_proj", "p4m"),
                Edge("p4m", "p4_td"),
                Edge("p4m", "a5", color=RED, dashed=True, points=((710, 360), (495, 360), (495, 690))),
            ]
        )
    else:
        edges.extend(
            [
                Edge("p4_proj", "p4_td"),
                Edge("p4_proj", "a5", color=RED, dashed=True, points=((500, 360), (500, 690))),
            ]
        )

    edges.append(Edge("p4_td", "y4"))

    if v.p3_upsample:
        edges.extend(
            [
                Edge("y4", "up_y4", points=((1425, 303), (1425, 415), (1130, 415))),
                Edge("up_y4", "p3_out"),
                Edge("p3m", "p3_out"),
            ]
        )
    else:
        edges.append(Edge("p3m", "p3_out"))

    if v.a4_used:
        edges.append(Edge("a4", "p4_out", color=RED, dashed=True))
    edges.append(Edge("y4", "p4_out", points=((1425, 303), (1425, 535), (1140, 535))))
    edges.append(Edge("p4_out", "p5_down"))

    return edges


def title_cells(v: Variant0627, page_id: str) -> list[str]:
    subtitle = "&#xa;".join(escape(line) for line in subtitle_lines(v))
    note = escape(v.note)
    return [
        f'<mxCell id="{page_id}_title" value="{escape(v.title)}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#16222d;fontSize=26;fontStyle=1;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="40" y="18" width="{WIDTH - 80}" height="34" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_subtitle" value="{subtitle}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#3b4a57;fontSize=13;fontStyle=0;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="80" y="56" width="{WIDTH - 160}" height="42" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_backbone_panel" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="40" y="115" width="290" height="610" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_neck_panel" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="380" y="115" width="1460" height="740" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_note" value="{note}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#344756;fontSize=15;fontStyle=0;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="260" y="895" width="{WIDTH - 520}" height="28" as="geometry"/>\n'
        "</mxCell>",
    ]


def drawio_page(v: Variant0627, idx: int) -> str:
    page_id = f"detr0627_v{idx}"
    cells = [
        f'<mxCell id="{page_id}_0"/>',
        f'<mxCell id="{page_id}_root" parent="{page_id}_0"/>',
        *title_cells(v, page_id),
    ]
    cells.extend(drawio_edge(edge, page_id) for edge in build_edges(v))
    cells.extend(drawio_cell_box(box, page_id) for box in build_boxes(v).values())
    root = "\n".join(cells)
    return (
        f'<diagram id="{page_id}" name="{v.slug}">\n'
        f'<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{WIDTH}" pageHeight="{HEIGHT}" math="0" shadow="0">\n'
        f"<root>\n{root}\n</root>\n"
        "</mxGraphModel>\n"
        "</diagram>"
    )


def write_combined_drawio() -> Path:
    pages = "\n".join(drawio_page(v, i) for i, v in enumerate(VARIANTS, start=1))
    COMBINED_DRAWIO_PATH.write_text(f'<mxfile host="codex" pages="{len(VARIANTS)}">\n{pages}\n</mxfile>\n', encoding="utf-8")
    return COMBINED_DRAWIO_PATH


def write_single_drawio(v: Variant0627, idx: int) -> Path:
    path = OUTDIR / v.drawio_name
    path.write_text(f'<mxfile host="codex" pages="1">\n{drawio_page(v, idx)}\n</mxfile>\n', encoding="utf-8")
    return path


def write_svg(v: Variant0627) -> Path:
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
            f'<rect x="380" y="115" width="1460" height="740" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
            '<text x="185" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Backbone</text>',
            '<text x="1110" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Hybrid Encoder and Stepwise Lateral Neck</text>',
        ]
    )
    for edge in edges:
        svg_edge(parts, edge, boxes)
    for box in boxes.values():
        svg_box(parts, box)
    parts.append(
        f'<text x="{WIDTH / 2:.0f}" y="915" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#344756">{escape(v.note)}</text>'
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
