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


OUTDIR = Path("assets/figures/detr0630")
COMBINED_DRAWIO_PATH = OUTDIR / "detr0630_variants.drawio"
WIDTH = 1960
HEIGHT = 980


@dataclass(frozen=True)
class Variant0630:
    slug: str
    title: str
    yaml_name: str
    note: str
    p3_mdhifi: bool
    p4_mdhifi: bool
    p3_upsample: bool
    p4_upsample: bool
    noise_suppression: bool = True

    @property
    def drawio_name(self) -> str:
        return f"{self.slug}_principle.drawio"

    @property
    def svg_name(self) -> str:
        return f"{self.slug}_principle.svg"


VARIANTS = [
    Variant0630(
        slug="b7-exp1-p4upsample-dualmdhifi-noise",
        title="DETR0630 B7 Exp1 P4Upsample DualMDHIFI",
        yaml_name="exp_cfg/detr0630/b7-exp1-p4upsample-dualmdhifi-noise.yaml",
        note="Dual MDHIFI with HFSCC noise suppression. P4 upsample is kept, but the P3 upsample stage is removed so P3c goes directly to P3_out.",
        p3_mdhifi=True,
        p4_mdhifi=True,
        p3_upsample=False,
        p4_upsample=True,
    ),
    Variant0630(
        slug="b7-exp2-noupsample-p3mdhifi-noise",
        title="DETR0630 B7 Exp2 NoUpsample P3MDHIFI",
        yaml_name="exp_cfg/detr0630/b7-exp2-noupsample-p3mdhifi-noise.yaml",
        note="Only the P3 branch uses MDHIFI before HFSCC. Both the P4 upsample and P3 upsample stages are removed; bottom-up fusion uses just P4c and then P5c.",
        p3_mdhifi=True,
        p4_mdhifi=False,
        p3_upsample=False,
        p4_upsample=False,
    ),
    Variant0630(
        slug="b7-exp3-dualupsample-dualmdhifi-noise",
        title="DETR0630 B7 Exp3 DualUpsample DualMDHIFI",
        yaml_name="exp_cfg/detr0630/b7-exp3-dualupsample-dualmdhifi-noise.yaml",
        note="Full B7 baseline: dual MDHIFI, HFSCC noise suppression, P4 upsample from P5c, and P3 upsample from Y4 proj.",
        p3_mdhifi=True,
        p4_mdhifi=True,
        p3_upsample=True,
        p4_upsample=True,
    ),
    Variant0630(
        slug="exp5",
        title="DETR0630 Exp5 P4Upsample P3MDHIFI",
        yaml_name="exp_cfg/detr0630/exp5.yaml",
        note="Derived from DETR0628 B6 P3 HFSCC: only the P3 branch uses MDHIFI, P4 stays plain, P4 upsample is kept, and the P3 upsample stage is removed so P3c goes directly to P3_out.",
        p3_mdhifi=True,
        p4_mdhifi=False,
        p3_upsample=False,
        p4_upsample=True,
    ),
]


def subtitle_lines(v: Variant0630) -> list[str]:
    return [f"YAML: {v.yaml_name}"]


def build_boxes(v: Variant0630) -> dict[str, Box]:
    hfscc_label = "HFSCC\nshift align + consistency\nnoise suppression on"
    p3_out_label = "Concat Up(Y4 proj)+P3c\nRepC3 -> P3_out [80]" if v.p3_upsample else "P3c -> RepC3\nP3_out [80]"
    p4_out_label = (
        "Concat Down(P3_out)+Y4 proj+P4c\nRepC3 -> P4_out [40]"
        if v.p4_upsample
        else "Concat Down(P3_out)+P4c\nRepC3 -> P4_out [40]"
    )
    items = [
        Box("input", "Input image\n640 x 640", 70, 155, 220, 52, ORANGE),
        Box("backbone", "TorchVision SwinV2-Tiny\nStage2-4 outputs", 70, 240, 220, 72, "#9ec8c0"),
        Box("p3", "P3/8 feature [80 x 80]\nIndex + Permute + EMA", 70, 370, 220, 72, BLUE),
        Box("p4", "P4/16 feature [40 x 40]\nIndex + Permute + EMA", 70, 500, 220, 72, BLUE2),
        Box("p5", "P5/32 feature [20 x 20]\nIndex + Permute + EMA", 70, 630, 220, 72, BLUE3),
        Box("p5_proj", "P5 1x1 Conv\n[20 x 20]", 420, 155, 150, 54, TAN),
        Box("aifi", "AIFI encoder\n[20 x 20]", 620, 155, 165, 54, PURPLE),
        Box("y5", "Y5\n[20 x 20]", 850, 155, 155, 54, PEACH),
        Box("p4_proj", "P4 1x1 Conv\n[40 x 40]", 420, 290, 150, 54, TAN),
        Box("p4m", "P4 MDHIFI\nP4m [40 x 40]", 620, 283, 175, 68, PINK, stroke="#8f2742") if v.p4_mdhifi else None,
        Box("p3_proj", "P3 1x1 Conv\n[80 x 80]", 420, 430, 150, 54, TAN),
        Box("p3m", "P3 MDHIFI\nP3m [80 x 80]", 620, 423, 175, 68, PINK, stroke="#8f2742") if v.p3_mdhifi else None,
        Box("hfscc", hfscc_label, 900, 290, 250, 180, GREEN, stroke="#587d3b"),
        Box("p3c", "P3c\n[80 x 80]", 1240, 440, 130, 58, PEACH),
        Box("p4c", "P4c\n[40 x 40]", 1240, 340, 130, 58, PEACH),
        Box("p5c", "P5c\n[20 x 20]", 1240, 240, 130, 58, PEACH),
        Box("up_p5c", "Up(P5c)\n20 -> 40", 1450, 240, 145, 54, TAN) if v.p4_upsample else None,
        Box("y4", "Concat Up(P5c)+P4c\nRepC3 -> Y4 [40]", 1630, 225, 240, 78, PEACH) if v.p4_upsample else None,
        Box("y4_proj", "Y4 proj\n1x1 Conv", 1630, 340, 180, 58, TAN) if v.p4_upsample else None,
        Box("up_y4", "Up(Y4 proj)\n40 -> 80", 1450, 430, 145, 54, TAN) if v.p3_upsample else None,
        Box("p3_out", p3_out_label, 1630, 415, 250, 78 if v.p3_upsample else 68, PEACH),
        Box("down_p3", "Down(P3_out)\nConv s=2\n80 -> 40", 1450, 560, 170, 70, TAN),
        Box("p4_out", p4_out_label, 1630, 545, 280 if v.p4_upsample else 240, 78, PEACH),
        Box("down_p4", "Down(P4_out)\nConv s=2\n40 -> 20", 1450, 695, 170, 70, TAN),
        Box("p5_out", "Concat Down(P4_out)+P5c\nRepC3 -> P5_out [20]", 1630, 685, 250, 78, PEACH),
        Box("multi", "Multi-scale features\nP3 80 / P4 40 / P5 20", 1540, 820, 190, 100, "#f8eee3", stroke="#bba58b"),
        Box("decoder", "RT-DETR decoder", 1770, 810, 150, 120, "#f8b39b", font_size=18),
    ]
    return {box.key: box for box in items if box is not None}


def build_edges(v: Variant0630) -> list[Edge]:
    edges = [
        Edge("input", "backbone"),
        Edge("backbone", "p3"),
        Edge("p3", "p4"),
        Edge("p4", "p5"),
        Edge("p5", "p5_proj", points=((340, 666), (340, 182))),
        Edge("p4", "p4_proj", points=((350, 536), (350, 317))),
        Edge("p3", "p3_proj"),
        Edge("p5_proj", "aifi"),
        Edge("aifi", "y5"),
        Edge("y5", "hfscc", points=((1040, 182), (1040, 235), (960, 235))),
        Edge("hfscc", "p5c"),
        Edge("hfscc", "p4c"),
        Edge("hfscc", "p3c"),
        Edge("p3_out", "down_p3"),
        Edge("down_p3", "p4_out"),
        Edge("p4_out", "down_p4"),
        Edge("down_p4", "p5_out"),
        Edge("p5c", "p5_out", points=((1370, 269), (1900, 269), (1900, 724), (1790, 724))),
        Edge("p3_out", "multi"),
        Edge("p4_out", "multi"),
        Edge("p5_out", "multi"),
        Edge("multi", "decoder"),
    ]

    if v.p4_mdhifi:
        edges.extend(
            [
                Edge("p4_proj", "p4m"),
                Edge("p4m", "hfscc", points=((795, 317), (860, 317), (860, 330), (900, 330))),
            ]
        )
    else:
        edges.append(Edge("p4_proj", "hfscc", points=((570, 317), (900, 317))))

    if v.p3_mdhifi:
        edges.extend(
            [
                Edge("p3_proj", "p3m"),
                Edge("p3m", "hfscc", points=((795, 457), (860, 457), (860, 430), (900, 430))),
            ]
        )
    else:
        edges.append(Edge("p3_proj", "hfscc", points=((570, 457), (900, 457))))

    if v.p4_upsample:
        edges.extend(
            [
                Edge("p5c", "up_p5c"),
                Edge("up_p5c", "y4"),
                Edge("p4c", "y4", points=((1370, 369), (1500, 369), (1500, 264), (1630, 264))),
                Edge("y4", "y4_proj"),
            ]
        )
    if v.p3_upsample:
        edges.extend(
            [
                Edge("y4_proj", "up_y4"),
                Edge("up_y4", "p3_out"),
                Edge("p3c", "p3_out", points=((1370, 469), (1495, 469), (1495, 454), (1630, 454))),
            ]
        )
    else:
        edges.append(Edge("p3c", "p3_out"))

    if v.p4_upsample:
        edges.extend(
            [
                Edge("y4_proj", "p4_out", points=((1810, 369), (1810, 525), (1770, 525))),
                Edge("p4c", "p4_out", points=((1370, 369), (1540, 369), (1540, 540), (1670, 540))),
            ]
        )
    else:
        edges.append(Edge("p4c", "p4_out", points=((1370, 369), (1530, 369), (1530, 584), (1630, 584))))

    return edges


def title_cells(v: Variant0630, page_id: str) -> list[str]:
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
        '  <mxGeometry x="40" y="125" width="290" height="610" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_neck_panel" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="380" y="125" width="1540" height="820" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_note" value="{note}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#344756;fontSize=15;fontStyle=0;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="220" y="945" width="{WIDTH - 440}" height="24" as="geometry"/>\n'
        "</mxCell>",
    ]


def drawio_page(v: Variant0630, idx: int) -> str:
    page_id = f"detr0630_v{idx}"
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


def write_single_drawio(v: Variant0630, idx: int) -> Path:
    path = OUTDIR / v.drawio_name
    path.write_text(f'<mxfile host="codex" pages="1">\n{drawio_page(v, idx)}\n</mxfile>\n', encoding="utf-8")
    return path


def write_svg(v: Variant0630) -> Path:
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
            f'<rect x="40" y="125" width="290" height="610" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
            f'<rect x="380" y="125" width="1540" height="820" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
            '<text x="185" y="148" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Backbone</text>',
            '<text x="1150" y="148" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">HFSCC-Calibrated Hybrid Neck</text>',
        ]
    )
    for edge in edges:
        svg_edge(parts, edge, boxes)
    for box in boxes.values():
        svg_box(parts, box)
    parts.append(
        f'<text x="{WIDTH / 2:.0f}" y="962" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#344756">{escape(v.note)}</text>'
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
