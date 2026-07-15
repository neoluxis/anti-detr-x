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


OUTDIR = Path("assets/figures/detr0703")
COMBINED_DRAWIO_PATH = OUTDIR / "detr0703_variants.drawio"
WIDTH = 2040
HEIGHT = 1020


@dataclass(frozen=True)
class Variant0703:
    slug: str
    title: str
    run_name: str
    yaml_name: str
    note: str
    lite_mdhifi: bool = False
    p2_head: bool = False

    @property
    def drawio_name(self) -> str:
        return f"{self.slug}_principle.drawio"

    @property
    def svg_name(self) -> str:
        return f"{self.slug}_principle.svg"


VARIANTS = [
    Variant0703(
        slug="b1-litemdhifi",
        title="DETR0703 B1 LiteMDHIFI + HFSCC RT-DETR",
        run_name="rtdetr-swinv2-tiny-0703-b1-litemdhifi-12k4k4k-e100",
        yaml_name="exp_cfg/0703/b1.yaml",
        note="P3/P4 lateral branches use LiteMDHIFI before HFSCC; decoder receives P3/P4/P5 outputs.",
        lite_mdhifi=True,
    ),
    Variant0703(
        slug="b2-p2head",
        title="DETR0703 B2 P2 Head + MDHIFI + HFSCC RT-DETR",
        run_name="rtdetr-swinv2-tiny-0703-b2-p2head-12k4k4k-e100",
        yaml_name="exp_cfg/0703/b2.yaml",
        note="Adds P2/4 branch with MDHIFI and extends the FPN/PAN path so the decoder receives P2/P3/P4/P5 outputs.",
        p2_head=True,
    ),
]


def subtitle_lines(v: Variant0703) -> list[str]:
    return [f"Run: {v.run_name}", f"YAML: {v.yaml_name}"]


def build_boxes(v: Variant0703) -> dict[str, Box]:
    mdhifi_name = "LiteMDHIFI" if v.lite_mdhifi else "MDHIFI"
    p3_node = "P3c\n[80 x 80]"
    p4_node = "P4c\n[40 x 40]"
    p5_node = "P5c\n[20 x 20]"
    boxes = [
        Box("input", "Input image\n640 x 640", 70, 155, 220, 52, ORANGE),
        Box("backbone", "TorchVision SwinV2-Tiny\nsplit stage outputs", 70, 240, 220, 72, "#9ec8c0"),
        Box("p2", "P2/4 feature [160 x 160]\nIndex + Permute + EMA", 70, 355, 220, 72, "#cfe8d8") if v.p2_head else None,
        Box("p3", "P3/8 feature [80 x 80]\nIndex + Permute + EMA", 70, 475 if v.p2_head else 370, 220, 72, BLUE),
        Box("p4", "P4/16 feature [40 x 40]\nIndex + Permute + EMA", 70, 595 if v.p2_head else 500, 220, 72, BLUE2),
        Box("p5", "P5/32 feature [20 x 20]\nIndex + Permute + EMA", 70, 715 if v.p2_head else 630, 220, 72, BLUE3),
        Box("p5_proj", "P5 1x1 Conv\n[20 x 20]", 420, 155, 150, 54, TAN),
        Box("aifi", "AIFI encoder\n[20 x 20]", 620, 155, 165, 54, PURPLE),
        Box("y5", "Y5\n[20 x 20]", 850, 155, 155, 54, PEACH),
        Box("p4_proj", "P4 1x1 Conv\n[40 x 40]", 420, 305, 150, 54, TAN),
        Box("p4m", f"P4 {mdhifi_name}\nP4m [40 x 40]", 620, 296, 185, 72, PINK, stroke="#8f2742"),
        Box("p3_proj", "P3 1x1 Conv\n[80 x 80]", 420, 455, 150, 54, TAN),
        Box("p3m", f"P3 {mdhifi_name}\nP3m [80 x 80]", 620, 446, 185, 72, PINK, stroke="#8f2742"),
        Box("p2_proj", "P2 1x1 Conv\n[160 x 160]", 420, 605, 150, 54, TAN) if v.p2_head else None,
        Box("p2m", "P2 MDHIFI\nP2m [160 x 160]", 620, 596, 185, 72, PINK, stroke="#8f2742") if v.p2_head else None,
        Box("hfscc", "HFSCC\nP3/P4/P5 shift align\ncross-scale calibration", 900, 280, 250, 190, GREEN, stroke="#587d3b"),
        Box("p3c", p3_node, 1230, 455, 130, 58, PEACH),
        Box("p4c", p4_node, 1230, 355, 130, 58, PEACH),
        Box("p5c", p5_node, 1230, 255, 130, 58, PEACH),
        Box("up_p5c", "Up(P5c)\n20 -> 40", 1430, 255, 140, 54, TAN),
        Box("y4", "Concat Up(P5c)+P4c\nRepC3 -> Y4 [40]", 1620, 240, 245, 78, PEACH),
        Box("y4_proj", "Y4 proj\n1x1 Conv", 1620, 355, 185, 58, TAN),
        Box("up_y4", "Up(Y4 proj)\n40 -> 80", 1430, 455, 140, 54, TAN),
        Box("y3", "Concat Up(Y4)+P3c\nRepC3 -> Y3 [80]", 1620, 440, 245, 78, PEACH) if v.p2_head else None,
        Box("y3_proj", "Y3 proj\n1x1 Conv", 1620, 555, 185, 58, TAN) if v.p2_head else None,
        Box("up_y3", "Up(Y3 proj)\n80 -> 160", 1430, 645, 150, 54, TAN) if v.p2_head else None,
        Box("p2_out", "Concat Up(Y3)+P2m\nRepC3 -> P2_out [160]", 1620, 630, 260, 78, PEACH) if v.p2_head else None,
        Box("p3_out", "Concat Up(Y4)+P3c\nRepC3 -> P3_out [80]", 1620, 440, 255, 78, PEACH) if not v.p2_head else None,
        Box("down_p2", "Down(P2_out)\nConv s=2\n160 -> 80", 1430, 755, 170, 70, TAN) if v.p2_head else None,
        Box("down_p3", "Down(P3_out)\nConv s=2\n80 -> 40", 1430, 610, 170, 70, TAN) if not v.p2_head else None,
        Box("p3_out_b2", "Concat Down(P2)+Y3 proj+P3c\nRepC3 -> P3_out [80]", 1620, 745, 300, 78, PEACH) if v.p2_head else None,
        Box("down_p3_b2", "Down(P3_out)\nConv s=2\n80 -> 40", 1430, 865, 170, 70, TAN) if v.p2_head else None,
        Box("p4_out", "Concat Down(P3)+Y4 proj+P4c\nRepC3 -> P4_out [40]", 1620, 600 if not v.p2_head else 855, 300, 78, PEACH),
        Box("down_p4", "Down(P4_out)\nConv s=2\n40 -> 20", 1430, 745 if not v.p2_head else 950, 170, 60, TAN),
        Box("p5_out", "Concat Down(P4)+P5c\nRepC3 -> P5_out [20]", 1620, 735 if not v.p2_head else 940, 265, 70, PEACH),
        Box("multi", "Multi-scale features\nP3/P4/P5" if not v.p2_head else "Multi-scale features\nP2/P3/P4/P5", 1220, 790 if not v.p2_head else 880, 190, 90, "#f8eee3", stroke="#bba58b"),
        Box("decoder", "RT-DETR decoder", 1020, 795 if not v.p2_head else 885, 170, 90, "#f8b39b", font_size=18),
    ]
    return {box.key: box for box in boxes if box is not None}


def build_edges(v: Variant0703) -> list[Edge]:
    edges = [
        Edge("input", "backbone"),
        Edge("backbone", "p3"),
        Edge("p3", "p4"),
        Edge("p4", "p5"),
        Edge("p5", "p5_proj", points=((345, 744 if v.p2_head else 666), (345, 182))),
        Edge("p5_proj", "aifi"),
        Edge("aifi", "y5"),
        Edge("p4", "p4_proj", points=((350, 631 if v.p2_head else 536), (350, 332))),
        Edge("p4_proj", "p4m"),
        Edge("p3", "p3_proj", points=((350, 511 if v.p2_head else 406), (350, 482))),
        Edge("p3_proj", "p3m"),
        Edge("y5", "hfscc", points=((1030, 182), (1030, 235), (960, 235))),
        Edge("p4m", "hfscc", points=((805, 332), (860, 332), (860, 335), (900, 335))),
        Edge("p3m", "hfscc", points=((805, 482), (860, 482), (860, 430), (900, 430))),
        Edge("hfscc", "p5c"),
        Edge("hfscc", "p4c"),
        Edge("hfscc", "p3c"),
        Edge("p5c", "up_p5c"),
        Edge("up_p5c", "y4"),
        Edge("p4c", "y4", points=((1360, 384), (1500, 384), (1500, 279), (1620, 279))),
        Edge("y4", "y4_proj"),
        Edge("y4_proj", "up_y4"),
        Edge("up_y4", "y3" if v.p2_head else "p3_out"),
        Edge("p3c", "y3" if v.p2_head else "p3_out", points=((1360, 484), (1495, 484), (1495, 479), (1620, 479))),
        Edge("p5c", "p5_out", points=((1360, 284), (1935, 284), (1935, 770 if not v.p2_head else 975), (1885, 975 if v.p2_head else 770))),
        Edge("p4_out", "down_p4"),
        Edge("down_p4", "p5_out"),
        Edge("p4_out", "multi"),
        Edge("p5_out", "multi"),
        Edge("multi", "decoder"),
    ]
    if v.p2_head:
        edges.extend(
            [
                Edge("backbone", "p2"),
                Edge("p2", "p3"),
                Edge("p2", "p2_proj", points=((345, 391), (345, 632))),
                Edge("p2_proj", "p2m"),
                Edge("y3", "y3_proj"),
                Edge("y3_proj", "up_y3"),
                Edge("up_y3", "p2_out"),
                Edge("p2m", "p2_out", points=((805, 632), (1380, 632), (1380, 669), (1620, 669))),
                Edge("p2_out", "down_p2"),
                Edge("down_p2", "p3_out_b2"),
                Edge("y3_proj", "p3_out_b2", points=((1805, 584), (1805, 728), (1770, 728))),
                Edge("p3c", "p3_out_b2", points=((1360, 484), (1390, 484), (1390, 784), (1620, 784))),
                Edge("p3_out_b2", "down_p3_b2"),
                Edge("down_p3_b2", "p4_out"),
                Edge("y4_proj", "p4_out", points=((1805, 384), (1805, 840), (1770, 840))),
                Edge("p4c", "p4_out", points=((1360, 384), (1390, 384), (1390, 894), (1620, 894))),
                Edge("p2_out", "multi"),
                Edge("p3_out_b2", "multi"),
            ]
        )
    else:
        edges.extend(
            [
                Edge("p3_out", "down_p3"),
                Edge("down_p3", "p4_out"),
                Edge("y4_proj", "p4_out", points=((1805, 384), (1805, 585), (1770, 585))),
                Edge("p4c", "p4_out", points=((1360, 384), (1390, 384), (1390, 639), (1620, 639))),
                Edge("p3_out", "multi"),
            ]
        )
    return edges


def drawio_title(v: Variant0703, page_id: str) -> list[str]:
    subtitle = "&#xa;".join(escape(line) for line in subtitle_lines(v))
    return [
        f'<mxCell id="{page_id}_title" value="{escape(v.title)}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#16222d;fontSize=26;fontStyle=1;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="40" y="18" width="{WIDTH - 80}" height="34" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_subtitle" value="{subtitle}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#3b4a57;fontSize=13;fontStyle=0;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="80" y="56" width="{WIDTH - 160}" height="42" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_backbone_panel" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="40" y="125" width="290" height="{710 if v.p2_head else 610}" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_neck_panel" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="380" y="125" width="1580" height="{860 if v.p2_head else 770}" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_note" value="{escape(v.note)}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#344756;fontSize=15;fontStyle=0;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="240" y="{HEIGHT - 36}" width="{WIDTH - 480}" height="24" as="geometry"/>\n'
        "</mxCell>",
    ]


def drawio_page(v: Variant0703, idx: int) -> str:
    page_id = f"detr0703_v{idx}"
    cells = [
        f'<mxCell id="{page_id}_0"/>',
        f'<mxCell id="{page_id}_root" parent="{page_id}_0"/>',
        *drawio_title(v, page_id),
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


def write_single_drawio(v: Variant0703, idx: int) -> Path:
    path = OUTDIR / v.drawio_name
    path.write_text(f'<mxfile host="codex" pages="1">\n{drawio_page(v, idx)}\n</mxfile>\n', encoding="utf-8")
    return path


def write_svg(v: Variant0703) -> Path:
    boxes = build_boxes(v)
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
            f'<rect x="40" y="125" width="290" height="{710 if v.p2_head else 610}" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
            f'<rect x="380" y="125" width="1580" height="{860 if v.p2_head else 770}" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
            '<text x="185" y="148" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Backbone</text>',
            '<text x="1170" y="148" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">HFSCC-Calibrated Hybrid Neck</text>',
        ]
    )
    for edge in build_edges(v):
        svg_edge(parts, edge, boxes)
    for box in boxes.values():
        svg_box(parts, box)
    parts.append(
        f'<text x="{WIDTH / 2:.0f}" y="{HEIGHT - 18}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="#344756">{escape(v.note)}</text>'
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
