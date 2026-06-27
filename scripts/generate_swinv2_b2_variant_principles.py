from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from textwrap import wrap


OUTDIR = Path("assets/figures")
DRAWIO_PATH = OUTDIR / "rtdetr_swinv2_tiny_b2_variant_principles.drawio"
BASELINE_DRAWIO_PATH = OUTDIR / "rtdetr_swinv2_tiny_b2_p3_mdhifi.drawio"


@dataclass(frozen=True)
class Box:
    key: str
    label: str
    x: int
    y: int
    w: int
    h: int
    fill: str
    stroke: str = "#263746"
    font_size: int = 15
    font_style: int = 0
    rounded: bool = True


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    color: str = "#41576b"
    dashed: bool = False
    points: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Variant:
    title: str
    page_name: str
    svg_name: str
    run_name: str
    yaml_name: str
    note: str
    p3_name: str = "P3"
    p4_name: str = "P4"
    p5_name: str = "P5"
    aux_p4: bool = False
    aux_p4_dilated: bool = False
    aux_p5: bool = False
    p4_mdhifi: bool = False


BLUE = "#a9d6e5"
BLUE2 = "#7db8d8"
BLUE3 = "#5aa4c8"
TAN = "#f0e3cc"
PEACH = "#efcfbf"
PINK = "#f6b1c3"
GREEN = "#d9efce"
ORANGE = "#f6c979"
PURPLE = "#d9ccff"
PANEL = "#fffdf9"
BG = "#f7f4ee"
RED = "#b94747"


VARIANTS = [
    Variant(
        title="B2 P3-MDHIFI baseline",
        page_name="b2-p3-mdhifi-e100-3",
        svg_name="rtdetr_swinv2_tiny_b2_p3_mdhifi.svg",
        run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-12k4k4k-e100-3",
        yaml_name="rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi.yaml",
        note="P3m joins Up(Y4), then standard bottom-up PAN produces P4/P5.",
    ),
    Variant(
        title="B2 P3-MDHIFI BiDFF-PAN",
        page_name="b2-p3-mdhifi-bidff-pan",
        svg_name="rtdetr_swinv2_tiny_b2_p3_mdhifi_bidff_pan.svg",
        run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-bidff-pan-12k4k4k-e100",
        yaml_name="rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi-bidff-pan.yaml",
        note="Top-down Y3 is refined by the bottom-up PAN path into Z4 and Z5.",
        p3_name="Y3/P3",
        p4_name="Z4/P4",
        p5_name="Z5/P5",
    ),
    Variant(
        title="B2 P3-MDHIFI + Aux PAN",
        page_name="b2-p3-mdhifi-aux-pan",
        svg_name="rtdetr_swinv2_tiny_b2_p3_mdhifi_aux_pan.svg",
        run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-aux-pan-12k4k4k-e100",
        yaml_name="rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi-aux-pan.yaml",
        note="P3m adds auxiliary routes into both P4 and P5 PAN fusion nodes.",
        aux_p4=True,
        aux_p5=True,
    ),
    Variant(
        title="B2 P3-MDHIFI + Aux P4 Dilated",
        page_name="b2-p3-mdhifi-aux-p4-dilated",
        svg_name="rtdetr_swinv2_tiny_b2_p3_mdhifi_aux_p4_dilated.svg",
        run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-aux-p4-dilated-12k4k4k-e100",
        yaml_name="rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi-aux-p4-dilated.yaml",
        note="Only the P3m -> P4 auxiliary route is kept, using dilated stride-2 Conv.",
        aux_p4=True,
        aux_p4_dilated=True,
    ),
    Variant(
        title="B2 P3-MDHIFI + Aux P4 Plain",
        page_name="b2-p3-mdhifi-aux-p4-plain",
        svg_name="rtdetr_swinv2_tiny_b2_p3_mdhifi_aux_p4_plain.svg",
        run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-aux-p4-plain-12k4k4k-e100",
        yaml_name="rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi-aux-p4-plain.yaml",
        note="Only the P3m -> P4 auxiliary route is kept, using plain stride-2 Conv.",
        aux_p4=True,
    ),
    Variant(
        title="B2 P3-MDHIFI + Aux P4 + P4-MDHIFI",
        page_name="b2-p3-mdhifi-aux-p4-p4mdhifi",
        svg_name="rtdetr_swinv2_tiny_b2_p3_mdhifi_aux_p4_p4mdhifi.svg",
        run_name="rtdetr-swinv2-tiny-unfreeze-last3-ema-b2-p3-mdhifi-aux-p4-p4mdhifi-12k4k4k-e100",
        yaml_name="rtdetr-swinv2-tiny-last3-pretrained-ema-b2-p3-mdhifi-aux-p4-p4mdhifi.yaml",
        note="P3m adds the P4 auxiliary route; P4_out is refined by MDHIFI before P5 downsample.",
        aux_p4=True,
        p4_mdhifi=True,
    ),
]


def variant_boxes(v: Variant) -> list[Box]:
    p4_concat = "Concat Down(Y3)+Y4" if v.p3_name.startswith("Y3") else "Concat Down(P3)+Y4"
    p5_concat = "Concat Down(Z4)+Y5" if v.p4_name.startswith("Z4") else "Concat Down(P4)+Y5"
    if v.aux_p4:
        p4_concat += "+AuxP3m"
    if v.aux_p5:
        p5_concat += "+AuxP3m"
    if v.p4_mdhifi:
        p5_concat = "Concat Down(P4m)+Y5"

    p4_down_name = "Down(Y3)" if v.p3_name.startswith("Y3") else "Down(P3)"
    p5_down_name = "Down(Z4)" if v.p4_name.startswith("Z4") else "Down(P4)"
    if v.p4_mdhifi:
        p5_down_name = "Down(P4m)"

    boxes = [
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
        Box("p3_out", f"Concat Up(Y4)+P3m\nRepC3 -> {v.p3_name} [80]", 850, 392, 235, 78, PEACH),
        Box("p4_down", f"{p4_down_name} Conv s=2\n80 -> 40", 620, 565, 190, 62, TAN),
        Box("p4_out", f"{p4_concat}\nRepC3 -> {v.p4_name} [40]", 875, 558, 220, 76, PEACH),
        Box("p5_down", f"{p5_down_name} Conv s=2\n40 -> 20", 620, 720, 190, 62, TAN),
        Box("p5_out", f"{p5_concat}\nRepC3 -> {v.p5_name} [20]", 875, 713, 220, 76, PEACH),
        Box("multi", "Multi-scale features\nP3 80 / P4 40 / P5 20", 1190, 495, 190, 100, "#f8eee3", stroke="#bba58b"),
        Box("decoder", "RT-DETR decoder", 1460, 485, 170, 120, "#f8b39b", font_size=18),
    ]
    if v.aux_p4:
        aux = "Aux Down(P3m)\nConv s=2 d=2\n80 -> 40" if v.aux_p4_dilated else "Aux Down(P3m)\nConv s=2\n80 -> 40"
        boxes.append(Box("aux_p4", aux, 405, 565, 170, 70, "#ffe4dc", stroke=RED))
    if v.aux_p5:
        boxes.append(Box("aux_p5", "Aux Down(P3m)\n2x Conv s=2\n80 -> 20", 405, 710, 170, 78, "#ffe4dc", stroke=RED))
    if v.p4_mdhifi:
        boxes.append(Box("p4m", "P4 MDHIFI\nP4m [40 x 40]", 620, 645, 190, 58, PINK, stroke="#8f2742"))
    return boxes


def variant_edges(v: Variant) -> list[Edge]:
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
        Edge("y5", "td_p4", points=((1120, 172), (1120, 235), (715, 235))),
        Edge("p4_proj", "td_p4"),
        Edge("td_p4", "y4"),
        Edge("y4", "p3_out", points=((1120, 303), (1120, 370), (970, 370))),
        Edge("p3_proj", "p3m"),
        Edge("p3m", "p3_out"),
        Edge("p3_out", "p4_down"),
        Edge("p4_down", "p4_out"),
        Edge("y4", "p4_out", points=((1130, 303), (1130, 535), (1000, 535))),
        Edge("p5_down", "p5_out"),
        Edge("y5", "p5_out", points=((1135, 172), (1135, 700), (1000, 700))),
        Edge("p3_out", "multi"),
        Edge("p4_out", "multi"),
        Edge("p5_out", "multi"),
        Edge("multi", "decoder"),
    ]

    if v.aux_p4:
        edges.extend(
            [
                Edge("p3m", "aux_p4", color=RED, dashed=True, points=((690, 500), (490, 500))),
                Edge("aux_p4", "p4_out", color=RED, dashed=True),
            ]
        )
    if v.aux_p5:
        edges.extend(
            [
                Edge("p3m", "aux_p5", color=RED, dashed=True, points=((675, 520), (490, 520))),
                Edge("aux_p5", "p5_out", color=RED, dashed=True),
            ]
        )
    if v.p4_mdhifi:
        edges.extend(
            [
                Edge("p4_out", "p4m"),
                Edge("p4m", "p5_down"),
            ]
        )
    else:
        edges.append(Edge("p4_out", "p5_down"))
    return edges


def escape_label(label: str) -> str:
    return escape(label).replace("\n", "&#xa;")


def box_center(box: Box) -> tuple[float, float]:
    return box.x + box.w / 2, box.y + box.h / 2


def svg_text(parts: list[str], x: float, y: float, text: str, size: int, fill: str = "#16222d", weight: str = "400") -> None:
    lines = text.split("\n")
    line_h = size + 4
    start = y - (len(lines) - 1) * line_h / 2 + size / 3
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x:.1f}" y="{start + i * line_h:.1f}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{escape(line)}</text>'
        )


def svg_box(parts: list[str], box: Box) -> None:
    rx = 8 if box.rounded else 0
    parts.append(
        f'<rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" rx="{rx}" '
        f'fill="{box.fill}" stroke="{box.stroke}" stroke-width="2"/>'
    )
    svg_text(parts, box.x + box.w / 2, box.y + box.h / 2, box.label, box.font_size)


def edge_points(edge: Edge, boxes: dict[str, Box]) -> list[tuple[float, float]]:
    src = boxes[edge.source]
    dst = boxes[edge.target]
    sx, sy = box_center(src)
    tx, ty = box_center(dst)
    if edge.points:
        return [(src.x + src.w, sy), *edge.points, (dst.x, ty)]
    if abs(tx - sx) > abs(ty - sy):
        return [(src.x + src.w, sy), (dst.x, ty)]
    return [(sx, src.y + src.h), (tx, dst.y)]


def svg_edge(parts: list[str], edge: Edge, boxes: dict[str, Box]) -> None:
    pts = edge_points(edge, boxes)
    coord = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dash = ' stroke-dasharray="8 6"' if edge.dashed else ""
    parts.append(
        f'<polyline points="{coord}" fill="none" stroke="{edge.color}" stroke-width="2.4"{dash} '
        f'marker-end="url(#arrow)"/>'
    )


def subtitle_lines(v: Variant) -> list[str]:
    text = f"Run: {v.run_name} | YAML: {v.yaml_name}"
    return wrap(text, width=150)


def write_svg(v: Variant) -> Path:
    boxes = {b.key: b for b in variant_boxes(v)}
    edges = variant_edges(v)
    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1700" height="940" viewBox="0 0 1700 940">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M0,0 L12,6 L0,12 z" fill="#41576b"/>',
        "</marker>",
        "</defs>",
        f'<rect x="0" y="0" width="1700" height="940" fill="{BG}"/>',
        f'<text x="850" y="42" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="26" font-weight="700" fill="#16222d">{escape(v.title)}</text>',
    ]
    for i, line in enumerate(subtitle_lines(v)):
        parts.append(
            f'<text x="850" y="{68 + i * 18}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="13" fill="#3b4a57">{escape(line)}</text>'
        )
    parts.extend(
        [
            f'<rect x="40" y="115" width="290" height="610" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
            f'<rect x="380" y="115" width="1040" height="720" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
            '<text x="185" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Backbone</text>',
            '<text x="900" y="138" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Hybrid Encoder and Neck</text>',
        ]
    )
    for edge in edges:
        svg_edge(parts, edge, boxes)
    for box in boxes.values():
        svg_box(parts, box)
    parts.append(
        f'<text x="850" y="895" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="15" fill="#344756">{escape(v.note)}</text>'
    )
    parts.append("</svg>")
    path = OUTDIR / v.svg_name
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def drawio_cell_box(box: Box, page_id: str) -> str:
    style = (
        "whiteSpace=wrap;html=1;rounded=1;arcSize=8;dashed=0;"
        f"fillColor={box.fill};strokeColor={box.stroke};strokeWidth=2;"
        f"fontColor=#16222d;fontSize={box.font_size};fontStyle={box.font_style};"
        "align=center;verticalAlign=middle;spacing=8"
    )
    cell_id = f"{page_id}_{box.key}"
    return (
        f'<mxCell id="{cell_id}" value="{escape_label(box.label)}" style="{style}" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" as="geometry"/>\n'
        "</mxCell>"
    )


def drawio_edge(edge: Edge, page_id: str) -> str:
    dash = "1" if edge.dashed else "0"
    style = (
        "edgeStyle=orthogonalEdgeStyle;orthogonal=1;rounded=0;orthogonalLoop=0;"
        f"jettySize=0;html=1;strokeColor={edge.color};strokeWidth=2.2;dashed={dash};"
        "endArrow=block;endFill=1;curved=0;jumpStyle=none;"
    )
    edge_id = f"{page_id}_e_{edge.source}_{edge.target}"
    source = f"{page_id}_{edge.source}"
    target = f"{page_id}_{edge.target}"
    if edge.points:
        pts = "\n".join(f'      <mxPoint x="{x}" y="{y}"/>' for x, y in edge.points)
        geom = f'<mxGeometry relative="1" as="geometry">\n    <Array as="points">\n{pts}\n    </Array>\n  </mxGeometry>'
    else:
        geom = '<mxGeometry relative="1" as="geometry"/>'
    return (
        f'<mxCell id="{edge_id}" style="{style}" edge="1" parent="{page_id}_root" source="{source}" target="{target}">\n'
        f"  {geom}\n"
        "</mxCell>"
    )


def drawio_title(v: Variant, page_id: str) -> list[str]:
    subtitle = "\n".join(subtitle_lines(v))
    return [
        f'<mxCell id="{page_id}_title" value="{escape_label(v.title)}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#16222d;fontSize=26;fontStyle=1;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="40" y="18" width="1620" height="34" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_subtitle" value="{escape_label(subtitle)}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#3b4a57;fontSize=13;fontStyle=0;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="80" y="56" width="1540" height="42" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_backbone_panel" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="40" y="115" width="290" height="610" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_neck_panel" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="380" y="115" width="1040" height="720" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_note" value="{escape_label(v.note)}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#344756;fontSize=15;fontStyle=0;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="260" y="870" width="1180" height="34" as="geometry"/>\n'
        "</mxCell>",
    ]


def drawio_page(v: Variant, idx: int) -> str:
    page_id = f"b2v{idx}"
    cells = [
        f'<mxCell id="{page_id}_0"/>',
        f'<mxCell id="{page_id}_root" parent="{page_id}_0"/>',
        *drawio_title(v, page_id),
    ]
    boxes = variant_boxes(v)
    edges = variant_edges(v)
    # Draw edges before nodes so arrows sit behind box labels.
    cells.extend(drawio_edge(edge, page_id) for edge in edges)
    cells.extend(drawio_cell_box(box, page_id) for box in boxes)
    root = "\n".join(cells)
    return (
        f'<diagram id="{page_id}" name="{escape(v.page_name)}">\n'
        '<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
        'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1700" pageHeight="940" math="0" shadow="0">\n'
        f"<root>\n{root}\n</root>\n"
        "</mxGraphModel>\n"
        "</diagram>"
    )


def write_drawio() -> Path:
    pages = "\n".join(drawio_page(v, i) for i, v in enumerate(VARIANTS, start=1))
    content = f'<mxfile host="codex" pages="{len(VARIANTS)}">\n{pages}\n</mxfile>\n'
    DRAWIO_PATH.write_text(content, encoding="utf-8")
    return DRAWIO_PATH


def write_single_drawio(v: Variant, idx: int, path: Path) -> Path:
    content = f'<mxfile host="codex" pages="1">\n{drawio_page(v, idx)}\n</mxfile>\n'
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    drawio = write_drawio()
    print(drawio)
    print(write_single_drawio(VARIANTS[0], 1, BASELINE_DRAWIO_PATH))
    for variant in VARIANTS:
        print(write_svg(variant))


if __name__ == "__main__":
    main()
