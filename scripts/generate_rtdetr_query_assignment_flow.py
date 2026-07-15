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
SVG_PATH = OUTDIR / "rtdetr_query_assignment_flow.svg"
PNG_PATH = OUTDIR / "rtdetr_query_assignment_flow.png"
DRAWIO_PATH = OUTDIR / "rtdetr_query_assignment_flow.drawio"
WIDTH = 2140
HEIGHT = 1160


def build_boxes() -> dict[str, Box]:
    boxes = [
        Box("train", "Training entry\nscripts/train_yolo.py\nYOLO(args.model).train(...)", 70, 120, 250, 74, ORANGE),
        Box("model", "Model forward\nRT-DETR model graph\nexp_cfg/0705/b5.yaml -> RTDETRDecoder", 70, 260, 250, 86, "#9ec8c0"),
        Box("head", "RTDETRDecoder.forward()\nhead.py:1543-1593", 70, 430, 250, 74, PURPLE),
        Box("encoder_input", "_get_encoder_input()\nproject P2/P3/P4 features\nflatten -> feats, shapes", 410, 120, 285, 86, TAN),
        Box("cdn", "get_cdn_group()\nDN queries from GT\nops.py:188+", 410, 260, 285, 86, "#f6ddd2"),
        Box("decoder_input", "_get_decoder_input()\nhead.py:1644-1695", 410, 430, 285, 86, BLUE),
        Box("topk", "Query selection\nenc_score_head(features)\n topk(max class score, num_queries)\nhead.py:1668-1674", 805, 120, 310, 98, PINK, stroke="#8f2742"),
        Box("refer", "Initial query state\n top_k_features -> embeddings\n top_k_anchors + enc_bbox_head -> refer_bbox\nhead.py:1676-1695", 805, 280, 310, 98, PEACH),
        Box("decoder", "Transformer decoder\nself.decoder(embed, refer_bbox, ...)\noutputs dec_bboxes, dec_scores", 805, 445, 310, 86, GREEN, stroke="#587d3b"),
        Box("preds", "Predictions used for training\nlast decoder layer queries\npred_bboxes[-1], pred_scores[-1]", 1215, 445, 300, 86, "#f8eee3", stroke="#bba58b"),
        Box("loss", "RTDETRDetectionLoss / DETRLoss\nloss.py", 1215, 610, 240, 74, ORANGE),
        Box("matcher", "HungarianMatcher.forward()\nops.py:79-155", 1610, 610, 280, 74, PURPLE),
        Box("cost", "Matching cost\nclass cost + L1 bbox + GIoU\nops.py:120-140", 1610, 755, 280, 86, BLUE),
        Box("assign", "linear_sum_assignment()\nreturn (query_idx, gt_idx)\nops.py:149-154", 1610, 920, 280, 86, PINK, stroke="#8f2742"),
        Box("targets", "_get_loss()\nuse match_indices to gather\npred/query <-> GT pairs\nloss.py:329-347", 1215, 900, 300, 98, PEACH),
        Box("note", "Key point:\nquery creation happens in RTDETRDecoder._get_decoder_input(); query-to-GT assignment happens in HungarianMatcher inside DETRLoss.", 480, 1020, 1180, 72, "#f8eee3", stroke="#bba58b"),
    ]
    return {box.key: box for box in boxes}


def build_edges() -> list[Edge]:
    return [
        Edge("train", "model"),
        Edge("model", "head"),
        Edge("head", "encoder_input"),
        Edge("head", "cdn", points=((345, 467), (370, 467), (370, 303), (410, 303))),
        Edge("head", "decoder_input"),
        Edge("encoder_input", "topk"),
        Edge("encoder_input", "decoder_input", points=((695, 163), (735, 163), (735, 473), (410, 473))),
        Edge("cdn", "decoder_input"),
        Edge("decoder_input", "topk", points=((695, 473), (740, 473), (740, 169), (805, 169))),
        Edge("topk", "refer"),
        Edge("decoder_input", "refer", points=((695, 473), (750, 473), (750, 329), (805, 329))),
        Edge("refer", "decoder"),
        Edge("decoder_input", "decoder", points=((695, 473), (805, 488))),
        Edge("decoder", "preds"),
        Edge("preds", "loss", points=((1515, 488), (1540, 488), (1540, 647), (1455, 647))),
        Edge("loss", "matcher"),
        Edge("matcher", "cost"),
        Edge("cost", "assign"),
        Edge("assign", "targets", points=((1610, 963), (1560, 963), (1560, 949), (1515, 949))),
        Edge("preds", "targets", points=((1365, 531), (1365, 900))),
        Edge("loss", "targets", points=((1335, 684), (1335, 900))),
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
        f'<text x="{WIDTH/2:.1f}" y="42" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="28" font-weight="700" fill="#16222d">RT-DETR Query Selection and Assignment Flow</text>',
        '<text x="1070" y="70" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#3b4a57">Training path: train_yolo.py -> RTDETRDecoder -> DETRLoss -> HungarianMatcher</text>',
        f'<rect x="40" y="95" width="300" height="450" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
        f'<rect x="380" y="95" width="760" height="470" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
        f'<rect x="1185" y="400" width="745" height="645" rx="10" fill="{PANEL}" stroke="#d8d0c2" stroke-width="2"/>',
        '<text x="190" y="122" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Training Entry</text>',
        '<text x="760" y="122" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Query Construction in RTDETRDecoder</text>',
        '<text x="1558" y="427" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#16222d">Query-to-GT Assignment in Loss</text>',
    ]
    for edge in build_edges():
        svg_edge(parts, edge, boxes)
    for box in boxes.values():
        svg_box(parts, box)
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")
    return SVG_PATH


def escape_label(label: str) -> str:
    return escape(label).replace("\n", "&#xa;")


def drawio_title(page_id: str) -> list[str]:
    return [
        f'<mxCell id="{page_id}_title" value="RT-DETR Query Selection and Assignment Flow" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#16222d;fontSize=28;fontStyle=1;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="40" y="18" width="{WIDTH - 80}" height="36" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_subtitle" value="Training path: train_yolo.py -&gt; RTDETRDecoder -&gt; DETRLoss -&gt; HungarianMatcher" style="whiteSpace=wrap;html=1;rounded=0;fillColor={BG};strokeColor={BG};fontColor=#3b4a57;fontSize=14;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
        f'  <mxGeometry x="110" y="56" width="{WIDTH - 220}" height="24" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_panel0" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="40" y="95" width="300" height="450" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_panel1" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="380" y="95" width="760" height="470" as="geometry"/>\n'
        "</mxCell>",
        f'<mxCell id="{page_id}_panel2" value="" style="whiteSpace=wrap;html=1;rounded=1;arcSize=10;fillColor={PANEL};strokeColor=#d8d0c2;strokeWidth=2" vertex="1" parent="{page_id}_root">\n'
        '  <mxGeometry x="1185" y="400" width="745" height="645" as="geometry"/>\n'
        "</mxCell>",
    ]


def drawio_header_labels(page_id: str) -> list[str]:
    labels = [
        ("entry_label", "Training Entry", 90, 110, 200, 24),
        ("query_label", "Query Construction in RTDETRDecoder", 520, 110, 480, 24),
        ("loss_label", "Query-to-GT Assignment in Loss", 1320, 415, 470, 24),
    ]
    cells = []
    for key, value, x, y, w, h in labels:
        cells.append(
            f'<mxCell id="{page_id}_{key}" value="{escape_label(value)}" style="whiteSpace=wrap;html=1;rounded=0;fillColor={PANEL};strokeColor={PANEL};fontColor=#16222d;fontSize=20;fontStyle=1;align=center;verticalAlign=middle" vertex="1" parent="{page_id}_root">\n'
            f'  <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>\n'
            "</mxCell>"
        )
    return cells


def write_drawio() -> Path:
    page_id = "rtdetr_query_assign"
    cells = [f'<mxCell id="{page_id}_0"/>', f'<mxCell id="{page_id}_root" parent="{page_id}_0"/>']
    cells.extend(drawio_title(page_id))
    cells.extend(drawio_header_labels(page_id))
    cells.extend(drawio_edge(edge, page_id) for edge in build_edges())
    cells.extend(drawio_cell_box(box, page_id) for box in build_boxes().values())
    root = "\n".join(cells)
    content = (
        '<mxfile host="codex" pages="1">\n'
        f'<diagram id="{page_id}" name="rtdetr-query-assignment-flow">\n'
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
    print(write_svg())
    print(write_drawio())


if __name__ == "__main__":
    main()
