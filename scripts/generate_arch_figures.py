from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OUTDIR = Path("assets/figures")


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    label: str
    fill: str
    stroke: str = "#22313f"
    text_color: str = "#13202b"
    dashed: bool = False
    rx: int = 14
    fs: int = 20


class SvgCanvas:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            "<defs>",
            '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">'
            '<path d="M0,0 L12,6 L0,12 z" fill="#435466"/>'
            "</marker>",
            '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
            '<feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#1f2d3d" flood-opacity="0.15"/>'
            "</filter>",
            "</defs>",
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f5f1e8"/>',
            f'<text x="{width / 2}" y="46" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="28" font-weight="700" fill="#13202b">{title}</text>',
        ]

    def add(self, svg: str) -> None:
        self.parts.append(svg)

    def rect(self, box: Box) -> None:
        dash = ' stroke-dasharray="10 7"' if box.dashed else ""
        self.add(
            f'<rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" rx="{box.rx}" '
            f'fill="{box.fill}" stroke="{box.stroke}" stroke-width="2"{dash} filter="url(#shadow)"/>'
        )
        self.multiline_text(box.x + box.w / 2, box.y + box.h / 2, box.label, box.fs, box.text_color)

    def multiline_text(self, x: float, y: float, text: str, fs: int = 18, fill: str = "#13202b") -> None:
        lines = text.split("\n")
        line_h = fs + 4
        start_y = y - (len(lines) - 1) * line_h / 2 + fs / 3
        for idx, line in enumerate(lines):
            safe = (
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            self.add(
                f'<text x="{x}" y="{start_y + idx * line_h}" text-anchor="middle" '
                f'font-family="Helvetica, Arial, sans-serif" font-size="{fs}" fill="{fill}">{safe}</text>'
            )

    def line(self, x1: float, y1: float, x2: float, y2: float, dashed: bool = False, color: str = "#435466") -> None:
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3"'
            f'{dash} marker-end="url(#arrow)"/>'
        )

    def polyline(self, pts: list[tuple[float, float]], dashed: bool = False, color: str = "#435466") -> None:
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        coord = " ".join(f"{x},{y}" for x, y in pts)
        self.add(
            f'<polyline points="{coord}" fill="none" stroke="{color}" stroke-width="3"{dash} '
            f'marker-end="url(#arrow)"/>'
        )

    def note(self, x: int, y: int, w: int, h: int, title: str, body: str, fill: str) -> None:
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" '
            f'stroke="#5e6b78" stroke-width="1.8"/>'
        )
        self.multiline_text(x + w / 2, y + 24, title, 18, "#13202b")
        lines = body.split("\n")
        for i, line in enumerate(lines):
            safe = (
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            self.add(
                f'<text x="{x + 16}" y="{y + 52 + i * 22}" font-family="Helvetica, Arial, sans-serif" '
                f'font-size="16" fill="#314252">{safe}</text>'
            )

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


def add_backbone(canvas: SvgCanvas) -> dict[str, tuple[int, int]]:
    x = 70
    w = 300
    boxes = [
        Box(x, 110, w, 54, "Input Image 640x640", "#f7c873"),
        Box(x, 190, w, 96, "SwinV2-Tiny Backbone\nStage 1-2", "#9bc1bc"),
        Box(x, 324, w, 76, "P3/8 Feature\nIndex + Permute + EMA", "#8ecae6"),
        Box(x, 434, w, 76, "P4/16 Feature\nIndex + Permute + EMA", "#72b6d8"),
        Box(x, 544, w, 76, "P5/32 Feature\nIndex + Permute + EMA", "#5fa8c9"),
    ]
    for b in boxes:
        canvas.rect(b)
    for b1, b2 in zip(boxes[:-1], boxes[1:]):
        canvas.line(b1.x + b1.w / 2, b1.y + b1.h, b2.x + b2.w / 2, b2.y)
    return {
        "p3": (x + w, 362),
        "p4": (x + w, 472),
        "p5": (x + w, 582),
    }


def add_baseline_head(canvas: SvgCanvas, anchors: dict[str, tuple[int, int]]) -> None:
    boxes = [
        Box(540, 100, 290, 60, "AIFI Encoder", "#d8c7ff"),
        Box(540, 194, 290, 58, "1x1 Conv -> Y5", "#efdfbf"),
        Box(540, 300, 290, 58, "Upsample + Concat(P4)", "#efe9dc"),
        Box(540, 390, 290, 64, "RepC3 Fusion -> Y4", "#f3d6c6"),
        Box(540, 494, 290, 58, "Upsample + Concat(P3)", "#efe9dc"),
        Box(540, 584, 290, 64, "RepC3 Fusion -> P3 Head", "#f3d6c6"),
        Box(900, 470, 300, 64, "RT-DETR Decoder\n(P3, P4, P5)", "#ffb4a2"),
    ]
    for b in boxes:
        canvas.rect(b)

    canvas.line(anchors["p5"][0], anchors["p5"][1], boxes[0].x, boxes[0].y + 30)
    canvas.line(boxes[0].x + 145, boxes[0].y + 60, boxes[1].x + 145, boxes[1].y)
    canvas.line(boxes[1].x + 145, boxes[1].y + 58, boxes[2].x + 145, boxes[2].y)
    canvas.line(anchors["p4"][0], anchors["p4"][1], boxes[2].x, boxes[2].y + 30)
    canvas.line(boxes[2].x + 145, boxes[2].y + 58, boxes[3].x + 145, boxes[3].y)
    canvas.line(boxes[3].x + 145, boxes[3].y + 64, boxes[4].x + 145, boxes[4].y)
    canvas.line(anchors["p3"][0], anchors["p3"][1], boxes[4].x, boxes[4].y + 30)
    canvas.line(boxes[4].x + 145, boxes[4].y + 58, boxes[5].x + 145, boxes[5].y)

    canvas.polyline([(boxes[3].x + boxes[3].w, 422), (860, 422), (860, 502), (900, 502)])
    canvas.polyline([(boxes[1].x + boxes[1].w, 223), (878, 223), (878, 502), (900, 502)])
    canvas.polyline([(boxes[5].x + boxes[5].w, 616), (860, 616), (860, 532), (900, 532)])

    canvas.note(1260, 170, 220, 132, "Backbone Attention", "EMA on P3/P4/P5\nkeeps channel count\nand refines local context.", "#fff4d6")
    canvas.note(1260, 348, 220, 138, "Neck Design", "Top-down FPN and\nbottom-up PAN both use\nstandard RepC3 blocks.", "#f8ebe5")
    canvas.note(1260, 544, 220, 118, "Decoder", "Three scales feed\nRT-DETR queries for\nfinal detection.", "#ffe0d6")


def add_enhanced_head(canvas: SvgCanvas, anchors: dict[str, tuple[int, int]]) -> None:
    boxes = [
        Box(520, 96, 320, 60, "AIFI Encoder", "#d8c7ff"),
        Box(520, 184, 320, 66, "MDHIFI\n(histogram + gradient + texture)", "#f7a7b8", stroke="#8f2742", text_color="#4d1120"),
        Box(520, 280, 320, 58, "1x1 Conv -> Y5", "#efdfbf"),
        Box(520, 382, 320, 58, "Upsample + Concat(P4)", "#efe9dc"),
        Box(520, 470, 320, 66, "BiAGCAU Fusion -> Y4", "#f4c2b7", stroke="#9b4c39"),
        Box(520, 572, 320, 58, "Upsample + Concat(P3)", "#efe9dc"),
        Box(520, 660, 320, 66, "BiAGCAU Fusion -> P3 Head", "#f4c2b7", stroke="#9b4c39"),
        Box(890, 548, 300, 66, "RT-DETR Decoder\n(P3, P4, P5)", "#ffb4a2"),
    ]
    for b in boxes:
        canvas.rect(b)

    canvas.add(
        '<rect x="486" y="74" width="388" height="680" rx="24" fill="none" stroke="#8f2742" '
        'stroke-width="2.5" stroke-dasharray="12 8"/>'
    )
    canvas.multiline_text(680, 54, "Enhanced Hybrid Encoder + Neck", 20, "#8f2742")

    canvas.line(anchors["p5"][0], anchors["p5"][1], boxes[0].x, boxes[0].y + 30)
    canvas.line(boxes[0].x + 160, boxes[0].y + 60, boxes[1].x + 160, boxes[1].y)
    canvas.line(boxes[1].x + 160, boxes[1].y + 66, boxes[2].x + 160, boxes[2].y)
    canvas.line(boxes[2].x + 160, boxes[2].y + 58, boxes[3].x + 160, boxes[3].y)
    canvas.line(anchors["p4"][0], anchors["p4"][1], boxes[3].x, boxes[3].y + 29)
    canvas.line(boxes[3].x + 160, boxes[3].y + 58, boxes[4].x + 160, boxes[4].y)
    canvas.line(boxes[4].x + 160, boxes[4].y + 66, boxes[5].x + 160, boxes[5].y)
    canvas.line(anchors["p3"][0], anchors["p3"][1], boxes[5].x, boxes[5].y + 29)
    canvas.line(boxes[5].x + 160, boxes[5].y + 58, boxes[6].x + 160, boxes[6].y)

    wt1 = Box(870, 470, 220, 66, "WTConv Downsample", "#d8f0d2", stroke="#4b7a34")
    wt2 = Box(870, 658, 220, 66, "WTConv Downsample", "#d8f0d2", stroke="#4b7a34")
    canvas.rect(wt1)
    canvas.rect(wt2)

    canvas.polyline([(boxes[4].x + boxes[4].w, 503), (855, 503), (855, 503), (870, 503)])
    canvas.polyline([(wt1.x + wt1.w, 503), (1115, 503), (1115, 581), (890, 581)])
    canvas.polyline([(boxes[2].x + boxes[2].w, 309), (1130, 309), (1130, 581), (890, 581)])
    canvas.polyline([(boxes[6].x + boxes[6].w, 693), (855, 693), (855, 693), (890, 693)])
    canvas.polyline([(boxes[6].x + boxes[6].w, 693), (855, 693), (855, 691), (870, 691)])
    canvas.polyline([(wt2.x + wt2.w, 691), (1115, 691), (1115, 611), (890, 611)])

    canvas.note(1230, 116, 250, 146, "MDHIFI", "Inserted after AIFI.\nCombines histogram attention,\nSobel-gradient branch,\ntexture residual, and noise gate.", "#fde3e8")
    canvas.note(1230, 306, 250, 150, "BiAGCAU", "Replaces RepC3 at all\nFPN/PAN fusion nodes.\nUses top-down and bottom-up\npaths with cross gating.", "#f9e0d7")
    canvas.note(1230, 520, 250, 134, "Dynamic WTConv", "Wavelet branch + base conv\nwith adaptive gating.\nUsed for both downsample\ntransitions in PAN.", "#e3f5de")
    canvas.note(1230, 686, 250, 90, "EMA", "Backbone-side EMA blocks on\nP3/P4/P5 stay unchanged.", "#fff4d6")


def build_baseline() -> Path:
    canvas = SvgCanvas(1520, 820, "SwinV2-Tiny-EMA RT-DETR Architecture")
    anchors = add_backbone(canvas)
    add_baseline_head(canvas, anchors)
    out = OUTDIR / "swinv2_tiny_ema_rtdetr_architecture.svg"
    canvas.save(out)
    return out


def build_enhanced() -> Path:
    canvas = SvgCanvas(1520, 840, "SwinV2-Tiny-EMA + WTConv + BiAGCAU + MDHIFI RT-DETR")
    anchors = add_backbone(canvas)
    add_enhanced_head(canvas, anchors)
    out = OUTDIR / "swinv2_tiny_ema_wtconv_biagcau_mdhifi_rtdetr_architecture.svg"
    canvas.save(out)
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    baseline = build_baseline()
    enhanced = build_enhanced()
    print(baseline)
    print(enhanced)


if __name__ == "__main__":
    main()
