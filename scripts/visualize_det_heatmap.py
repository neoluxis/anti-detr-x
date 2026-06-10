import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import torch
from ultralytics import RTDETR, YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate detection attention heatmaps for YOLO and RT-DETR models.")
    parser.add_argument("--model", type=str, required=True, help="Model weights or yaml path.")
    parser.add_argument("--source", type=str, required=True, help="Image file or directory.")
    parser.add_argument("--arch", type=str, default="auto", choices=("auto", "yolo", "detr"), help="Model family.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", type=str, default=None, help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--conf", type=float, default=0.05, help="Minimum score when selecting a target.")
    parser.add_argument("--rank", type=int, default=0, help="Select the Nth highest-scoring detection.")
    parser.add_argument(
        "--layer-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional feature layer indices for CAM. Defaults to head input sources.",
    )
    parser.add_argument("--max-images", type=int, default=50, help="Maximum number of images to process.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha.")
    parser.add_argument("--crop-scale", type=float, default=3.0, help="Zoom factor for the cropped panel.")
    parser.add_argument("--crop-pad", type=float, default=1.4, help="Extra crop padding relative to bbox size.")
    parser.add_argument(
        "--cam-method",
        type=str,
        default="eigencam",
        choices=("eigencam", "gradcam"),
        help="Heatmap method. eigencam is more stable for small-object detection.",
    )
    parser.add_argument("--outdir", type=str, default="runs/heatmap", help="Output directory.")
    return parser.parse_args()


def infer_arch(model_path: str, arch: str) -> str:
    if arch != "auto":
        return arch
    lower = model_path.lower()
    return "detr" if "detr" in lower else "yolo"


def collect_images(source: str, max_images: int) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")
    images = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    return images[:max_images]


def load_wrapper(model_path: str, arch: str, device: str | None):
    wrapper = RTDETR(model_path) if arch == "detr" else YOLO(model_path)
    if device is not None:
        wrapper.to(device)
    wrapper.model.eval()
    wrapper.model.requires_grad_(True)
    return wrapper


def infer_model_channels(core_model) -> int:
    yaml_channels = getattr(getattr(core_model, "yaml", None), "get", lambda *_: None)("channels")
    if yaml_channels is not None:
        return int(yaml_channels)
    for child in core_model.modules():
        conv = getattr(child, "conv", None)
        if conv is not None and hasattr(conv, "in_channels"):
            return int(conv.in_channels)
        if hasattr(child, "in_channels"):
            return int(child.in_channels)
    return 3


def preprocess_image(
    image_bgr: np.ndarray,
    arch: str,
    imgsz: int,
    stride: int | torch.Tensor,
    input_channels: int = 3,
) -> torch.Tensor:
    if arch == "detr":
        letterbox = LetterBox(imgsz, auto=False, scale_fill=True)
    else:
        stride = int(stride.max().item()) if isinstance(stride, torch.Tensor) else int(stride)
        letterbox = LetterBox(imgsz, auto=False, stride=stride)
    transformed = letterbox(image=image_bgr)
    if input_channels == 1:
        transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)[..., None]
    else:
        transformed = transformed[..., ::-1]
    transformed = transformed.transpose(2, 0, 1)
    transformed = np.ascontiguousarray(transformed)
    tensor = torch.from_numpy(transformed).float().unsqueeze(0) / 255.0
    return tensor


def resolve_layer_indices(core_model, layer_indices: list[int] | None) -> list[int]:
    if layer_indices:
        return layer_indices
    head = core_model.model[-1]
    if hasattr(head, "f"):
        f = head.f
        if isinstance(f, int):
            return [f]
        return [int(i) for i in f if isinstance(i, int) and i >= 0]
    return [len(core_model.model) - 2]


def infer_label_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    if "images" not in parts:
        return None
    idx = parts.index("images")
    label_parts = parts.copy()
    label_parts[idx] = "labels"
    label_path = Path(*label_parts).with_suffix(".txt")
    return label_path


def load_gt_boxes(image_path: Path, image_shape: tuple[int, int]) -> list[torch.Tensor]:
    label_path = infer_label_path(image_path)
    if label_path is None or not label_path.exists():
        return []
    h, w = image_shape[:2]
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls_id, xc, yc, bw, bh = map(float, line.split())
        x1 = (xc - bw / 2.0) * w
        y1 = (yc - bh / 2.0) * h
        x2 = (xc + bw / 2.0) * w
        y2 = (yc + bh / 2.0) * h
        boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
    return boxes


def box_iou_single(boxes1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(boxes1[:, 0], box2[0])
    y1 = torch.maximum(boxes1[:, 1], box2[1])
    x2 = torch.minimum(boxes1[:, 2], box2[2])
    y2 = torch.minimum(boxes1[:, 3], box2[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (box2[2] - box2[0]).clamp(min=0) * (box2[3] - box2[1]).clamp(min=0)
    union = area1 + area2 - inter + 1e-6
    return inter / union


def map_box_from_orig_to_input(box_xyxy: torch.Tensor, orig_shape: tuple[int, int], input_shape: tuple[int, int], arch: str) -> torch.Tensor:
    box = box_xyxy.clone().float()
    oh, ow = orig_shape[:2]
    ih, iw = input_shape[:2]
    if arch == "detr":
        box[[0, 2]] *= iw / ow
        box[[1, 3]] *= ih / oh
        return box

    gain = min(iw / ow, ih / oh)
    new_w = round(ow * gain)
    new_h = round(oh * gain)
    pad_w = (iw - new_w) / 2.0
    pad_h = (ih - new_h) / 2.0
    box[[0, 2]] = box[[0, 2]] * gain + pad_w
    box[[1, 3]] = box[[1, 3]] * gain + pad_h
    return box


class ActivationCollector:
    def __init__(self, modules: list[torch.nn.Module]):
        self.activations: list[torch.Tensor] = []
        self.handles = [m.register_forward_hook(self._hook) for m in modules]

    def _hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.retain_grad()
            self.activations.append(output)

    def clear(self):
        self.activations.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def select_yolo_target(raw_output, rank: int, conf: float, head=None, gt_box_xyxy: torch.Tensor | None = None, input_hw=None):
    pred = raw_output[0] if isinstance(raw_output, tuple) else raw_output
    raw = raw_output[1] if isinstance(raw_output, tuple) and len(raw_output) > 1 else None
    if isinstance(raw, dict) and "one2many" in raw:
        raw = raw["one2many"]
    if isinstance(raw, dict) and "scores" in raw and "boxes" in raw:
        boxes = raw["boxes"]
        scores = raw["scores"]
        cls_scores, cls_ids = scores[0].max(dim=0)
        decoded_box = None
        if head is not None and hasattr(head, "_get_decode_boxes"):
            decoded_box = head._get_decode_boxes(raw)[0].transpose(0, 1)
            decoded_box = ops.xywh2xyxy(decoded_box)
        elif boxes.shape[1] == 4:
            decoded_box = ops.xywh2xyxy(boxes[0].transpose(0, 1))

        if gt_box_xyxy is not None and decoded_box is not None and input_hw is not None:
            ious = box_iou_single(decoded_box, gt_box_xyxy.to(decoded_box.device))
            anchor_idx = int(torch.argmax(ious).item())
        else:
            valid = torch.nonzero(cls_scores >= conf).squeeze(1)
            ordered = (
                valid[torch.argsort(cls_scores[valid], descending=True)]
                if valid.numel()
                else torch.argsort(cls_scores, descending=True)
            )
            anchor_idx = int(ordered[min(rank, ordered.numel() - 1)].item())
        cls_idx = cls_ids[anchor_idx]
        if boxes.shape[1] == 4:
            box = boxes[0, :, anchor_idx]
        elif head is not None and hasattr(head, "_get_decode_boxes"):
            box = head._get_decode_boxes(raw)[0, :, anchor_idx]
        elif pred.ndim == 3 and pred.shape[-1] == 6 and anchor_idx < pred.shape[1]:
            box = pred[0, anchor_idx, :4]
        elif pred.ndim == 3 and pred.shape[1] >= 4:
            box = pred[0, :4, anchor_idx]
        else:
            box = boxes[0, :, anchor_idx]
        return {
            "score": scores[0, cls_idx, anchor_idx],
            "box": box,
            "cls": int(cls_idx.item()),
            "format": "xywh_input",
            "target_scalar": scores[0, cls_idx, anchor_idx] + 0.05 * box.mean(),
        }

    if pred.ndim != 3:
        raise RuntimeError(f"Unexpected YOLO output shape: {tuple(pred.shape)}")

    if pred.shape[-1] == 6:
        scores = pred[0, :, 4]
        valid = torch.nonzero(scores >= conf).squeeze(1)
        ordered = valid[torch.argsort(scores[valid], descending=True)] if valid.numel() else torch.argsort(scores, descending=True)
        idx = ordered[min(rank, ordered.numel() - 1)]
        return {
            "score": pred[0, idx, 4],
            "box": pred[0, idx, :4],
            "cls": int(pred[0, idx, 5].item()),
            "format": "xyxy_input",
            "target_scalar": pred[0, idx, 4] + 0.05 * pred[0, idx, :4].mean(),
        }

    boxes = pred[:, :4, :]
    scores = pred[:, 4:, :]
    cls_scores, cls_ids = scores[0].max(dim=0)
    valid = torch.nonzero(cls_scores >= conf).squeeze(1)
    ordered = valid[torch.argsort(cls_scores[valid], descending=True)] if valid.numel() else torch.argsort(cls_scores, descending=True)
    anchor_idx = ordered[min(rank, ordered.numel() - 1)]
    cls_idx = cls_ids[anchor_idx]
    return {
        "score": scores[0, cls_idx, anchor_idx],
        "box": boxes[0, :, anchor_idx],
        "cls": int(cls_idx.item()),
        "format": "xywh_input",
        "target_scalar": scores[0, cls_idx, anchor_idx] + 0.05 * boxes[0, :, anchor_idx].mean(),
    }


def select_detr_target(raw_output, rank: int, conf: float, gt_box_xyxy: torch.Tensor | None = None, orig_shape=None):
    pred, extra = raw_output if isinstance(raw_output, tuple) else (raw_output, None)
    if extra is None:
        raise RuntimeError("RT-DETR forward did not return decoder extras required for Grad-CAM.")
    dec_bboxes, dec_scores, _, _, _ = extra
    final_scores = dec_scores[-1, 0].sigmoid()
    if gt_box_xyxy is not None and orig_shape is not None:
        boxes = ops.xywh2xyxy(dec_bboxes[-1, 0].detach().cpu())
        boxes[:, [0, 2]] *= orig_shape[1]
        boxes[:, [1, 3]] *= orig_shape[0]
        ious = box_iou_single(boxes, gt_box_xyxy.cpu())
        query_idx = int(torch.argmax(ious).item())
        cls_idx = int(torch.argmax(final_scores[query_idx]).item())
    else:
        flat_scores = final_scores.flatten()
        valid = torch.nonzero(flat_scores >= conf).squeeze(1)
        ordered = valid[torch.argsort(flat_scores[valid], descending=True)] if valid.numel() else torch.argsort(flat_scores, descending=True)
        flat_idx = ordered[min(rank, ordered.numel() - 1)]
        query_idx = torch.div(flat_idx, final_scores.shape[1], rounding_mode="floor")
        cls_idx = flat_idx - query_idx * final_scores.shape[1]
    return {
        "score": dec_scores[-1, 0, query_idx, cls_idx],
        "box": dec_bboxes[-1, 0, query_idx, :4],
        "cls": int(cls_idx if isinstance(cls_idx, int) else cls_idx.item()),
        "format": "xywh_normalized",
        "target_scalar": dec_scores[-1, 0, query_idx, cls_idx] + 0.05 * dec_bboxes[-1, 0, query_idx, :4].mean(),
    }


def compute_cam(activations: list[torch.Tensor], input_hw: tuple[int, int]) -> np.ndarray:
    cams = []
    h, w = input_hw
    for act in activations:
        grad = act.grad
        if grad is None:
            continue
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=False))[0]
        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        cams.append(cam)
    if not cams:
        raise RuntimeError("No CAM maps were produced. Check target layers and gradient flow.")
    cam = np.mean(cams, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam


def compute_eigencam(activations: list[torch.Tensor], input_hw: tuple[int, int], gt_box_input: torch.Tensor | None = None) -> np.ndarray:
    cams = []
    scores = []
    h, w = input_hw
    for act in activations:
        feat = act.detach().float()[0].cpu().numpy()  # C,H,W
        c, fh, fw = feat.shape
        flat = feat.reshape(c, -1).T  # HW,C
        flat = flat - flat.mean(axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(flat, full_matrices=False)
            principal = flat @ vh[0]
        except np.linalg.LinAlgError:
            principal = flat.mean(axis=1)
        cam = principal.reshape(fh, fw)
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        cams.append(cam)
        if gt_box_input is not None:
            x1, y1, x2, y2 = np.round(gt_box_input.cpu().numpy()).astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w - 1), min(y2, h - 1)
            if x2 > x1 and y2 > y1:
                scores.append(float(cam[y1 : y2 + 1, x1 : x2 + 1].mean()))
            else:
                scores.append(0.0)
        else:
            scores.append(float(cam.mean()))

    if not cams:
        raise RuntimeError("No activation maps were captured for EigenCAM.")
    best = cams[int(np.argmax(scores))]
    best = np.maximum(best, 0)
    return best / (best.max() + 1e-8)


def prepare_yolo_feat_fallback(raw_output) -> list[torch.Tensor]:
    raw = raw_output[1] if isinstance(raw_output, tuple) and len(raw_output) > 1 else None
    if isinstance(raw, dict) and "one2many" in raw:
        raw = raw["one2many"]
    feats = raw.get("feats") if isinstance(raw, dict) else None
    prepared = []
    if isinstance(feats, list):
        for feat in feats:
            if isinstance(feat, torch.Tensor) and feat.requires_grad:
                feat.retain_grad()
                prepared.append(feat)
    return prepared


def scale_box_to_original(box: torch.Tensor, box_format: str, arch: str, input_shape: tuple[int, int], orig_shape: tuple[int, int]):
    box = box.detach().float().cpu().view(1, 4).clone()
    if arch == "detr":
        box = ops.xywh2xyxy(box)
        box[:, [0, 2]] *= orig_shape[1]
        box[:, [1, 3]] *= orig_shape[0]
        return box[0].numpy()

    if box_format == "xywh_input":
        box = ops.xywh2xyxy(box)
    box = ops.scale_boxes(input_shape, box, orig_shape)
    return box[0].numpy()


def make_overlay(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    heat = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0.0)
    return overlay, colored


def pad_crop_box(box_xyxy: np.ndarray, image_shape: tuple[int, int], pad_scale: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy.astype(np.float32)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(x2 - x1, 4.0)
    bh = max(y2 - y1, 4.0)
    half_w = bw * pad_scale * 0.5
    half_h = bh * pad_scale * 0.5
    h, w = image_shape[:2]
    left = max(int(round(cx - half_w)), 0)
    top = max(int(round(cy - half_h)), 0)
    right = min(int(round(cx + half_w)), w - 1)
    bottom = min(int(round(cy + half_h)), h - 1)
    if right <= left:
        right = min(left + 1, w - 1)
    if bottom <= top:
        bottom = min(top + 1, h - 1)
    return left, top, right, bottom


def draw_dashed_line(image: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], thickness: int = 2):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    dist = np.linalg.norm(p2 - p1)
    if dist < 1:
        return
    step = 10.0
    direction = (p2 - p1) / dist
    for start in np.arange(0, dist, step * 2):
        s = p1 + direction * start
        e = p1 + direction * min(start + step, dist)
        cv2.line(image, tuple(np.round(s).astype(int)), tuple(np.round(e).astype(int)), color, thickness, cv2.LINE_AA)


def compose_panel(image_bgr: np.ndarray, overlay_bgr: np.ndarray, box_xyxy: np.ndarray, score: float, cls_name: str, crop_scale: float, crop_pad: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = np.round(box_xyxy).astype(int)
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h - 1)

    full = overlay_bgr.copy()
    cv2.rectangle(full, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        full,
        f"{cls_name} {score:.3f}",
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    crop_left, crop_top, crop_right, crop_bottom = pad_crop_box(box_xyxy, image_bgr.shape, crop_pad)
    crop = overlay_bgr[crop_top : crop_bottom + 1, crop_left : crop_right + 1]
    zoom_w = max(int(crop.shape[1] * crop_scale), 160)
    zoom_h = max(int(crop.shape[0] * crop_scale), 160)
    zoom = cv2.resize(crop, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
    zoom_box = np.array(
        [
            (x1 - crop_left) * crop_scale,
            (y1 - crop_top) * crop_scale,
            (x2 - crop_left) * crop_scale,
            (y2 - crop_top) * crop_scale,
        ]
    ).round().astype(int)
    cv2.rectangle(zoom, (zoom_box[0], zoom_box[1]), (zoom_box[2], zoom_box[3]), (0, 255, 255), 2)

    gap = 40
    canvas_h = max(h, zoom_h) + 20
    canvas_w = w + gap + zoom_w + 20
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    left_origin = (10, 10)
    right_origin = (10 + w + gap, 10 + max((h - zoom_h) // 2, 0))
    canvas[left_origin[1] : left_origin[1] + h, left_origin[0] : left_origin[0] + w] = full
    canvas[right_origin[1] : right_origin[1] + zoom_h, right_origin[0] : right_origin[0] + zoom_w] = zoom
    cv2.rectangle(
        canvas,
        (right_origin[0], right_origin[1]),
        (right_origin[0] + zoom_w, right_origin[1] + zoom_h),
        (60, 60, 60),
        2,
    )

    left_top = (left_origin[0] + x1, left_origin[1] + y1)
    left_bottom = (left_origin[0] + x1, left_origin[1] + y2)
    right_top = (right_origin[0], right_origin[1])
    right_bottom = (right_origin[0], right_origin[1] + zoom_h)
    draw_dashed_line(canvas, left_top, right_top, (0, 0, 0))
    draw_dashed_line(canvas, left_bottom, right_bottom, (0, 0, 0))
    return canvas


def build_names(wrapper) -> dict[int, str]:
    names = getattr(wrapper.model, "names", None)
    if isinstance(names, dict):
        return names
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return {}


def main():
    args = parse_args()
    arch = infer_arch(args.model, args.arch)
    image_paths = collect_images(args.source, args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.source}")

    wrapper = load_wrapper(args.model, arch, args.device)
    core_model = wrapper.model
    names = build_names(wrapper)
    device = next(core_model.parameters()).device
    input_channels = infer_model_channels(core_model)
    layer_indices = resolve_layer_indices(core_model, args.layer_indices)
    modules = [core_model.model[i] for i in layer_indices]
    collector = ActivationCollector(modules)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"skip unreadable image: {image_path}")
                continue
            gt_boxes = load_gt_boxes(Path(image_path), image_bgr.shape)
            gt_box_orig = gt_boxes[0] if gt_boxes else None

            im = preprocess_image(image_bgr, arch, args.imgsz, core_model.stride, input_channels=input_channels).to(device)
            collector.clear()
            core_model.zero_grad(set_to_none=True)
            raw_output = core_model(im)
            fallback_feats = prepare_yolo_feat_fallback(raw_output) if arch == "yolo" else []
            gt_box_input = (
                map_box_from_orig_to_input(gt_box_orig, image_bgr.shape[:2], tuple(im.shape[2:]), arch)
                if gt_box_orig is not None
                else None
            )

            target = (
                select_detr_target(raw_output, args.rank, args.conf, gt_box_xyxy=gt_box_orig, orig_shape=image_bgr.shape[:2])
                if arch == "detr"
                else select_yolo_target(
                    raw_output,
                    args.rank,
                    args.conf,
                    head=core_model.model[-1],
                    gt_box_xyxy=gt_box_input,
                    input_hw=tuple(im.shape[2:]),
                )
            )
            cam_sources = collector.activations
            if arch == "yolo" and not cam_sources:
                cam_sources = fallback_feats

            if args.cam_method == "gradcam":
                target.get("target_scalar", target["score"]).backward()
                if arch == "yolo" and (not cam_sources or not any(getattr(x, "grad", None) is not None for x in cam_sources)):
                    cam_sources = fallback_feats
                cam_input = compute_cam(cam_sources, tuple(im.shape[2:]))
            else:
                cam_input = compute_eigencam(cam_sources, tuple(im.shape[2:]), gt_box_input=gt_box_input)
            cam_orig = cv2.resize(cam_input, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            box_xyxy = scale_box_to_original(
                target["box"],
                target["format"],
                arch,
                tuple(im.shape[2:]),
                image_bgr.shape[:2],
            )
            overlay, heat_only = make_overlay(image_bgr, cam_orig, args.alpha)
            cls_name = names.get(target["cls"], str(target["cls"]))
            panel = compose_panel(
                image_bgr=image_bgr,
                overlay_bgr=overlay,
                box_xyxy=box_xyxy,
                score=float(target["score"].detach().sigmoid().item() if arch == "detr" else target["score"].detach().item()),
                cls_name=cls_name,
                crop_scale=args.crop_scale,
                crop_pad=args.crop_pad,
            )

            stem = f"{image_path.parent.name}__{image_path.stem}"
            cv2.imwrite(str(outdir / f"{stem}_heatmap_panel.jpg"), panel)
            cv2.imwrite(str(outdir / f"{stem}_heatmap_overlay.jpg"), overlay)
            cv2.imwrite(str(outdir / f"{stem}_heatmap_mask.jpg"), heat_only)
            print(f"saved heatmaps for {image_path} -> {outdir}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
