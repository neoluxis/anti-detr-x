import argparse
import random
from pathlib import Path

import cv2

from visualize_det_heatmap import (
    ActivationCollector,
    build_names,
    compute_cam,
    compute_eigencam,
    compose_panel,
    infer_arch,
    infer_model_channels,
    load_wrapper,
    load_gt_boxes,
    make_overlay,
    map_box_from_orig_to_input,
    preprocess_image,
    prepare_yolo_feat_fallback,
    resolve_layer_indices,
    scale_box_to_original,
    select_detr_target,
    select_yolo_target,
)
import yaml


# Keep this aligned with the RT-DETR comparison set in scripts/report_best_curves.py.
DEFAULT_RUNS = [
    "redetr-l-init-2",
    "rtdetr-p2-init",
    "rtdetr-r18-init",
    "rtdetr-hifi-init-2",
    "rtdetr-wtconv-init",
    "rtdetr-mrfpn-init",
    "rtdetr-HIFI-MRFPN-WTConv-init",
    "exp1-dgwrn",
    "exp2-dgwrn-biagcau",
    "exp3-dgwrn-biagcau-mdhifi",
    "rtdetr-ema-p2-init",
    "rtdetr-resnet101",
    "rtdetr-resnet50",
    "rtdetr-exp3-ema-biagcau-mdhifi",
    "rtdetr-exp2-ema-biagcau",
    "rtdetr-exp1-ema",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-plot Grad-CAM heatmaps for multiple detection runs.")
    parser.add_argument(
        "--project-path",
        type=Path,
        default=Path("runs/detect/cst-sample-5k1k1k-s100"),
        help="Run project directory containing experiment subdirectories.",
    )
    parser.add_argument("--run", action="append", default=[], help="Run directory name. Can be passed multiple times.")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test"), help="Dataset split to sample.")
    parser.add_argument("--num-images", type=int, default=6, help="Number of random labeled images to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image sampling.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size. Default uses run args.")
    parser.add_argument("--device", type=str, default=None, help="Inference device.")
    parser.add_argument("--conf", type=float, default=0.05, help="Minimum score for selecting a target.")
    parser.add_argument("--rank", type=int, default=0, help="Select the Nth highest-scoring detection.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha.")
    parser.add_argument("--crop-scale", type=float, default=3.0, help="Zoom factor for the crop panel.")
    parser.add_argument("--crop-pad", type=float, default=1.4, help="Extra crop padding relative to bbox size.")
    parser.add_argument(
        "--cam-method",
        type=str,
        default="eigencam",
        choices=("eigencam", "gradcam"),
        help="Heatmap method. eigencam is more stable for small-object detection.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("runs/heatmap_compare"), help="Output root directory.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_labeled_images(data_yaml: Path, split: str, num_images: int, seed: int) -> tuple[Path, list[Path]]:
    data_cfg = load_yaml(data_yaml)
    root = Path(data_cfg["path"])
    image_dir = root / data_cfg[split]
    label_dir = root / split / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing split directories under {root} for split={split}")

    labeled_images = []
    for label_path in sorted(label_dir.rglob("*.txt")):
        if label_path.stat().st_size == 0:
            continue
        rel = label_path.relative_to(label_dir).with_suffix(".jpg")
        image_path = image_dir / rel
        if not image_path.exists():
            image_path = image_dir / rel.with_suffix(".png")
        if image_path.exists():
            labeled_images.append(image_path)

    if not labeled_images:
        raise RuntimeError(f"No non-empty labels found under {label_dir}")

    rng = random.Random(seed)
    chosen = rng.sample(labeled_images, k=min(num_images, len(labeled_images)))
    return root, chosen


def main():
    args = parse_args()
    run_names = args.run or list(DEFAULT_RUNS)
    first_run = args.project_path / run_names[0]
    first_args = load_yaml(first_run / "args.yaml")
    data_yaml = Path(first_args["data"])
    dataset_root, sampled_images = sample_labeled_images(data_yaml, args.split, args.num_images, args.seed)

    manifest_dir = args.outdir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "sampled_images.txt"
    manifest_path.write_text(
        "\n".join(str(p.relative_to(dataset_root)) for p in sampled_images) + "\n",
        encoding="utf-8",
    )

    for run_name in run_names:
        run_dir = args.project_path / run_name
        best_pt = run_dir / "weights" / "best.pt"
        train_args = load_yaml(run_dir / "args.yaml")
        arch = infer_arch(str(train_args.get("model", best_pt)), "auto")
        imgsz = args.imgsz or int(train_args.get("imgsz", 640))
        wrapper = load_wrapper(str(best_pt), arch, args.device)
        core_model = wrapper.model
        names = build_names(wrapper)
        device = next(core_model.parameters()).device
        input_channels = infer_model_channels(core_model)
        layer_indices = resolve_layer_indices(core_model, None)
        collector = ActivationCollector([core_model.model[i] for i in layer_indices])
        run_outdir = args.outdir / run_name
        run_outdir.mkdir(parents=True, exist_ok=True)

        try:
            for image_path in sampled_images:
                image_bgr = cv2.imread(str(image_path))
                if image_bgr is None:
                    print(f"skip unreadable image: {image_path}")
                    continue
                gt_boxes = load_gt_boxes(image_path, image_bgr.shape)
                gt_box_orig = gt_boxes[0] if gt_boxes else None

                im = preprocess_image(image_bgr, arch, imgsz, core_model.stride, input_channels=input_channels).to(device)
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
                score = float(target["score"].detach().sigmoid().item() if arch == "detr" else target["score"].detach().item())
                cls_name = names.get(target["cls"], str(target["cls"]))
                panel = compose_panel(
                    image_bgr=image_bgr,
                    overlay_bgr=overlay,
                    box_xyxy=box_xyxy,
                    score=score,
                    cls_name=cls_name,
                    crop_scale=args.crop_scale,
                    crop_pad=args.crop_pad,
                )

                stem = f"{image_path.parent.name}__{image_path.stem}"
                cv2.imwrite(str(run_outdir / f"{stem}_heatmap_panel.jpg"), panel)
                cv2.imwrite(str(run_outdir / f"{stem}_heatmap_overlay.jpg"), overlay)
                cv2.imwrite(str(run_outdir / f"{stem}_heatmap_mask.jpg"), heat_only)
                print(f"[{run_name}] saved {image_path} -> {run_outdir}")
        finally:
            collector.close()


if __name__ == "__main__":
    main()
