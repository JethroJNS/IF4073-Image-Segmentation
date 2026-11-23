"""Single-image CLI semantic segmentation inference for MATLAB GUI integration."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

# Ensure project root is on sys.path when invoked as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.config import (
    CLASS_COLORS_11,
    CLASS_ID_TO_COLOR,
    CLASS_ID_TO_NAME,
    CLASS_NAMES_11,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
    NUM_CLASSES,
)
from src.deep_learning.dataset import mask_to_rgb, overlay_mask, rgb_to_mask
from src.deep_learning.metrics import SegmentationMetrics
from src.deep_learning.model import get_device, load_checkpoint

DEFAULT_CHECKPOINT = MODELS_DIR / "camvid11_full_25epochs_best.pth"
MODEL_INPUT_SIZE: Tuple[int, int] = (360, 480)  # (height, width)


def select_inference_device(device_spec: str = "auto") -> torch.device:
    """Resolve CLI device argument ('auto', 'cpu', 'cuda') into a torch.device."""
    spec = device_spec.lower().strip()
    if spec == "auto":
        return get_device(prefer_cuda=True)
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Device 'cuda' was explicitly requested, but CUDA is not available on this system."
            )
        return torch.device("cuda")
    if spec == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{device_spec}'. Choose from: 'auto', 'cpu', 'cuda'.")


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image file as an RGB uint8 NumPy array of shape (H, W, 3)."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image file not found: {image_path}")
    try:
        with Image.open(image_path) as im:
            rgb = np.array(im.convert("RGB"), dtype=np.uint8, copy=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to read image '{image_path}': {exc}") from exc

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError(f"Invalid RGB image dimensions for '{image_path}': {rgb.shape}")
    return rgb


def preprocess_image(
    image_rgb: np.ndarray,
    input_size: Tuple[int, int] = MODEL_INPUT_SIZE,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Apply training-identical preprocessing:
    1. Resize RGB uint8 image to `input_size` (H=360, W=480) using bilinear interpolation.
    2. Scale to float32 in [0.0, 1.0].
    3. Normalize using ImageNet mean and std.
    4. Return batch tensor of shape (1, 3, 360, 480) and resized uint8 RGB array (360, 480, 3).
    """
    target_h, target_w = input_size
    if (image_rgb.shape[0], image_rgb.shape[1]) != (target_h, target_w):
        pil_img = Image.fromarray(image_rgb)
        resized_rgb = np.array(
            pil_img.resize((target_w, target_h), resample=Image.BILINEAR),
            dtype=np.uint8,
            copy=True,
        )
    else:
        resized_rgb = image_rgb.copy()

    tensor = (
        torch.from_numpy(np.ascontiguousarray(resized_rgb))
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div_(255.0)
    )
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    batch_tensor = tensor.unsqueeze(0)

    if device is not None:
        batch_tensor = batch_tensor.to(device)
    return batch_tensor, resized_rgb


def compute_prediction_distribution(pred_mask: np.ndarray) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Compute pixel count and percentage for each of the 11 classes on `pred_mask`."""
    total_pixels = int(pred_mask.size)
    unique_ids, counts = np.unique(pred_mask, return_counts=True)
    count_map = {int(k): int(v) for k, v in zip(unique_ids.tolist(), counts.tolist())}

    class_distribution: List[Dict[str, Any]] = []
    detected_classes: List[str] = []

    for cid, cname in enumerate(CLASS_NAMES_11):
        px_count = count_map.get(cid, 0)
        pct = (100.0 * px_count / total_pixels) if total_pixels > 0 else 0.0
        r, g, b = CLASS_COLORS_11[cid]
        is_present = px_count > 0
        if is_present:
            detected_classes.append(cname)
        class_distribution.append(
            {
                "class_id": cid,
                "class_name": cname,
                "rgb": [int(r), int(g), int(b)],
                "pixel_count": int(px_count),
                "percentage": round(float(pct), 4),
                "present": bool(is_present),
            }
        )

    return class_distribution, detected_classes


def evaluate_against_ground_truth(
    gt_path: Path,
    pred_mask_orig: np.ndarray,
    pred_mask_model: np.ndarray,
    orig_hw: Tuple[int, int],
    model_hw: Tuple[int, int] = MODEL_INPUT_SIZE,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Load ground-truth RGB mask, map 32->11 classes (with ignore_index=255),
    and compute Pixel Accuracy, mIoU, and per-class IoU using `SegmentationMetrics`.

    Computes metrics both at original image resolution `(orig_h, orig_w)` and at
    the model evaluation resolution `(360, 480)` for direct comparability with `evaluate_test.py`.
    """
    gt_rgb_raw = load_rgb_image(gt_path)
    gt_mask_raw = rgb_to_mask(gt_rgb_raw)

    orig_h, orig_w = orig_hw
    if (gt_mask_raw.shape[0], gt_mask_raw.shape[1]) != (orig_h, orig_w):
        gt_mask_orig = np.array(
            Image.fromarray(gt_mask_raw).resize((orig_w, orig_h), resample=Image.NEAREST),
            dtype=np.uint8,
            copy=True,
        )
    else:
        gt_mask_orig = gt_mask_raw

    model_h, model_w = model_hw
    if (gt_mask_raw.shape[0], gt_mask_raw.shape[1]) != (model_h, model_w):
        gt_mask_model = np.array(
            Image.fromarray(gt_mask_raw).resize((model_w, model_h), resample=Image.NEAREST),
            dtype=np.uint8,
            copy=True,
        )
    else:
        gt_mask_model = gt_mask_raw

    # 1. Metrics at original resolution (H_orig x W_orig)
    metrics_orig = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    metrics_orig.update(pred_mask_orig, gt_mask_orig)
    res_orig = metrics_orig.compute()

    # 2. Metrics at model evaluation resolution (360 x 480)
    metrics_eval = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    metrics_eval.update(pred_mask_model, gt_mask_model)
    res_eval = metrics_eval.compute()

    total_gt_px = int(gt_mask_orig.size)
    ignored_gt_px = int((gt_mask_orig == IGNORE_INDEX).sum())
    valid_gt_px = int(res_orig["total_valid_pixels"])

    per_class_list: List[Dict[str, Any]] = []
    for cid, cname in enumerate(CLASS_NAMES_11):
        per_class_list.append(
            {
                "class_id": cid,
                "class_name": cname,
                "iou": round(float(res_orig["per_class_iou"][cname]), 6),
                "iou_percent": round(float(res_orig["per_class_iou"][cname]) * 100.0, 4),
                "eval_360x480_iou": round(float(res_eval["per_class_iou"][cname]), 6),
                "eval_360x480_iou_percent": round(float(res_eval["per_class_iou"][cname]) * 100.0, 4),
                "gt_pixels": int(res_orig["per_class_gt_pixels"][cname]),
                "pred_valid_pixels": int(res_orig["per_class_pred_pixels"][cname]),
                "active_in_union": bool(res_orig["per_class_valid"][cname]),
            }
        )

    gt_metrics = {
        "ground_truth_path": str(gt_path.resolve()),
        "ignore_index": IGNORE_INDEX,
        "total_pixels": total_gt_px,
        "valid_pixels": valid_gt_px,
        "ignored_pixels": ignored_gt_px,
        "ignored_percent": round(100.0 * ignored_gt_px / max(total_gt_px, 1), 4),
        "pixel_accuracy": round(float(res_orig["pixel_accuracy"]), 6),
        "pixel_accuracy_percent": round(float(res_orig["pixel_accuracy"]) * 100.0, 4),
        "miou": round(float(res_orig["miou"]), 6),
        "miou_percent": round(float(res_orig["miou"]) * 100.0, 4),
        "num_valid_classes": int(res_orig["num_valid_classes"]),
        "per_class_iou": {
            cname: round(float(res_orig["per_class_iou"][cname]), 6)
            for cname in CLASS_NAMES_11
        },
        "per_class_iou_percent": {
            cname: round(float(res_orig["per_class_iou"][cname]) * 100.0, 4)
            for cname in CLASS_NAMES_11
        },
        "eval_360x480": {
            "pixel_accuracy": round(float(res_eval["pixel_accuracy"]), 6),
            "pixel_accuracy_percent": round(float(res_eval["pixel_accuracy"]) * 100.0, 4),
            "miou": round(float(res_eval["miou"]), 6),
            "miou_percent": round(float(res_eval["miou"]) * 100.0, 4),
            "num_valid_classes": int(res_eval["num_valid_classes"]),
            "per_class_iou": {
                cname: round(float(res_eval["per_class_iou"][cname]), 6)
                for cname in CLASS_NAMES_11
            },
            "per_class_iou_percent": {
                cname: round(float(res_eval["per_class_iou"][cname]) * 100.0, 4)
                for cname in CLASS_NAMES_11
            },
        },
        "per_class_details": per_class_list,
        "confusion_matrix_11x11": res_orig["confusion_matrix"].tolist(),
    }

    gt_color_orig = mask_to_rgb(gt_mask_orig)
    return gt_metrics, gt_mask_orig, gt_color_orig


def run_single_prediction(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
    ground_truth_path: Optional[Union[str, Path]] = None,
    device_spec: str = "auto",
) -> Dict[str, Any]:
    """
    Run semantic segmentation inference on a single image and save all outputs to `output_dir`:
    - `pred_mask.png` (uint8 class IDs 0..10 at original image resolution)
    - `pred_color.png` (RGB colorized mask using 11-class palette at original image resolution)
    - `pred_overlay.png` (RGB overlay on original image at original image resolution)
    - `result.json` (metadata, class distribution, timing, device, and optional GT metrics)
    """
    t_total_start = time.perf_counter()

    img_path = Path(image_path)
    ckpt_path = Path(checkpoint_path)
    out_dir = Path(output_dir)

    if not img_path.is_file():
        raise FileNotFoundError(f"Input image not found: {img_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    gt_path: Optional[Path] = None
    if ground_truth_path is not None and str(ground_truth_path).strip() != "":
        gt_path = Path(ground_truth_path)
        if not gt_path.is_file():
            raise FileNotFoundError(f"Ground-truth label file not found: {gt_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    device = select_inference_device(device_spec)

    # 1. Load image at original resolution
    image_rgb = load_rgb_image(img_path)
    orig_h, orig_w = int(image_rgb.shape[0]), int(image_rgb.shape[1])

    # 2. Load trained model from checkpoint
    t_load_start = time.perf_counter()
    model, ckpt_meta = load_checkpoint(ckpt_path, device=device)
    model.eval()
    model_load_ms = (time.perf_counter() - t_load_start) * 1000.0

    # 3. Preprocess (resize 360x480 + ImageNet normalization) and run inference
    t_infer_start = time.perf_counter()
    input_tensor, _ = preprocess_image(image_rgb, input_size=MODEL_INPUT_SIZE, device=device)

    with torch.inference_mode():
        logits = model(input_tensor)  # (1, 11, 360, 480)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model produced non-finite logits (NaN or Inf) during inference.")
        pred_mask_360x480 = (
            torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        )

    inference_ms = (time.perf_counter() - t_infer_start) * 1000.0

    # 4. Resize predicted mask (360, 480) back to original image resolution using nearest-neighbor
    if (orig_h, orig_w) != MODEL_INPUT_SIZE:
        pred_mask_orig = np.array(
            Image.fromarray(pred_mask_360x480).resize((orig_w, orig_h), resample=Image.NEAREST),
            dtype=np.uint8,
            copy=True,
        )
    else:
        pred_mask_orig = pred_mask_360x480.copy()

    # Verify predicted IDs are strictly in 0..10
    unique_pred_ids = set(np.unique(pred_mask_orig).tolist())
    if not unique_pred_ids.issubset(set(range(NUM_CLASSES))):
        raise RuntimeError(f"Invalid predicted class IDs outside 0..10: {unique_pred_ids}")

    # 5. Generate colorized mask and overlay at original resolution
    pred_color_orig = mask_to_rgb(pred_mask_orig)
    pred_overlay_orig = overlay_mask(image_rgb, pred_mask_orig, alpha=0.45)

    # 6. Save output images
    pred_mask_path = out_dir / "pred_mask.png"
    pred_color_path = out_dir / "pred_color.png"
    pred_overlay_path = out_dir / "pred_overlay.png"

    Image.fromarray(pred_mask_orig, mode="L").save(pred_mask_path)
    Image.fromarray(pred_color_orig, mode="RGB").save(pred_color_path)
    Image.fromarray(pred_overlay_orig, mode="RGB").save(pred_overlay_path)

    # 7. Compute class distribution on predicted mask
    class_distribution, detected_classes = compute_prediction_distribution(pred_mask_orig)

    # 8. Optional ground-truth evaluation
    gt_metrics: Optional[Dict[str, Any]] = None
    gt_mask_out_path: Optional[Path] = None
    gt_color_out_path: Optional[Path] = None

    if gt_path is not None:
        gt_metrics, gt_mask_orig, gt_color_orig = evaluate_against_ground_truth(
            gt_path=gt_path,
            pred_mask_orig=pred_mask_orig,
            pred_mask_model=pred_mask_360x480,
            orig_hw=(orig_h, orig_w),
            model_hw=MODEL_INPUT_SIZE,
        )
        gt_mask_out_path = out_dir / "gt_mask.png"
        gt_color_out_path = out_dir / "gt_color.png"
        Image.fromarray(gt_mask_orig, mode="L").save(gt_mask_out_path)
        Image.fromarray(gt_color_orig, mode="RGB").save(gt_color_out_path)

    total_ms = (time.perf_counter() - t_total_start) * 1000.0

    # 9. Construct MATLAB-friendly result.json
    result_payload: Dict[str, Any] = {
        "status": "success",
        "image_path": str(img_path.resolve()),
        "image_filename": img_path.name,
        "original_resolution": {
            "height": orig_h,
            "width": orig_w,
        },
        "model_input_resolution": {
            "height": MODEL_INPUT_SIZE[0],
            "width": MODEL_INPUT_SIZE[1],
        },
        "checkpoint": {
            "path": str(ckpt_path.resolve()),
            "filename": ckpt_path.name,
            "architecture": str(ckpt_meta.get("architecture", "deeplabv3_mobilenet_v3_large")),
            "epoch": int(ckpt_meta.get("epoch", 0)),
            "best_val_miou": round(float(ckpt_meta.get("best_miou", 0.0) or 0.0), 6),
        },
        "device": str(device),
        "timing": {
            "model_load_ms": round(model_load_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "inference_seconds": round(inference_ms / 1000.0, 4),
            "total_ms": round(total_ms, 2),
            "total_seconds": round(total_ms / 1000.0, 4),
        },
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "class_names": list(CLASS_NAMES_11),
        "class_colors_rgb": [[int(r), int(g), int(b)] for (r, g, b) in CLASS_COLORS_11],
        "num_detected_classes": len(detected_classes),
        "detected_classes": detected_classes,
        "class_distribution": class_distribution,
        "has_ground_truth": bool(gt_path is not None),
        "ground_truth_metrics": gt_metrics,
        "outputs": {
            "output_dir": str(out_dir.resolve()),
            "pred_mask_png": str(pred_mask_path.resolve()),
            "pred_color_png": str(pred_color_path.resolve()),
            "pred_overlay_png": str(pred_overlay_path.resolve()),
            "gt_mask_png": str(gt_mask_out_path.resolve()) if gt_mask_out_path else None,
            "gt_color_png": str(gt_color_out_path.resolve()) if gt_color_out_path else None,
            "result_json": str((out_dir / "result.json").resolve()),
        },
    }

    result_json_path = out_dir / "result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    return result_payload


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for single-image semantic segmentation inference."""
    parser = argparse.ArgumentParser(
        description="Run single-image semantic segmentation inference using trained CamVid 11-class checkpoint."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input RGB image (required).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to trained .pth checkpoint (default: models/camvid11_full_25epochs_best.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save pred_mask.png, pred_color.png, pred_overlay.png, and result.json (required).",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Optional path to ground-truth RGB label image for computing Pixel Accuracy and mIoU.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device: 'auto', 'cpu', or 'cuda' (default: 'auto').",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        res = run_single_prediction(
            image_path=args.image,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            ground_truth_path=args.ground_truth,
            device_spec=args.device,
        )
        print(
            f"[OK] Inference completed for '{res['image_filename']}' on device='{res['device']}' "
            f"in {res['timing']['inference_ms']:.1f} ms."
        )
        print(
            f"     Original resolution: {res['original_resolution']['width']}x{res['original_resolution']['height']} | "
            f"Detected classes ({res['num_detected_classes']}/11): {', '.join(res['detected_classes'])}"
        )
        if res["has_ground_truth"] and res["ground_truth_metrics"] is not None:
            gtm = res["ground_truth_metrics"]
            print(
                f"     GT Metrics (Orig Res): Pixel Acc = {gtm['pixel_accuracy_percent']:.2f}%, "
                f"mIoU = {gtm['miou_percent']:.2f}% | "
                f"(360x480 Eval Res): Pixel Acc = {gtm['eval_360x480']['pixel_accuracy_percent']:.2f}%, "
                f"mIoU = {gtm['eval_360x480']['miou_percent']:.2f}%"
            )
        print(f"     Outputs saved to   : {res['outputs']['output_dir']}")
        return 0
    except Exception as exc:
        print(f"[ERROR] Single-image inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
