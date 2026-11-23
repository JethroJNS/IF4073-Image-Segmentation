"""Visual evaluation of pilot checkpoint predictions on 10 CamVid validation samples."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

# Ensure project root is on sys.path when invoked as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.config import (
    CAMVID_DIR,
    CLASS_ID_TO_COLOR,
    CLASS_ID_TO_NAME,
    CLASS_NAMES_11,
    IGNORE_INDEX,
    MODELS_DIR,
    NUM_CLASSES,
    RESULTS_DIR,
)
from src.deep_learning.dataset import CamVidDataset, mask_to_rgb, overlay_mask
from src.deep_learning.metrics import SegmentationMetrics
from src.deep_learning.model import get_device, load_checkpoint


def select_validation_indices(num_total: int, num_samples: int = 10) -> List[int]:
    """Select evenly spaced deterministic sample indices across the validation split."""
    if num_samples >= num_total:
        return list(range(num_total))
    return [int(round(i * (num_total - 1) / (num_samples - 1))) for i in range(num_samples)]


def analyze_sample_prediction(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute per-sample class distribution on prediction and ground truth,
    detected classes, and per-sample Pixel Accuracy & mIoU (ignoring GT == 255).
    """
    total_pixels = int(pred_mask.size)
    valid_gt_mask = gt_mask != IGNORE_INDEX
    valid_pixels = int(valid_gt_mask.sum())
    ignored_pixels = int((gt_mask == IGNORE_INDEX).sum())

    # Predicted distribution across all pixels of the 360x480 image
    pred_ids, pred_counts = np.unique(pred_mask, return_counts=True)
    pred_count_map = {int(k): int(v) for k, v in zip(pred_ids.tolist(), pred_counts.tolist())}

    # Ground-truth distribution across all pixels (0..10 and 255)
    gt_ids, gt_counts = np.unique(gt_mask, return_counts=True)
    gt_count_map = {int(k): int(v) for k, v in zip(gt_ids.tolist(), gt_counts.tolist())}

    pred_distribution: Dict[str, Dict[str, Any]] = {}
    detected_classes: List[str] = []

    for cid, cname in enumerate(CLASS_NAMES_11):
        p_cnt = pred_count_map.get(cid, 0)
        g_cnt = gt_count_map.get(cid, 0)
        p_pct = 100.0 * p_cnt / total_pixels
        g_pct = 100.0 * g_cnt / total_pixels
        if p_cnt > 0:
            detected_classes.append(cname)
        pred_distribution[cname] = {
            "class_id": cid,
            "pred_pixels": p_cnt,
            "pred_percent": round(p_pct, 2),
            "gt_pixels": g_cnt,
            "gt_percent": round(g_pct, 2),
        }

    # Evaluate sample accuracy and mIoU strictly excluding GT == 255
    sample_metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    sample_metrics.update(pred_mask, gt_mask)
    eval_res = sample_metrics.compute()

    return {
        "total_pixels": total_pixels,
        "valid_gt_pixels": valid_pixels,
        "ignored_gt_pixels": ignored_pixels,
        "ignored_gt_percent": round(100.0 * ignored_pixels / total_pixels, 2),
        "detected_classes": detected_classes,
        "num_detected_classes": len(detected_classes),
        "pixel_accuracy": eval_res["pixel_accuracy"],
        "miou": eval_res["miou"],
        "per_class_iou": eval_res["per_class_iou"],
        "class_distribution": pred_distribution,
    }


def build_legend_handles(
    analysis: Optional[Dict[str, Any]] = None,
) -> List[mpatches.Patch]:
    """Build legend patches using the official 11-class RGB palette from config.py."""
    handles: List[mpatches.Patch] = []
    for cid in range(NUM_CLASSES):
        cname = CLASS_NAMES_11[cid]
        r, g, b = CLASS_ID_TO_COLOR[cid]
        if analysis is not None:
            d = analysis["class_distribution"][cname]
            label = f"[{cid:2d}] {cname} (GT:{d['gt_percent']:.1f}% | Pred:{d['pred_percent']:.1f}%)"
        else:
            label = f"[{cid:2d}] {cname}"
        handles.append(
            mpatches.Patch(
                facecolor=(r / 255.0, g / 255.0, b / 255.0),
                edgecolor="black",
                linewidth=0.6,
                label=label,
            )
        )

    # Also add Void/Ignore (255) patch for Ground Truth reference
    vr, vg, vb = CLASS_ID_TO_COLOR[IGNORE_INDEX]
    if analysis is not None:
        v_label = f"[255] Void/Ignore (GT:{analysis['ignored_gt_percent']:.1f}%)"
    else:
        v_label = "[255] Void/Ignore (GT only)"
    handles.append(
        mpatches.Patch(
            facecolor=(vr / 255.0, vg / 255.0, vb / 255.0),
            edgecolor="black",
            linewidth=0.6,
            label=v_label,
        )
    )
    return handles


def run_pilot_visualization(
    checkpoint_path: Path = MODELS_DIR / "camvid11_pilot_colab_best.pth",
    output_dir: Path = RESULTS_DIR / "pilot_visualizations",
    num_samples: int = 10,
    input_size: Tuple[int, int] = (360, 480),
) -> Dict[str, Any]:
    """
    Load the pilot checkpoint and generate 4-panel visual comparisons on 10 validation samples
    plus a 10-sample contact sheet.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, ckpt_meta = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    # Load validation split ONLY (never test split!) at target_size=(360, 480)
    val_dataset_norm = CamVidDataset(
        root_dir=CAMVID_DIR,
        split="val",
        target_size=input_size,
        imagenet_normalize=True,
        return_dict=True,
    )
    val_dataset_unnorm = CamVidDataset(
        root_dir=CAMVID_DIR,
        split="val",
        target_size=input_size,
        imagenet_normalize=False,
        return_dict=True,
    )
    assert val_dataset_norm.split == "val", "Must strictly use 'val' split!"

    indices = select_validation_indices(len(val_dataset_norm), num_samples=num_samples)

    print("=== EVALUASI VISUAL CHECKPOINT PILOT (VALIDATION SET) ===")
    print(f"  Checkpoint          : {checkpoint_path}")
    print(f"  Checkpoint Epoch    : {ckpt_meta.get('epoch')} (Stage: {ckpt_meta.get('stage')})")
    print(f"  Checkpoint Best mIoU: {float(ckpt_meta.get('best_miou', 0.0)) * 100:.2f}%")
    print(f"  Device              : {device}")
    print(f"  Dataset Split       : val ({len(val_dataset_norm)} total samples, {len(indices)} selected)")
    print(f"  Input Resolution    : {input_size[0]} x {input_size[1]} (H x W)")
    print(f"  Output Directory    : {output_dir}")

    contact_fig, contact_axes = plt.subplots(
        len(indices), 4, figsize=(18, 3.3 * len(indices) + 1.8)
    )
    contact_fig.suptitle(
        f"CamVid Pilot Checkpoint Visual Evaluation — {len(indices)} Validation Samples (360×480)\n"
        f"Checkpoint: {checkpoint_path.name} (Epoch {ckpt_meta.get('epoch')}, Val mIoU: {float(ckpt_meta.get('best_miou', 0.0))*100:.2f}%)",
        fontsize=15,
        fontweight="bold",
        y=0.993,
    )

    saved_individual_paths: List[str] = []
    sample_reports: List[Dict[str, Any]] = []

    with torch.no_grad():
        for row_idx, ds_idx in enumerate(indices):
            norm_item = val_dataset_norm[ds_idx]
            unnorm_item = val_dataset_unnorm[ds_idx]

            stem = norm_item["stem"]
            filename = f"{stem}.png"
            img_tensor = norm_item["image"].unsqueeze(0).to(device)  # (1, 3, 360, 480)
            gt_mask = norm_item["mask"].cpu().numpy().astype(np.uint8)  # (360, 480) in {0..10, 255}

            # Verify GT mask only contains 0..10 and 255
            unique_gt = set(np.unique(gt_mask).tolist())
            assert unique_gt.issubset(set(range(NUM_CLASSES)) | {IGNORE_INDEX}), (
                f"Unexpected GT IDs in {filename}: {unique_gt}"
            )

            # Reconstruct 360x480 uint8 RGB image for display
            orig_rgb = (
                unnorm_item["image"].permute(1, 2, 0).cpu().numpy() * 255.0
            ).round().astype(np.uint8)

            # Forward pass in eval() + torch.no_grad()
            logits = model(img_tensor)  # (1, 11, 360, 480)
            assert tuple(logits.shape) == (1, NUM_CLASSES, input_size[0], input_size[1])
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Colorize GT and Prediction with the exact same RGB palette from config.py
            gt_rgb = mask_to_rgb(gt_mask)
            pred_rgb = mask_to_rgb(pred_mask)
            pred_overlay = overlay_mask(orig_rgb, pred_mask, alpha=0.45)

            analysis = analyze_sample_prediction(gt_mask=gt_mask, pred_mask=pred_mask)

            # Print per-sample analysis to console
            top_preds = sorted(
                analysis["class_distribution"].items(),
                key=lambda kv: -kv[1]["pred_pixels"],
            )
            top_str = ", ".join(
                f"{cname}: {info['pred_percent']:.1f}% (GT {info['gt_percent']:.1f}%)"
                for cname, info in top_preds
                if info["pred_pixels"] > 0 or info["gt_pixels"] > 0
            )
            print(
                f"\n[Sample {row_idx + 1:02d} | val idx={ds_idx:3d}] File: {filename}\n"
                f"  - Pixel Acc (excl. 255): {analysis['pixel_accuracy']*100:.2f}% | "
                f"Sample mIoU: {analysis['miou']*100:.2f}% | "
                f"GT Void(255): {analysis['ignored_gt_percent']:.2f}%\n"
                f"  - Kelas Terdeteksi ({analysis['num_detected_classes']}/11): {', '.join(analysis['detected_classes'])}\n"
                f"  - Distribusi Prediksi : {top_str}"
            )

            # 1. Save individual 4-panel figure
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.9))
            fig.suptitle(
                f"Validation Sample #{row_idx + 1:02d}: val/{filename} (Index {ds_idx}) — "
                f"Pixel Acc: {analysis['pixel_accuracy']*100:.2f}% | Sample mIoU: {analysis['miou']*100:.2f}%",
                fontsize=12,
                fontweight="bold",
                y=0.97,
            )

            axes[0].imshow(orig_rgb)
            axes[0].set_title(f"1. Original Image\n({input_size[1]}×{input_size[0]})", fontsize=10.5, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(gt_rgb)
            axes[1].set_title(
                f"2. Ground Truth (11 Classes)\n(Void/255: {analysis['ignored_gt_percent']:.1f}%)",
                fontsize=10.5,
                fontweight="bold",
            )
            axes[1].axis("off")

            axes[2].imshow(pred_rgb)
            axes[2].set_title(
                f"3. Model Prediction (11 Classes)\n({analysis['num_detected_classes']} kelas terdeteksi)",
                fontsize=10.5,
                fontweight="bold",
            )
            axes[2].axis("off")

            axes[3].imshow(pred_overlay)
            axes[3].set_title(
                "4. Overlay Prediction\npada Original Image",
                fontsize=10.5,
                fontweight="bold",
            )
            axes[3].axis("off")

            handles = build_legend_handles(analysis=analysis)
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=4,
                fontsize=8.5,
                frameon=True,
                title="Palet Warna 11 Kelas & Perbandingan Distribusi Piksel (Ground Truth vs Model Prediction)",
                title_fontsize=9.5,
            )
            fig.tight_layout(rect=[0, 0.17, 1, 0.93])

            indiv_path = output_dir / f"val_sample_{row_idx + 1:02d}_{stem}.png"
            fig.savefig(indiv_path, dpi=140)
            plt.close(fig)
            saved_individual_paths.append(str(indiv_path))

            # 2. Populate row in the 10-sample contact sheet
            ax_row = contact_axes[row_idx]
            ax_row[0].imshow(orig_rgb)
            ax_row[0].set_title(f"[{row_idx + 1:02d}] Original: {filename}", fontsize=9.5, fontweight="bold")
            ax_row[0].axis("off")

            ax_row[1].imshow(gt_rgb)
            ax_row[1].set_title(
                f"[{row_idx + 1:02d}] GT 11-Class (Void: {analysis['ignored_gt_percent']:.1f}%)",
                fontsize=9.5,
                fontweight="bold",
            )
            ax_row[1].axis("off")

            ax_row[2].imshow(pred_rgb)
            ax_row[2].set_title(
                f"[{row_idx + 1:02d}] Pred ({analysis['num_detected_classes']} cls | Acc: {analysis['pixel_accuracy']*100:.1f}%)",
                fontsize=9.5,
                fontweight="bold",
            )
            ax_row[2].axis("off")

            ax_row[3].imshow(pred_overlay)
            ax_row[3].set_title(
                f"[{row_idx + 1:02d}] Overlay (mIoU: {analysis['miou']*100:.1f}%)",
                fontsize=9.5,
                fontweight="bold",
            )
            ax_row[3].axis("off")

            sample_reports.append(
                {
                    "sample_number": row_idx + 1,
                    "val_index": ds_idx,
                    "filename": filename,
                    "stem": stem,
                    "output_image": str(indiv_path),
                    **analysis,
                }
            )

    # Save 10-sample contact sheet
    shared_handles = build_legend_handles(analysis=None)
    contact_fig.legend(
        handles=shared_handles,
        loc="lower center",
        ncol=6,
        fontsize=10,
        frameon=True,
        title="Palet Warna Konsisten 11 Kelas CamVid (ID 0–10) & Void/Ignore (255 pada GT)",
        title_fontsize=11,
    )
    contact_fig.tight_layout(rect=[0, 0.025, 1, 0.978])
    contact_sheet_path = output_dir / "pilot_val_10_samples_contact_sheet.png"
    contact_fig.savefig(contact_sheet_path, dpi=130)
    plt.close(contact_fig)

    # Save structured JSON report of the 10 samples
    report_json_path = output_dir / "pilot_val_10_samples_analysis.json"
    report_payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt_meta.get("epoch"),
        "checkpoint_best_val_miou": ckpt_meta.get("best_miou"),
        "split": "val",
        "input_resolution_hw": list(input_size),
        "num_samples_visualized": len(sample_reports),
        "contact_sheet": str(contact_sheet_path),
        "individual_visualizations": saved_individual_paths,
        "samples": sample_reports,
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nSaved 10 individual visualizations to: {output_dir}")
    print(f"Saved 10-sample contact sheet to     : {contact_sheet_path}")
    print(f"Saved analysis JSON report to        : {report_json_path}")
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pilot checkpoint predictions on CamVid validation samples."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(MODELS_DIR / "camvid11_pilot_colab_best.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR / "pilot_visualizations"),
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=480)
    args = parser.parse_args()

    run_pilot_visualization(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        input_size=(args.height, args.width),
    )


if __name__ == "__main__":
    main()
