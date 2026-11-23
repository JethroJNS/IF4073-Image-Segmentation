"""Final evaluation of the best full-training checkpoint on the 168-image CamVid test set."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when invoked as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.config import (
    CAMVID_DIR,
    CLASS_ID_TO_COLOR,
    CLASS_NAMES_11,
    EXPECTED_SPLIT_COUNTS,
    IGNORE_INDEX,
    MODELS_DIR,
    NUM_CLASSES,
    RESULTS_DIR,
)
from src.deep_learning.dataset import CamVidDataset, mask_to_rgb, overlay_mask
from src.deep_learning.metrics import SegmentationMetrics
from src.deep_learning.model import create_loss_function, get_device, load_checkpoint


def select_evenly_spaced_indices(num_total: int, num_samples: int = 15) -> List[int]:
    """Select `num_samples` evenly spaced indices across `[0, num_total - 1]`."""
    if num_samples >= num_total:
        return list(range(num_total))
    return [int(round(i * (num_total - 1) / (num_samples - 1))) for i in range(num_samples)]


def compute_detailed_metrics_from_cm(
    cm: np.ndarray,
    val_per_class_iou: Dict[str, float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compute global and per-class metrics (IoU, Precision, Recall, F1/Dice, Support)
    from the 11x11 confusion matrix (where ignore_index=255 has already been excluded).
    """
    total_valid_pixels = int(cm.sum())
    tp = np.diag(cm).astype(np.float64)
    gt_support = cm.sum(axis=1).astype(np.float64)
    pred_support = cm.sum(axis=0).astype(np.float64)
    fn = gt_support - tp
    fp = pred_support - tp
    union = gt_support + pred_support - tp

    pixel_accuracy = float(tp.sum() / total_valid_pixels) if total_valid_pixels > 0 else 0.0

    per_class_rows: List[Dict[str, Any]] = []
    valid_ious: List[float] = []
    valid_precisions: List[float] = []
    valid_recalls: List[float] = []
    valid_f1s: List[float] = []

    for cid, cname in enumerate(CLASS_NAMES_11):
        u_c = float(union[cid])
        gt_c = float(gt_support[cid])
        pr_c = float(pred_support[cid])
        tp_c = float(tp[cid])
        fp_c = float(fp[cid])
        fn_c = float(fn[cid])

        iou_c = (tp_c / u_c) if u_c > 0 else 0.0
        prec_c = (tp_c / pr_c) if pr_c > 0 else 0.0
        rec_c = (tp_c / gt_c) if gt_c > 0 else 0.0
        f1_c = (2.0 * tp_c / (gt_c + pr_c)) if (gt_c + pr_c) > 0 else 0.0

        val_iou_c = float(val_per_class_iou.get(cname, 0.0))
        gap_iou_c = iou_c - val_iou_c

        if u_c > 0:
            valid_ious.append(iou_c)
        if pr_c > 0:
            valid_precisions.append(prec_c)
        if gt_c > 0:
            valid_recalls.append(rec_c)
        if (gt_c + pr_c) > 0:
            valid_f1s.append(f1_c)

        per_class_rows.append(
            {
                "class_id": cid,
                "class_name": cname,
                "test_iou": iou_c,
                "test_iou_percent": round(iou_c * 100.0, 4),
                "val_iou": val_iou_c,
                "val_iou_percent": round(val_iou_c * 100.0, 4),
                "iou_diff_test_minus_val_percent": round(gap_iou_c * 100.0, 4),
                "precision": prec_c,
                "precision_percent": round(prec_c * 100.0, 4),
                "recall": rec_c,
                "recall_percent": round(rec_c * 100.0, 4),
                "f1_score": f1_c,
                "f1_percent": round(f1_c * 100.0, 4),
                "gt_pixels": int(gt_c),
                "pred_pixels": int(pr_c),
                "tp_pixels": int(tp_c),
                "fp_pixels": int(fp_c),
                "fn_pixels": int(fn_c),
                "union_pixels": int(u_c),
                "gt_share_percent": round(100.0 * gt_c / max(total_valid_pixels, 1), 4),
                "pred_share_percent": round(100.0 * pr_c / max(total_valid_pixels, 1), 4),
            }
        )

    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    mean_precision = float(np.mean(valid_precisions)) if valid_precisions else 0.0
    mean_recall = float(np.mean(valid_recalls)) if valid_recalls else 0.0
    mean_f1 = float(np.mean(valid_f1s)) if valid_f1s else 0.0

    # Frequency-weighted IoU
    fw_iou = (
        float(
            sum(r["gt_pixels"] * r["test_iou"] for r in per_class_rows)
            / max(total_valid_pixels, 1)
        )
        if total_valid_pixels > 0
        else 0.0
    )

    summary_metrics = {
        "total_valid_pixels": total_valid_pixels,
        "pixel_accuracy": pixel_accuracy,
        "pixel_accuracy_percent": round(pixel_accuracy * 100.0, 4),
        "miou": miou,
        "miou_percent": round(miou * 100.0, 4),
        "frequency_weighted_iou": fw_iou,
        "frequency_weighted_iou_percent": round(fw_iou * 100.0, 4),
        "mean_precision": mean_precision,
        "mean_precision_percent": round(mean_precision * 100.0, 4),
        "mean_recall": mean_recall,
        "mean_recall_percent": round(mean_recall * 100.0, 4),
        "mean_f1": mean_f1,
        "mean_f1_percent": round(mean_f1 * 100.0, 4),
        "num_active_classes": len(valid_ious),
    }
    return summary_metrics, per_class_rows


def save_confusion_matrix_csv(cm: np.ndarray, csv_path: Path) -> None:
    """Save the 11x11 confusion matrix to CSV with row and column class headers."""
    header = ["GT \\ Pred"] + [f"{cid}:{name}" for cid, name in enumerate(CLASS_NAMES_11)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for cid, name in enumerate(CLASS_NAMES_11):
            row = [f"{cid}:{name}"] + [int(v) for v in cm[cid].tolist()]
            writer.writerow(row)


def save_per_class_metrics_csv(per_class_rows: List[Dict[str, Any]], csv_path: Path) -> None:
    """Save per-class IoU, precision, recall, F1, and pixel counts to CSV."""
    if not per_class_rows:
        return
    fieldnames = list(per_class_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_class_rows:
            writer.writerow(r)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    per_class_rows: List[Dict[str, Any]],
    test_miou_pct: float,
    val_miou_pct: float,
    out_path: Path,
) -> None:
    """
    Create a 2-panel figure:
    - Left: 11x11 Row-Normalized Confusion Matrix Heatmap (% Recall per GT class + raw counts)
    - Right: Per-Class IoU Bar Chart comparing Validation IoU vs Test IoU (11 classes)
    """
    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
    cm_norm = np.divide(cm.astype(np.float64) * 100.0, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums > 0)

    fig, (ax_cm, ax_bar) = plt.subplots(
        1, 2, figsize=(18.5, 8.2), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )
    fig.suptitle(
        f"CamVid Test Set Evaluation (168 Images, 360×480) — DeepLabV3 MobileNetV3-Large (Best Checkpoint Epoch 22)\n"
        f"Test mIoU: {test_miou_pct:.2f}% | Validation mIoU: {val_miou_pct:.2f}% (Gap: {test_miou_pct - val_miou_pct:+.2f}%)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: 11x11 Confusion Matrix Heatmap
    im = ax_cm.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label("Row-Normalized Percentage (% of GT Class)", fontsize=10)

    tick_labels = [f"[{i}] {name}" for i, name in enumerate(CLASS_NAMES_11)]
    ax_cm.set_xticks(np.arange(NUM_CLASSES))
    ax_cm.set_yticks(np.arange(NUM_CLASSES))
    ax_cm.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax_cm.set_yticklabels(tick_labels, fontsize=9)
    ax_cm.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax_cm.set_ylabel("Ground Truth Class (excl. Void=255)", fontsize=11, fontweight="bold")
    ax_cm.set_title("11×11 Confusion Matrix (Row-Normalized %)", fontsize=11.5, fontweight="bold")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val_pct = cm_norm[i, j]
            raw_cnt = cm[i, j]
            text_color = "white" if val_pct > 50.0 else "black"
            if val_pct >= 1.0 or i == j:
                txt = f"{val_pct:.1f}%\n({raw_cnt // 1000}k)" if raw_cnt >= 1000 else f"{val_pct:.1f}%\n({raw_cnt})"
            elif raw_cnt > 0:
                txt = f"{val_pct:.2f}%"
            else:
                txt = "0"
            ax_cm.text(
                j, i, txt,
                ha="center", va="center",
                color=text_color, fontsize=7.2,
                fontweight="bold" if i == j else "normal",
            )

    # Panel 2: Per-Class IoU Comparison (Validation vs Test)
    y_pos = np.arange(NUM_CLASSES)
    height = 0.38
    val_ious = [r["val_iou_percent"] for r in per_class_rows]
    test_ious = [r["test_iou_percent"] for r in per_class_rows]

    bars_val = ax_bar.barh(y_pos - height / 2, val_ious, height, label=f"Validation IoU (mIoU={val_miou_pct:.2f}%)", color="#6baed6", edgecolor="black", linewidth=0.6)
    bars_test = ax_bar.barh(y_pos + height / 2, test_ious, height, label=f"Test IoU (mIoU={test_miou_pct:.2f}%)", color="#2ca02c", edgecolor="black", linewidth=0.6)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(tick_labels, fontsize=9.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Intersection over Union (%)", fontsize=11, fontweight="bold")
    ax_bar.set_title("Per-Class IoU: Validation (112) vs Test (168)", fontsize=11.5, fontweight="bold")
    ax_bar.set_xlim(0, 108)
    ax_bar.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax_bar.legend(loc="lower right", frameon=True, fontsize=9.5)

    for bar in bars_val:
        w = bar.get_width()
        ax_bar.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=7.8, color="#1f4e78")
    for bar in bars_test:
        w = bar.get_width()
        ax_bar.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=8.0, fontweight="bold", color="#1b5e20")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def build_legend_handles(
    gt_mask: np.ndarray = None,
    pred_mask: np.ndarray = None,
) -> List[mpatches.Patch]:
    """Create legend patches with consistent 11-class RGB palette from config.py."""
    handles: List[mpatches.Patch] = []
    tot_px = int(pred_mask.size) if pred_mask is not None else 0

    for cid, cname in enumerate(CLASS_NAMES_11):
        r, g, b = CLASS_ID_TO_COLOR[cid]
        if gt_mask is not None and pred_mask is not None and tot_px > 0:
            g_pct = 100.0 * float((gt_mask == cid).sum()) / tot_px
            p_pct = 100.0 * float((pred_mask == cid).sum()) / tot_px
            label = f"[{cid:2d}] {cname} (GT:{g_pct:.1f}% | Pred:{p_pct:.1f}%)"
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

    vr, vg, vb = CLASS_ID_TO_COLOR[IGNORE_INDEX]
    if gt_mask is not None and tot_px > 0:
        v_pct = 100.0 * float((gt_mask == IGNORE_INDEX).sum()) / tot_px
        v_label = f"[255] Void/Ignore (GT:{v_pct:.1f}%)"
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


def generate_markdown_report(
    report_path: Path,
    checkpoint_info: Dict[str, Any],
    summary_metrics: Dict[str, Any],
    per_class_rows: List[Dict[str, Any]],
    cm: np.ndarray,
    sample_summaries: List[Dict[str, Any]],
) -> None:
    """Write a comprehensive Markdown evaluation report to `test_evaluation_report.md`."""
    val_miou_pct = checkpoint_info["val_miou_percent"]
    test_miou_pct = summary_metrics["miou_percent"]
    val_acc_pct = checkpoint_info["val_pixel_accuracy_percent"]
    test_acc_pct = summary_metrics["pixel_accuracy_percent"]
    gap_miou = test_miou_pct - val_miou_pct
    gap_acc = test_acc_pct - val_acc_pct

    lines = [
        "# Laporan Evaluasi Final CamVid Test Set (11 Kelas) — DeepLabV3 MobileNetV3-Large",
        "",
        "## 1. Informasi Checkpoint & Konfigurasi Evaluasi",
        f"- **Checkpoint**: `{checkpoint_info['checkpoint_path']}`",
        f"- **Epoch Terbaik**: Epoch `{checkpoint_info['epoch']}` (`stage = {checkpoint_info['stage']}`)",
        f"- **Arsitektur**: `{checkpoint_info['architecture']}` (`num_classes = 11`, `ignore_index = 255`)",
        f"- **Split Dataset**: `test` (**{checkpoint_info['num_test_images']} citra** dari `data/CamVid/test` & `data/CamVid/test_labels`)",
        f"- **Resolusi Evaluasi**: `{checkpoint_info['input_resolution'][0]} × {checkpoint_info['input_resolution'][1]}` ($H \\times W$), normalisasi ImageNet konsisten dengan training",
        f"- **Total Piksel Valid (excl. `ignore_index=255`)**: `{summary_metrics['total_valid_pixels']:,d}` piksel (`{checkpoint_info['ignored_pixels']:,d}` piksel `Void/255` atau `{checkpoint_info['ignored_percent']:.2f}%` diabaikan)",
        f"- **Waktu Evaluasi 168 Citra ({checkpoint_info['device']})**: `{checkpoint_info['evaluation_seconds']:.2f}` detik (`{checkpoint_info['ms_per_image']:.1f}` ms/citra)",
        "",
        "## 2. Ringkasan Metrik Utama (Validation vs Test Set)",
        "",
        "| Metrik | Validation Set (112 Citra) | Test Set (168 Citra) | Selisih (Test - Val) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Pixel Accuracy (excl. 255)** | `{val_acc_pct:.2f}%` | **`{test_acc_pct:.2f}%`** | `{gap_acc:+.2f}%` |",
        f"| **Mean IoU (mIoU, 11 Kelas)** | `{val_miou_pct:.2f}%` | **`{test_miou_pct:.2f}%`** | `{gap_miou:+.2f}%` |",
        f"| **Frequency-Weighted IoU** | — | **`{summary_metrics['frequency_weighted_iou_percent']:.2f}%`** | — |",
        f"| **Mean Precision (11 Kelas)** | — | **`{summary_metrics['mean_precision_percent']:.2f}%`** | — |",
        f"| **Mean Recall (11 Kelas)** | — | **`{summary_metrics['mean_recall_percent']:.2f}%`** | — |",
        f"| **Mean F1-Score / Dice** | — | **`{summary_metrics['mean_f1_percent']:.2f}%`** | — |",
        f"| **Cross-Entropy Loss** | `{checkpoint_info['val_loss']:.4f}` | **`{checkpoint_info['test_loss']:.4f}`** | `{checkpoint_info['test_loss'] - checkpoint_info['val_loss']:+.4f}` |",
        "",
        "## 3. Tabel Metrik Per-Kelas pada Test Set (11 Kelas)",
        "",
        "| ID | Nama Kelas | Val IoU (%) | Test IoU (%) | Selisih IoU | Precision (%) | Recall (%) | F1-Score (%) | GT Pixels | Pred Pixels |",
        "| :-: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in per_class_rows:
        lines.append(
            f"| `{r['class_id']}` | **{r['class_name']}** | `{r['val_iou_percent']:.2f}%` | "
            f"**`{r['test_iou_percent']:.2f}%`** | `{r['iou_diff_test_minus_val_percent']:+.2f}%` | "
            f"`{r['precision_percent']:.2f}%` | `{r['recall_percent']:.2f}%` | "
            f"`{r['f1_percent']:.2f}%` | `{r['gt_pixels']:,d}` | `{r['pred_pixels']:,d}` |"
        )

    lines.extend(
        [
            "",
            "## 4. Analisis Generalization Gap (Validation vs Test Set)",
            f"- **Selisih Global**: Validation mIoU adalah **`{val_miou_pct:.2f}%`** sedangkan Test mIoU adalah **`{test_miou_pct:.2f}%`** (selisih **`{gap_miou:+.2f}%`**), dan Validation Pixel Accuracy adalah **`{val_acc_pct:.2f}%`** sedangkan Test Pixel Accuracy adalah **`{test_acc_pct:.2f}%`** (selisih **`{gap_acc:+.2f}%`**).",
            "- **Faktor Penyebab Perbedaan Per-Kelas**:",
            "  1. **Pemilihan Model Berdasarkan Validation mIoU**: Checkpoint Epoch 22 dipilih ketika validation mIoU mencapai puncaknya (`65.43%`), sehingga secara statistik wajar terdapat sedikit *optimism bias* pada validation set dibanding unseen test set.",
            "  2. **Perbedaan Komposisi Sekuens & Pencahayaan pada Test Split**: Pada split CamVid, beberapa sekuens memiliki variasi kondisi pencahayaan/bayangan dan proporsi objek kecil (`Pole`, `SignSymbol`, `Pedestrian`, `Bicyclist`, `Fence`) yang berbeda antara `val` (`112` citra) dan `test` (`168` citra).",
            "  3. **Trade-off Class Weighting ENet pada Kelas Minor (Recall Tinggi vs Precision Lebih Rendah)**: Karena *class weights* ENet memberikan bobot tinggi pada kelas langka (`Bicyclist = 2.00`, `Pedestrian = 1.87`, `Pole = 1.67`, `SignSymbol = 1.61`), model memiliki **Recall yang tinggi** pada kelas-kelas kecil tersebut, namun menghasilkan sejumlah *false positive* di tepi objek (tercermin pada Precision yang lebih rendah dibanding Recall untuk `Pole`, `SignSymbol`, dan `Pedestrian`).",
            "",
            "## 5. Ringkasan 15 Sampel Visualisasi Test Set",
            "",
            "| No | Test Idx | Nama File | Pixel Acc (%) | Sample mIoU (%) | GT Void 255 (%) | Kelas Terdeteksi |",
            "| :-: | :-: | :--- | ---: | ---: | ---: | :--- |",
        ]
    )

    for s in sample_summaries:
        lines.append(
            f"| `{s['sample_number']:02d}` | `{s['test_index']}` | `{s['filename']}` | "
            f"`{s['pixel_accuracy']*100:.2f}%` | `{s['miou']*100:.2f}%` | "
            f"`{s['ignored_gt_percent']:.2f}%` | {', '.join(s['detected_classes'])} |"
        )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_test_evaluation(
    checkpoint_path: Path = MODELS_DIR / "camvid11_full_25epochs_best.pth",
    output_dir: Path = RESULTS_DIR / "test_evaluation",
    input_size: Tuple[int, int] = (360, 480),
    batch_size: int = 4,
    num_visual_samples: int = 15,
) -> Dict[str, Any]:
    """Run full evaluation on all 168 CamVid test images and generate all artifacts."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(output_dir)
    vis_dir = output_dir / "test_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, meta = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    # 1. Verify checkpoint identity (must be Epoch 22 best checkpoint from camvid11_full_25epochs)
    ckpt_epoch = int(meta.get("epoch", -1))
    ckpt_best_miou = float(meta.get("best_miou", 0.0))
    ckpt_stage = str(meta.get("stage", meta.get("metrics", {}).get("stage", "finetune")))
    ckpt_metrics = meta.get("metrics", {})
    val_miou = float(ckpt_metrics.get("val_miou", ckpt_best_miou))
    val_acc = float(ckpt_metrics.get("val_pixel_accuracy", 0.0))
    val_loss = float(ckpt_metrics.get("val_loss", 0.0))
    val_per_class_iou = ckpt_metrics.get("val_per_class_iou", {})

    assert ckpt_epoch == 22, f"Expected best checkpoint from epoch 22, got epoch={ckpt_epoch}"
    assert abs(ckpt_best_miou - 0.6543315666561222) < 1e-4, (
        f"Unexpected best_miou in checkpoint: {ckpt_best_miou}"
    )

    # 2. Load entire 168-image test split with exact training preprocessing
    test_dataset = CamVidDataset(
        root_dir=CAMVID_DIR,
        split="test",
        target_size=input_size,
        imagenet_normalize=True,
        transform=None,
        return_dict=True,
    )
    test_dataset_unnorm = CamVidDataset(
        root_dir=CAMVID_DIR,
        split="test",
        target_size=input_size,
        imagenet_normalize=False,
        transform=None,
        return_dict=True,
    )
    assert test_dataset.split == "test"
    assert len(test_dataset) == EXPECTED_SPLIT_COUNTS["test"] == 168, (
        f"Expected 168 test images, got {len(test_dataset)}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Reconstruct loss criterion with the exact class weights stored in the checkpoint
    ckpt_weights = meta.get("class_weights") or ckpt_metrics.get("class_weights")
    weight_tensor = (
        torch.tensor(ckpt_weights, dtype=torch.float32, device=device)
        if ckpt_weights is not None
        else None
    )
    criterion = create_loss_function(class_weights=weight_tensor, ignore_index=IGNORE_INDEX)

    print("=== EVALUASI FINAL CAMVID TEST SET (168 CITRA) ===")
    print(f"  Checkpoint          : {checkpoint_path}")
    print(f"  Verified Epoch      : {ckpt_epoch} (Stage: {ckpt_stage})")
    print(f"  Validation Best mIoU: {val_miou * 100:.2f}% (Val Pixel Acc: {val_acc * 100:.2f}%)")
    print(f"  Test Set Samples    : {len(test_dataset)} images (360x480)")
    print(f"  Device              : {device}")

    # 3. Evaluate all 168 test images
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    total_loss_sum = 0.0
    total_images_evaluated = 0
    total_ignored_pixels = 0
    total_all_pixels = 0

    # Cache predictions for the 15 visual samples so we don't have to run inference twice
    vis_indices = select_evenly_spaced_indices(len(test_dataset), num_samples=num_visual_samples)
    vis_index_set = set(vis_indices)
    cached_vis_preds: Dict[int, np.ndarray] = {}

    t0 = time.perf_counter()
    with torch.no_grad():
        sample_cursor = 0
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            b_size = int(images.size(0))

            outputs = model(images, return_dict=True)
            main_logits = outputs["out"]
            aux_logits = outputs.get("aux")
            main_loss = criterion(main_logits, masks)
            if aux_logits is not None:
                batch_loss = main_loss + 0.4 * criterion(aux_logits, masks)
            else:
                batch_loss = main_loss

            total_loss_sum += float(batch_loss.item()) * b_size
            preds = torch.argmax(main_logits, dim=1)  # (B, H, W)
            metrics.update(preds, masks)

            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy().astype(np.uint8)
            total_ignored_pixels += int((masks_np == IGNORE_INDEX).sum())
            total_all_pixels += int(masks_np.size)

            for i in range(b_size):
                global_idx = sample_cursor + i
                if global_idx in vis_index_set:
                    cached_vis_preds[global_idx] = preds_np[i]

            sample_cursor += b_size
            total_images_evaluated += b_size

    eval_seconds = time.perf_counter() - t0
    assert total_images_evaluated == 168, f"Expected 168 evaluated images, got {total_images_evaluated}"
    test_loss = total_loss_sum / total_images_evaluated

    raw_computed = metrics.compute()
    cm = raw_computed["confusion_matrix"]
    assert cm.shape == (11, 11)
    assert int(cm.sum()) + total_ignored_pixels == total_all_pixels, (
        "Valid pixels in confusion matrix + ignored pixels (255) must equal total pixels!"
    )

    summary_metrics, per_class_rows = compute_detailed_metrics_from_cm(
        cm=cm,
        val_per_class_iou=val_per_class_iou,
    )

    # 4. Save confusion matrix CSV, per-class metrics CSV, and heatmap PNG
    cm_csv_path = output_dir / "test_confusion_matrix.csv"
    per_class_csv_path = output_dir / "test_per_class_metrics.csv"
    cm_png_path = output_dir / "test_confusion_matrix.png"

    save_confusion_matrix_csv(cm, cm_csv_path)
    save_per_class_metrics_csv(per_class_rows, per_class_csv_path)
    plot_confusion_matrix_heatmap(
        cm=cm,
        per_class_rows=per_class_rows,
        test_miou_pct=summary_metrics["miou_percent"],
        val_miou_pct=round(val_miou * 100.0, 4),
        out_path=cm_png_path,
    )

    # 5. Generate 15 test sample visualizations + 15-sample contact sheet
    contact_fig, contact_axes = plt.subplots(
        len(vis_indices), 4, figsize=(18, 3.1 * len(vis_indices) + 1.8)
    )
    contact_fig.suptitle(
        f"CamVid Test Set Visual Evaluation — {len(vis_indices)} Representative Samples (360×480)\n"
        f"Checkpoint: {checkpoint_path.name} (Epoch 22) | Global Test mIoU: {summary_metrics['miou_percent']:.2f}% | "
        f"Test Pixel Acc: {summary_metrics['pixel_accuracy_percent']:.2f}%",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    sample_summaries: List[Dict[str, Any]] = []
    saved_vis_files: List[str] = []

    for row_idx, ds_idx in enumerate(vis_indices):
        item_norm = test_dataset[ds_idx]
        item_unnorm = test_dataset_unnorm[ds_idx]
        stem = item_norm["stem"]
        filename = f"{stem}.png"

        orig_rgb = (
            item_unnorm["image"].permute(1, 2, 0).cpu().numpy() * 255.0
        ).round().astype(np.uint8)
        gt_mask = item_norm["mask"].cpu().numpy().astype(np.uint8)
        pred_mask = cached_vis_preds[ds_idx]

        gt_rgb = mask_to_rgb(gt_mask)
        pred_rgb = mask_to_rgb(pred_mask)
        ov_rgb = overlay_mask(orig_rgb, pred_mask, alpha=0.45)

        s_metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
        s_metrics.update(pred_mask, gt_mask)
        s_res = s_metrics.compute()

        ignored_pct = 100.0 * float((gt_mask == IGNORE_INDEX).sum()) / float(gt_mask.size)
        present_pred_ids = sorted(set(np.unique(pred_mask).tolist()))
        detected_names = [CLASS_NAMES_11[cid] for cid in present_pred_ids if 0 <= cid < NUM_CLASSES]

        # Individual 4-panel figure
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.9))
        fig.suptitle(
            f"Test Sample #{row_idx + 1:02d}: test/{filename} (Index {ds_idx}) — "
            f"Pixel Acc: {s_res['pixel_accuracy']*100:.2f}% | Sample mIoU: {s_res['miou']*100:.2f}%",
            fontsize=12,
            fontweight="bold",
            y=0.97,
        )

        axes[0].imshow(orig_rgb)
        axes[0].set_title(f"1. Original Image\n({input_size[1]}×{input_size[0]})", fontsize=10.5, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_rgb)
        axes[1].set_title(f"2. Ground Truth (11 Classes)\n(Void/255: {ignored_pct:.1f}%)", fontsize=10.5, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(pred_rgb)
        axes[2].set_title(f"3. Predicted Segmentation\n({len(detected_names)} kelas terdeteksi)", fontsize=10.5, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(ov_rgb)
        axes[3].set_title("4. Overlay Prediction\npada Original Image", fontsize=10.5, fontweight="bold")
        axes[3].axis("off")

        handles = build_legend_handles(gt_mask=gt_mask, pred_mask=pred_mask)
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            fontsize=8.5,
            frameon=True,
            title="Palet Warna 11 Kelas & Perbandingan Distribusi Piksel (Ground Truth vs Test Prediction)",
            title_fontsize=9.5,
        )
        fig.tight_layout(rect=[0, 0.17, 1, 0.93])

        indiv_out = vis_dir / f"test_sample_{row_idx + 1:02d}_{stem}.png"
        fig.savefig(indiv_out, dpi=140)
        plt.close(fig)
        saved_vis_files.append(str(indiv_out))

        # Populate row in 15-sample contact sheet
        ax_row = contact_axes[row_idx]
        ax_row[0].imshow(orig_rgb)
        ax_row[0].set_title(f"[{row_idx + 1:02d}] Original: {filename}", fontsize=9.2, fontweight="bold")
        ax_row[0].axis("off")

        ax_row[1].imshow(gt_rgb)
        ax_row[1].set_title(f"[{row_idx + 1:02d}] GT 11-Class (Void: {ignored_pct:.1f}%)", fontsize=9.2, fontweight="bold")
        ax_row[1].axis("off")

        ax_row[2].imshow(pred_rgb)
        ax_row[2].set_title(
            f"[{row_idx + 1:02d}] Pred ({len(detected_names)} cls | Acc: {s_res['pixel_accuracy']*100:.1f}%)",
            fontsize=9.2,
            fontweight="bold",
        )
        ax_row[2].axis("off")

        ax_row[3].imshow(ov_rgb)
        ax_row[3].set_title(
            f"[{row_idx + 1:02d}] Overlay (mIoU: {s_res['miou']*100:.1f}%)",
            fontsize=9.2,
            fontweight="bold",
        )
        ax_row[3].axis("off")

        sample_summaries.append(
            {
                "sample_number": row_idx + 1,
                "test_index": ds_idx,
                "filename": filename,
                "stem": stem,
                "pixel_accuracy": s_res["pixel_accuracy"],
                "miou": s_res["miou"],
                "ignored_gt_percent": round(ignored_pct, 2),
                "detected_classes": detected_names,
                "output_path": str(indiv_out),
            }
        )

    shared_handles = build_legend_handles()
    contact_fig.legend(
        handles=shared_handles,
        loc="lower center",
        ncol=6,
        fontsize=10,
        frameon=True,
        title="Palet Warna Konsisten 11 Kelas CamVid (ID 0–10) & Void/Ignore (255 pada GT)",
        title_fontsize=11,
    )
    contact_fig.tight_layout(rect=[0, 0.018, 1, 0.983])
    contact_sheet_path = output_dir / "test_visualization_contact_sheet.png"
    contact_fig.savefig(contact_sheet_path, dpi=125)
    plt.close(contact_fig)

    # 6. Save test_metrics.json and test_evaluation_report.md
    checkpoint_info = {
        "checkpoint_path": str(checkpoint_path),
        "architecture": meta.get("architecture", "deeplabv3_mobilenet_v3_large"),
        "epoch": ckpt_epoch,
        "stage": ckpt_stage,
        "val_miou": val_miou,
        "val_miou_percent": round(val_miou * 100.0, 4),
        "val_pixel_accuracy": val_acc,
        "val_pixel_accuracy_percent": round(val_acc * 100.0, 4),
        "val_loss": val_loss,
        "test_loss": test_loss,
        "num_test_images": total_images_evaluated,
        "input_resolution": list(input_size),
        "ignore_index": IGNORE_INDEX,
        "total_pixels_all": total_all_pixels,
        "ignored_pixels": total_ignored_pixels,
        "ignored_percent": round(100.0 * total_ignored_pixels / max(total_all_pixels, 1), 4),
        "device": str(device),
        "evaluation_seconds": round(eval_seconds, 2),
        "ms_per_image": round(1000.0 * eval_seconds / total_images_evaluated, 2),
    }

    metrics_json_path = output_dir / "test_metrics.json"
    full_json_payload = {
        "checkpoint_info": checkpoint_info,
        "test_summary_metrics": summary_metrics,
        "validation_vs_test_comparison": {
            "val_miou_percent": round(val_miou * 100.0, 4),
            "test_miou_percent": summary_metrics["miou_percent"],
            "miou_gap_test_minus_val_percent": round(summary_metrics["miou_percent"] - val_miou * 100.0, 4),
            "val_pixel_accuracy_percent": round(val_acc * 100.0, 4),
            "test_pixel_accuracy_percent": summary_metrics["pixel_accuracy_percent"],
            "pixel_accuracy_gap_test_minus_val_percent": round(
                summary_metrics["pixel_accuracy_percent"] - val_acc * 100.0, 4
            ),
            "val_loss": round(val_loss, 6),
            "test_loss": round(test_loss, 6),
        },
        "per_class_metrics": per_class_rows,
        "confusion_matrix_11x11": cm.tolist(),
        "visualized_samples": sample_summaries,
        "artifacts": {
            "test_metrics_json": str(metrics_json_path),
            "test_per_class_metrics_csv": str(per_class_csv_path),
            "test_confusion_matrix_csv": str(cm_csv_path),
            "test_confusion_matrix_png": str(cm_png_path),
            "test_visualizations_dir": str(vis_dir),
            "test_visualization_contact_sheet_png": str(contact_sheet_path),
            "test_evaluation_report_md": str(output_dir / "test_evaluation_report.md"),
        },
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(full_json_payload, f, indent=2)

    report_md_path = output_dir / "test_evaluation_report.md"
    generate_markdown_report(
        report_path=report_md_path,
        checkpoint_info=checkpoint_info,
        summary_metrics=summary_metrics,
        per_class_rows=per_class_rows,
        cm=cm,
        sample_summaries=sample_summaries,
    )

    # Print summary to console
    print("\n=== HASIL AKHIR EVALUASI TEST SET (168 CITRA) ===")
    print(f"  Test Pixel Accuracy (excl. 255) : {summary_metrics['pixel_accuracy_percent']:.2f}% (Val: {val_acc*100:.2f}%)")
    print(f"  Test Mean IoU (mIoU, 11 Kelas)  : {summary_metrics['miou_percent']:.2f}% (Val: {val_miou*100:.2f}%)")
    print(f"  Test Frequency-Weighted IoU     : {summary_metrics['frequency_weighted_iou_percent']:.2f}%")
    print(f"  Test Mean Precision             : {summary_metrics['mean_precision_percent']:.2f}%")
    print(f"  Test Mean Recall                : {summary_metrics['mean_recall_percent']:.2f}%")
    print(f"  Test Mean F1-Score              : {summary_metrics['mean_f1_percent']:.2f}%")
    print(f"  Test Cross-Entropy Loss         : {test_loss:.4f} (Val: {val_loss:.4f})")
    print(f"  Total Piksel Valid              : {summary_metrics['total_valid_pixels']:,d} (Void/255 diabaikan: {total_ignored_pixels:,d} px / {checkpoint_info['ignored_percent']:.2f}%)")
    print(f"  Waktu Evaluasi 168 Citra        : {eval_seconds:.2f} detik ({checkpoint_info['ms_per_image']:.1f} ms/citra)")

    print("\n=== TABEL PER-KELAS TEST SET (11 KELAS) ===")
    print(
        f"{'ID':>2} | {'Nama Kelas':<12} | {'Val IoU':>8} | {'Test IoU':>8} | {'Gap':>7} | "
        f"{'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'GT Pixels':>11} | {'Pred Pixels':>11}"
    )
    print("-" * 106)
    for r in per_class_rows:
        print(
            f"{r['class_id']:2d} | {r['class_name']:<12} | {r['val_iou_percent']:7.2f}% | "
            f"{r['test_iou_percent']:7.2f}% | {r['iou_diff_test_minus_val_percent']:+6.2f}% | "
            f"{r['precision_percent']:6.2f}% | {r['recall_percent']:6.2f}% | {r['f1_percent']:6.2f}% | "
            f"{r['gt_pixels']:11,d} | {r['pred_pixels']:11,d}"
        )

    print("\n=== ARTEFAK DISIMPAN DI results/test_evaluation/ ===")
    for k, v in full_json_payload["artifacts"].items():
        print(f"  - {k:<36}: {v}")

    return full_json_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Final evaluation on CamVid test set (168 images).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(MODELS_DIR / "camvid11_full_25epochs_best.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR / "test_evaluation"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-visual-samples", type=int, default=15)
    args = parser.parse_args()

    run_test_evaluation(
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        num_visual_samples=args.num_visual_samples,
    )


if __name__ == "__main__":
    main()
