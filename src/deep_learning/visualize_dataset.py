"""Visualize CamVid 11-class ground truth masks, overlays, and class legend."""

import os
from pathlib import Path
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

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
    NUM_CLASSES,
    RESULTS_DIR,
)
from src.deep_learning.dataset import CamVidDataset


def compute_sample_class_distribution(mask: np.ndarray) -> List[Tuple[int, str, Tuple[int, int, int], int, float]]:
    """Compute pixel count and percentage for each class ID present in a 2D mask."""
    total_px = mask.size
    unique_ids, counts = np.unique(mask, return_counts=True)
    items = []
    for cid, cnt in zip(unique_ids.tolist(), counts.tolist()):
        name = CLASS_ID_TO_NAME.get(cid, f"Unknown({cid})")
        color = CLASS_ID_TO_COLOR.get(cid, (0, 0, 0))
        pct = 100.0 * cnt / total_px
        items.append((cid, name, color, cnt, pct))
    # Sort by pixel count descending
    items.sort(key=lambda x: -x[3])
    return items


def build_11class_legend_handles(
    mask: np.ndarray = None,
    include_all_classes: bool = True,
) -> List[mpatches.Patch]:
    """Create Matplotlib legend patches for all 11 classes + Void/Ignore (255)."""
    pct_by_id = {}
    if mask is not None:
        total_px = mask.size
        unique_ids, counts = np.unique(mask, return_counts=True)
        for cid, cnt in zip(unique_ids.tolist(), counts.tolist()):
            pct_by_id[cid] = 100.0 * cnt / total_px

    ordered_ids = list(range(NUM_CLASSES)) + [IGNORE_INDEX]
    handles = []
    for cid in ordered_ids:
        if not include_all_classes and cid not in pct_by_id:
            continue
        r, g, b = CLASS_ID_TO_COLOR[cid]
        name = CLASS_ID_TO_NAME[cid]
        if mask is not None:
            pct = pct_by_id.get(cid, 0.0)
            label = f"[{cid:2d}] {name} ({pct:.1f}%)" if cid != IGNORE_INDEX else f"[255] {name} ({pct:.1f}%)"
        else:
            label = f"[{cid:2d}] {name}" if cid != IGNORE_INDEX else f"[255] {name}"
        handles.append(
            mpatches.Patch(
                facecolor=(r / 255.0, g / 255.0, b / 255.0),
                edgecolor="black",
                linewidth=0.6,
                label=label,
            )
        )
    return handles


def generate_visualizations(
    output_dir: Path = RESULTS_DIR / "camvid_11classes_inspection",
) -> List[Path]:
    """
    Generate individual and overview visualizations for 6 representative samples
    (5 standard samples across train/val/test + the special Seq05VD_f02610 test sample
    containing 175 non-palette boundary pixels).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train": CamVidDataset(CAMVID_DIR, split="train"),
        "val": CamVidDataset(CAMVID_DIR, split="val"),
        "test": CamVidDataset(CAMVID_DIR, split="test"),
    }

    # Find index of Seq05VD_f02610 in test split to explicitly verify the 175 non-palette pixels sample
    seq05_idx = 0
    for idx, (_, _, stem) in enumerate(datasets["test"].samples):
        if stem == "Seq05VD_f02610":
            seq05_idx = idx
            break

    sample_specs = [
        ("train", 0),            # 0001TP_006690
        ("train", 120),          # 0006R0_f03210
        ("train", 260),          # 0016E5_07997
        ("val", 30),             # 0006R0_f02790
        ("test", seq05_idx),     # Seq05VD_f02610 (contains the 175 non-palette pixels!)
    ]

    saved_files: List[Path] = []
    fig_all, axes_all = plt.subplots(len(sample_specs), 3, figsize=(16, 21))
    fig_all.suptitle(
        "CamVid 11-Class Semantic Segmentation Mapping Validation (5 Samples)",
        fontsize=15,
        fontweight="bold",
        y=0.992,
    )

    for row_idx, (split_name, sample_idx) in enumerate(sample_specs):
        ds = datasets[split_name]
        sample = ds.get_raw_sample(sample_idx)
        stem = sample["stem"]
        image = sample["image"]
        mask = sample["mask"]
        mapped_rgb = sample["mapped_label_rgb"]
        overlay = sample["overlay"]

        dist = compute_sample_class_distribution(mask)
        n_active_classes = sum(1 for cid, _, _, cnt, _ in dist if cid != IGNORE_INDEX and cnt > 0)

        # 1. Save individual high-resolution figure with full 11-class + Void legend
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
        axes[0].imshow(image)
        axes[0].set_title(
            f"Citra Asli ({split_name}/{stem}.png)\nResolusi: {image.shape[1]}x{image.shape[0]}",
            fontsize=11,
            fontweight="bold",
        )
        axes[0].axis("off")

        axes[1].imshow(mapped_rgb)
        axes[1].set_title(
            f"Ground Truth 11 Kelas ({stem}_L.png)\n{n_active_classes}/11 kelas aktif (+ Void=255)",
            fontsize=11,
            fontweight="bold",
        )
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Overlay Citra & Ground Truth 11 Kelas\n(Alpha Blend 55% Citra / 45% Label)",
            fontsize=11,
            fontweight="bold",
        )
        axes[2].axis("off")

        handles = build_11class_legend_handles(mask, include_all_classes=True)
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=6,
            fontsize=8.8,
            frameon=True,
            title="Legenda 11 Kelas Target (Class ID 0–10) & Void/Ignore (255) beserta Persentase Piksel",
            title_fontsize=9.5,
        )
        fig.tight_layout(rect=[0, 0.15, 1, 0.96])

        out_path = output_dir / f"sample_{row_idx + 1}_{split_name}_{stem}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        saved_files.append(out_path)
        print(f"Saved individual 11-class sample visualization: {out_path}")

        # 2. Populate row in combined overview figure
        axes_all[row_idx, 0].imshow(image)
        axes_all[row_idx, 0].set_title(
            f"[{row_idx + 1}] Citra Asli ({split_name}): {stem}.png",
            fontsize=10,
            fontweight="bold",
        )
        axes_all[row_idx, 0].axis("off")

        axes_all[row_idx, 1].imshow(mapped_rgb)
        axes_all[row_idx, 1].set_title(
            f"[{row_idx + 1}] GT 11 Kelas: {stem}_L.png ({n_active_classes} kelas)",
            fontsize=10,
            fontweight="bold",
        )
        axes_all[row_idx, 1].axis("off")

        axes_all[row_idx, 2].imshow(overlay)
        axes_all[row_idx, 2].set_title(
            f"[{row_idx + 1}] Overlay 11 Kelas: {stem}",
            fontsize=10,
            fontweight="bold",
        )
        axes_all[row_idx, 2].axis("off")

    overview_handles = build_11class_legend_handles(mask=None, include_all_classes=True)
    fig_all.legend(
        handles=overview_handles,
        loc="lower center",
        ncol=6,
        fontsize=9.5,
        frameon=True,
        title="Palet Warna 11 Kelas Semantic Segmentation CamVid (ID 0–10) & Void/Ignore (255)",
        title_fontsize=10.5,
    )
    fig_all.tight_layout(rect=[0, 0.045, 1, 0.982])
    overview_path = output_dir / "camvid_11classes_5_samples_overview.png"
    fig_all.savefig(overview_path, dpi=140)
    plt.close(fig_all)
    saved_files.append(overview_path)
    print(f"Saved combined 5-sample 11-class overview: {overview_path}")

    return saved_files


if __name__ == "__main__":
    generate_visualizations()
