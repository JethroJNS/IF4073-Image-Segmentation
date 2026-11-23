"""Generate training curves (Loss, mIoU/Accuracy, and Learning Rate schedule) from history CSV."""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.config import RESULTS_DIR


def load_history_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Read training history CSV and parse numeric fields."""
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            parsed: Dict[str, Any] = {}
            for k, v in r.items():
                if k == "stage":
                    parsed[k] = v
                elif k in ("epoch", "stage_epoch"):
                    parsed[k] = int(float(v))
                else:
                    parsed[k] = float(v)
            rows.append(parsed)
    return rows


def plot_training_curves(
    csv_path: Path,
    output_dir: Path,
    experiment_name: str = "camvid11_full_25epochs",
) -> Dict[str, Path]:
    """
    Create and save:
    1. Train Loss vs Validation Loss plot
    2. Validation mIoU & Pixel Accuracy plot
    3. Learning Rate (Backbone & Head) schedule plot
    4. Combined 3-panel dashboard plot
    """
    rows = load_history_csv(csv_path)
    if not rows:
        raise ValueError(f"History CSV is empty: {csv_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    train_miou = [r["train_miou"] * 100.0 for r in rows]
    val_miou = [r["val_miou"] * 100.0 for r in rows]
    train_acc = [r["train_pixel_acc"] * 100.0 for r in rows]
    val_acc = [r["val_pixel_acc"] * 100.0 for r in rows]
    bb_lr = [r["backbone_lr"] for r in rows]
    hd_lr = [r["head_lr"] for r in rows]

    best_idx = int(np.argmax(val_miou))
    best_ep = epochs[best_idx]
    best_miou_val = val_miou[best_idx]

    warmup_epochs = sum(1 for r in rows if r["stage"] == "warmup")

    # 1. Loss Curve
    loss_path = output_dir / f"{experiment_name}_loss_curve.png"
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4", linewidth=2, markersize=5)
    ax.plot(epochs, val_loss, "s-", label="Validation Loss", color="#d62728", linewidth=2, markersize=5)
    if 0 < warmup_epochs < len(epochs):
        ax.axvline(warmup_epochs + 0.5, color="gray", linestyle="--", linewidth=1.3, label="Warm-up -> Fine-tune")
    ax.axvline(best_ep, color="#2ca02c", linestyle=":", linewidth=1.5, label=f"Best Val mIoU (Epoch {best_ep})")
    ax.set_title(f"Train vs Validation Loss ({experiment_name})", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss (Total)", fontsize=11)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(loss_path, dpi=140)
    plt.close(fig)

    # 2. mIoU & Pixel Accuracy Curve
    miou_path = output_dir / f"{experiment_name}_miou_curve.png"
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(epochs, val_miou, "s-", label="Validation mIoU (%)", color="#2ca02c", linewidth=2.2, markersize=5)
    ax.plot(epochs, train_miou, "o--", label="Train mIoU (%)", color="#98df8a", linewidth=1.8, markersize=4)
    ax.plot(epochs, val_acc, "^-", label="Validation Pixel Acc (%)", color="#9467bd", linewidth=2, markersize=5)
    ax.plot(epochs, train_acc, "v--", label="Train Pixel Acc (%)", color="#c5b0d5", linewidth=1.6, markersize=4)
    ax.scatter([best_ep], [best_miou_val], color="gold", edgecolor="black", s=110, zorder=6,
               label=f"Best Val mIoU: {best_miou_val:.2f}% (Ep {best_ep})")
    if 0 < warmup_epochs < len(epochs):
        ax.axvline(warmup_epochs + 0.5, color="gray", linestyle="--", linewidth=1.3, label="Warm-up -> Fine-tune")
    ax.set_title(f"Validation mIoU & Pixel Accuracy per Epoch ({experiment_name})", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9.2)
    fig.tight_layout()
    fig.savefig(miou_path, dpi=140)
    plt.close(fig)

    # 3. Learning Rate Curve
    lr_path = output_dir / f"{experiment_name}_lr_curve.png"
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(epochs, hd_lr, "s-", label="Classifier Head LR", color="#ff7f0e", linewidth=2, markersize=5)
    ax.plot(epochs, bb_lr, "o-", label="Backbone LR", color="#17becf", linewidth=2, markersize=5)
    if 0 < warmup_epochs < len(epochs):
        ax.axvline(warmup_epochs + 0.5, color="gray", linestyle="--", linewidth=1.3, label="Warm-up -> Fine-tune")
    ax.set_title(f"Learning Rate Schedule ({experiment_name})", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(lr_path, dpi=140)
    plt.close(fig)

    # 4. Combined 3-Panel Dashboard
    combined_path = output_dir / f"{experiment_name}_curves_combined.png"
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    fig.suptitle(
        f"DeepLabV3 MobileNetV3-Large CamVid 11-Class Full Training ({experiment_name}) — "
        f"Best Val mIoU: {best_miou_val:.2f}% @ Epoch {best_ep}",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Loss
    axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4", linewidth=2, markersize=4)
    axes[0].plot(epochs, val_loss, "s-", label="Val Loss", color="#d62728", linewidth=2, markersize=4)
    if 0 < warmup_epochs < len(epochs):
        axes[0].axvline(warmup_epochs + 0.5, color="gray", linestyle="--", linewidth=1.2, label="Stage 1 -> 2")
    axes[0].axvline(best_ep, color="#2ca02c", linestyle=":", linewidth=1.4, label=f"Best Ep {best_ep}")
    axes[0].set_title("1. Train & Validation Loss", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8.5)

    # Panel 2: mIoU & Accuracy
    axes[1].plot(epochs, val_miou, "s-", label="Val mIoU (%)", color="#2ca02c", linewidth=2.2, markersize=4)
    axes[1].plot(epochs, train_miou, "o--", label="Train mIoU (%)", color="#7fbc41", linewidth=1.6, markersize=3.5)
    axes[1].plot(epochs, val_acc, "^-", label="Val Pixel Acc (%)", color="#9467bd", linewidth=2, markersize=4)
    axes[1].scatter([best_ep], [best_miou_val], color="gold", edgecolor="black", s=95, zorder=6,
                    label=f"Best: {best_miou_val:.2f}%")
    if 0 < warmup_epochs < len(epochs):
        axes[1].axvline(warmup_epochs + 0.5, color="gray", linestyle="--", linewidth=1.2)
    axes[1].set_title("2. Validation mIoU & Pixel Accuracy", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8.5)

    # Panel 3: Learning Rate
    axes[2].plot(epochs, hd_lr, "s-", label="Head LR", color="#ff7f0e", linewidth=2, markersize=4)
    axes[2].plot(epochs, bb_lr, "o-", label="Backbone LR", color="#17becf", linewidth=2, markersize=4)
    if 0 < warmup_epochs < len(epochs):
        axes[2].axvline(warmup_epochs + 0.5, color="gray", linestyle="--", linewidth=1.2, label="Stage 1 -> 2")
    axes[2].set_title("3. Learning Rate Schedule", fontsize=11.5, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(combined_path, dpi=140)
    plt.close(fig)

    return {
        "loss_curve": loss_path,
        "miou_curve": miou_path,
        "lr_curve": lr_path,
        "combined_curve": combined_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training history curves from CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(RESULTS_DIR / "training" / "camvid11_full_25epochs_history.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR / "training"),
    )
    parser.add_argument("--experiment-name", type=str, default="camvid11_full_25epochs")
    args = parser.parse_args()

    paths = plot_training_curves(
        csv_path=Path(args.csv),
        output_dir=Path(args.output_dir),
        experiment_name=args.experiment_name,
    )
    for k, p in paths.items():
        print(f"Saved {k}: {p}")


if __name__ == "__main__":
    main()
