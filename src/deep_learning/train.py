"""Two-stage transfer learning training pipeline for CamVid 11-class Semantic Segmentation."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when invoked as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.config import CLASS_NAMES_11, IGNORE_INDEX, NUM_CLASSES
from src.deep_learning.dataset import CamVidDataset, rgb_to_mask
from src.deep_learning.metrics import SegmentationMetrics
from src.deep_learning.model import (
    CamVidDeepLabV3,
    ModelConfig,
    create_deeplabv3_mobilenet_v3_large,
    create_loss_function,
    get_device,
    load_checkpoint,
    save_checkpoint,
)
from src.deep_learning.training_config import TrainingConfig


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_spec: str = "auto") -> torch.device:
    """Resolve device specification ('auto', 'cpu', 'cuda') safely."""
    spec = device_spec.lower().strip()
    if spec == "auto":
        return get_device(prefer_cuda=True)
    if spec == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


class RandomHorizontalFlipTransform:
    """Lightweight joint random horizontal flip for (image_rgb, mask) NumPy arrays."""

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(
        self, image_rgb: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.prob:
            image_rgb = np.ascontiguousarray(image_rgb[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
        return image_rgb, mask


def compute_class_weights_from_counts(
    class_pixel_counts: Union[np.ndarray, List[int]],
    num_classes: int = NUM_CLASSES,
    method: str = "enet",
    min_weight: float = 0.25,
    max_weight: float = 5.0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Compute 11-element class weight tensor from per-class pixel counts.

    Guarantees:
    - Always returns a 1D float32 tensor of shape `(11,)`.
    - Handles classes with 0 samples safely without division-by-zero, Inf, or NaN
      (zero-sample classes receive neutral weight 1.0).
    - Active class weights are normalized to have mean 1.0 and clamped to `[min_weight, max_weight]`.
    """
    counts = np.asarray(class_pixel_counts, dtype=np.float64)
    if counts.shape != (num_classes,):
        raise ValueError(
            f"Expected class_pixel_counts of shape ({num_classes},), got {counts.shape}"
        )
    if np.any(counts < 0):
        raise ValueError("Class pixel counts cannot be negative.")

    active_mask = counts > 0
    total_valid = float(counts[active_mask].sum())

    weights = np.ones(num_classes, dtype=np.float64)
    method_key = method.lower().strip()

    if total_valid > 0 and np.any(active_mask) and method_key != "uniform":
        probs = np.zeros(num_classes, dtype=np.float64)
        probs[active_mask] = counts[active_mask] / total_valid

        if method_key == "enet":
            # ENet log-smoothed weighting: w_c = 1 / ln(1.02 + p_c)
            raw_w = 1.0 / np.log(1.02 + probs[active_mask])
        elif method_key in ("median_freq_sqrt", "damped_inverse"):
            # Square-root damped median frequency balancing: sqrt(median(p) / p_c)
            median_prob = float(np.median(probs[active_mask]))
            raw_w = np.sqrt(median_prob / probs[active_mask])
        elif method_key == "median_freq":
            median_prob = float(np.median(probs[active_mask]))
            raw_w = median_prob / probs[active_mask]
        else:
            raise ValueError(
                f"Unknown class_weight_method '{method}'. "
                "Supported: 'enet', 'median_freq_sqrt', 'median_freq', 'uniform'."
            )

        mean_active = float(raw_w.mean())
        if mean_active > 0:
            raw_w = raw_w / mean_active
        # Clip after mean-normalization so every active weight strictly satisfies
        # min_weight <= w_c <= max_weight while preserving monotonic ordering.
        raw_w = np.clip(raw_w, min_weight, max_weight)
        weights[active_mask] = raw_w
        # Inactive (0-pixel) classes keep neutral weight 1.0 (clamped into [min_weight, max_weight])
        weights[~active_mask] = float(np.clip(1.0, min_weight, max_weight))

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if not torch.isfinite(weight_tensor).all():
        raise RuntimeError(f"Non-finite class weights computed: {weight_tensor}")
    if device is not None:
        weight_tensor = weight_tensor.to(torch.device(device))
    return weight_tensor


def compute_class_weights_from_dataset(
    train_dataset: CamVidDataset,
    num_classes: int = NUM_CLASSES,
    method: str = "enet",
    min_weight: float = 0.25,
    max_weight: float = 5.0,
    device: Optional[Union[str, torch.device]] = None,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Scan the training dataset labels ONLY (ignoring label 255) to compute class pixel counts
    and the 11-element class weight tensor.

    Raises a ValueError if `train_dataset.split != 'train'` to prevent accidental leakage
    of validation or test labels into class weights.
    """
    if getattr(train_dataset, "split", "train") != "train":
        raise ValueError(
            f"Class weights must ONLY be computed from the 'train' split, "
            f"got split='{train_dataset.split}'."
        )

    counts = np.zeros(num_classes, dtype=np.int64)
    n_items = len(train_dataset) if max_samples is None else min(len(train_dataset), max_samples)

    for idx in range(n_items):
        _, label_rgb, _, _, _ = train_dataset.load_raw_pair(idx)
        mask = rgb_to_mask(label_rgb)
        valid = mask[mask < num_classes]
        if valid.size > 0:
            counts += np.bincount(valid, minlength=num_classes)

    weights = compute_class_weights_from_counts(
        counts,
        num_classes=num_classes,
        method=method,
        min_weight=min_weight,
        max_weight=max_weight,
        device=device,
    )
    return weights, counts


def create_dataloaders(
    config: TrainingConfig,
) -> Tuple[CamVidDataset, CamVidDataset, DataLoader, DataLoader]:
    """
    Create training and validation datasets and DataLoaders using the fixed CamVid splits.

    Note: The test set is strictly excluded from training and validation.
    """
    train_transform = (
        RandomHorizontalFlipTransform(prob=0.5) if config.use_augmentation else None
    )
    train_dataset = CamVidDataset(
        root_dir=config.data_dir,
        split="train",
        target_size=config.image_size,
        imagenet_normalize=True,
        transform=train_transform,
    )
    val_dataset = CamVidDataset(
        root_dir=config.data_dir,
        split="val",
        target_size=config.image_size,
        imagenet_normalize=True,
        transform=None,
    )

    use_pin_memory = bool(config.pin_memory and torch.cuda.is_available())

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def compute_segmentation_loss(
    outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    targets: torch.Tensor,
    criterion: nn.Module,
    aux_loss_weight: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Compute segmentation loss from model outputs and ground-truth masks.

    - If `outputs` is a dict containing `'out'` and `'aux'`, computes:
      `total_loss = main_loss + aux_loss_weight * aux_loss` (with `aux_loss_weight = 0.4`).
    - If `outputs` is a single logits tensor (or a dict with only `'out'`),
      uses `total_loss = main_loss`.

    Returns:
        `(total_loss, main_loss, aux_loss, main_logits)`
    """
    if isinstance(outputs, dict):
        main_logits = outputs["out"]
        main_loss = criterion(main_logits, targets)
        aux_logits = outputs.get("aux", None)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, targets)
            total_loss = main_loss + aux_loss_weight * aux_loss
        else:
            aux_loss = None
            total_loss = main_loss
    elif isinstance(outputs, torch.Tensor):
        main_logits = outputs
        main_loss = criterion(main_logits, targets)
        aux_loss = None
        total_loss = main_loss
    else:
        raise TypeError(f"Unsupported model output type: {type(outputs)}")

    return total_loss, main_loss, aux_loss, main_logits


def train_one_step(
    model: CamVidDeepLabV3,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    aux_loss_weight: float = 0.4,
    grad_clip_norm: Optional[float] = 5.0,
    return_dict_from_model: bool = True,
) -> Dict[str, Any]:
    """Execute a single training step (forward, loss, backward, optimizer step)."""
    model.train()
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(images, return_dict=return_dict_from_model)
    total_loss, main_loss, aux_loss, main_logits = compute_segmentation_loss(
        outputs=outputs,
        targets=targets,
        criterion=criterion,
        aux_loss_weight=aux_loss_weight,
    )

    total_loss.backward()
    if grad_clip_norm is not None and grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=grad_clip_norm,
        )
    optimizer.step()

    return {
        "total_loss": float(total_loss.detach().item()),
        "main_loss": float(main_loss.detach().item()),
        "aux_loss": float(aux_loss.detach().item()) if aux_loss is not None else None,
        "logits": main_logits.detach(),
    }


def validate_one_step(
    model: CamVidDeepLabV3,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    aux_loss_weight: float = 0.4,
    return_dict_from_model: bool = True,
) -> Dict[str, Any]:
    """Execute a single validation step using `model.eval()` and `torch.no_grad()`."""
    model.eval()
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images, return_dict=return_dict_from_model)
        total_loss, main_loss, aux_loss, main_logits = compute_segmentation_loss(
            outputs=outputs,
            targets=targets,
            criterion=criterion,
            aux_loss_weight=aux_loss_weight,
        )

    return {
        "total_loss": float(total_loss.detach().item()),
        "main_loss": float(main_loss.detach().item()),
        "aux_loss": float(aux_loss.detach().item()) if aux_loss is not None else None,
        "logits": main_logits.detach(),
    }


def train_one_epoch(
    model: CamVidDeepLabV3,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    aux_loss_weight: float = 0.4,
    grad_clip_norm: Optional[float] = 5.0,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one training epoch and return loss and segmentation metrics."""
    model.train()
    metrics = SegmentationMetrics(num_classes=model.config.num_classes, ignore_index=model.config.ignore_index)

    total_loss_sum = 0.0
    main_loss_sum = 0.0
    num_samples = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch_size = int(images.size(0))
        step_out = train_one_step(
            model=model,
            images=images,
            targets=targets,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            aux_loss_weight=aux_loss_weight,
            grad_clip_norm=grad_clip_norm,
            return_dict_from_model=model.config.aux_loss,
        )
        total_loss_sum += step_out["total_loss"] * batch_size
        main_loss_sum += step_out["main_loss"] * batch_size
        num_samples += batch_size
        metrics.update(step_out["logits"], targets)

    computed = metrics.compute()
    avg_total_loss = total_loss_sum / max(num_samples, 1)
    avg_main_loss = main_loss_sum / max(num_samples, 1)

    return {
        "loss": avg_total_loss,
        "main_loss": avg_main_loss,
        "pixel_accuracy": computed["pixel_accuracy"],
        "miou": computed["miou"],
        "per_class_iou": computed["per_class_iou"],
        "confusion_matrix": computed["confusion_matrix"],
        "num_samples": num_samples,
    }


def validate(
    model: CamVidDeepLabV3,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    aux_loss_weight: float = 0.4,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Run validation pass using `model.eval()` and `torch.no_grad()`."""
    model.eval()
    metrics = SegmentationMetrics(num_classes=model.config.num_classes, ignore_index=model.config.ignore_index)

    total_loss_sum = 0.0
    main_loss_sum = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_size = int(images.size(0))
            step_out = validate_one_step(
                model=model,
                images=images,
                targets=targets,
                criterion=criterion,
                device=device,
                aux_loss_weight=aux_loss_weight,
                return_dict_from_model=model.config.aux_loss,
            )
            total_loss_sum += step_out["total_loss"] * batch_size
            main_loss_sum += step_out["main_loss"] * batch_size
            num_samples += batch_size
            metrics.update(step_out["logits"], targets)

    computed = metrics.compute()
    avg_total_loss = total_loss_sum / max(num_samples, 1)
    avg_main_loss = main_loss_sum / max(num_samples, 1)

    return {
        "loss": avg_total_loss,
        "main_loss": avg_main_loss,
        "pixel_accuracy": computed["pixel_accuracy"],
        "miou": computed["miou"],
        "per_class_iou": computed["per_class_iou"],
        "confusion_matrix": computed["confusion_matrix"],
        "num_samples": num_samples,
        "formatted_report": metrics.format_report(prefix="  "),
    }


def configure_stage_optimizer_and_scheduler(
    model: CamVidDeepLabV3,
    config: TrainingConfig,
    stage: str,
) -> Tuple[torch.optim.Optimizer, ReduceLROnPlateau]:
    """
    Configure backbone freeze state, AdamW optimizer, and ReduceLROnPlateau scheduler
    for either Stage 1 ('warmup') or Stage 2 ('finetune').
    """
    stage_key = stage.lower().strip()
    if stage_key == "warmup":
        model.set_backbone_trainable(False)
        param_groups = model.get_parameter_groups(
            backbone_lr=0.0,
            head_lr=config.warmup_head_lr,
            weight_decay=config.weight_decay,
        )
    elif stage_key == "finetune":
        model.set_backbone_trainable(True)
        param_groups = model.get_parameter_groups(
            backbone_lr=config.finetune_backbone_lr,
            head_lr=config.finetune_head_lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown training stage '{stage}'. Expected 'warmup' or 'finetune'.")

    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",  # Monitor validation mIoU (higher is better)
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lr,
    )
    return optimizer, scheduler


def get_current_lrs(optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    """Extract current (backbone_lr, head_lr) from optimizer parameter groups."""
    backbone_lr = 0.0
    head_lr = 0.0
    for group in optimizer.param_groups:
        gname = group.get("name", "")
        if gname == "backbone":
            backbone_lr = float(group["lr"])
        elif gname == "classifier_head":
            head_lr = float(group["lr"])
    if head_lr == 0.0 and optimizer.param_groups:
        head_lr = float(optimizer.param_groups[-1]["lr"])
    return backbone_lr, head_lr


def resolve_output_paths(config: TrainingConfig) -> Dict[str, Path]:
    """
    Resolve checkpoint and CSV log file paths.
    If `config.overwrite=False` and target files already exist, appends a timestamp
    so existing checkpoints or logs are never overwritten.
    """
    models_dir = Path(config.models_dir)
    results_dir = Path(config.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    exp = config.experiment_name
    best_ckpt = models_dir / f"{exp}_best.pth"
    last_ckpt = models_dir / f"{exp}_last.pth"
    csv_log = results_dir / f"{exp}_history.csv"
    summary_json = results_dir / f"{exp}_summary.json"

    if not config.overwrite and (best_ckpt.exists() or csv_log.exists()):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp = f"{exp}_{stamp}"
        best_ckpt = models_dir / f"{exp}_best.pth"
        last_ckpt = models_dir / f"{exp}_last.pth"
        csv_log = results_dir / f"{exp}_history.csv"
        summary_json = results_dir / f"{exp}_summary.json"

    return {
        "experiment_name": Path(exp),
        "best_checkpoint": best_ckpt,
        "last_checkpoint": last_ckpt,
        "csv_log": csv_log,
        "summary_json": summary_json,
    }


def save_training_checkpoint(
    checkpoint_path: Union[str, Path],
    model: CamVidDeepLabV3,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    epoch: int,
    stage: str,
    best_miou: float,
    metrics: Dict[str, Any],
    class_weights: Optional[torch.Tensor] = None,
    scheduler: Optional[ReduceLROnPlateau] = None,
) -> Path:
    """
    Save checkpoint compatible with `load_checkpoint()` in `model.py`, including:
    - Model state dict
    - Optimizer state dict
    - Scheduler state dict (if provided)
    - Epoch & Stage
    - Best validation mIoU
    - Training configuration
    - 11 class names
    """
    extra_metrics = dict(metrics)
    extra_metrics["stage"] = stage
    extra_metrics["training_config"] = training_config.to_dict()
    if class_weights is not None:
        extra_metrics["class_weights"] = [
            float(w) for w in class_weights.detach().cpu().tolist()
        ]

    saved_path = save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_miou=best_miou,
        metrics=extra_metrics,
    )

    # Also store training_config and scheduler_state_dict at top level of the checkpoint dict
    ckpt_data = torch.load(saved_path, map_location="cpu", weights_only=False)
    ckpt_data["stage"] = stage
    ckpt_data["training_config"] = training_config.to_dict()
    if scheduler is not None:
        ckpt_data["scheduler_state_dict"] = scheduler.state_dict()
    if class_weights is not None:
        ckpt_data["class_weights"] = [
            float(w) for w in class_weights.detach().cpu().tolist()
        ]
    torch.save(ckpt_data, saved_path)
    return saved_path


def load_training_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[
    CamVidDeepLabV3,
    torch.optim.Optimizer,
    ReduceLROnPlateau,
    TrainingConfig,
    Dict[str, Any],
]:
    """
    Load a training checkpoint and reconstruct the model, stage-appropriate optimizer
    (with restored optimizer state_dict), ReduceLROnPlateau scheduler, and TrainingConfig
    so training can be resumed seamlessly.
    """
    target_device = resolve_device(str(device)) if device is not None else get_device()
    model, meta = load_checkpoint(checkpoint_path, device=target_device)

    cfg_dict = meta.get("training_config") or meta.get("metrics", {}).get("training_config", {})
    training_config = TrainingConfig.from_dict(cfg_dict) if cfg_dict else TrainingConfig()
    stage = meta.get("stage") or meta.get("metrics", {}).get("stage", "finetune")

    optimizer, scheduler = configure_stage_optimizer_and_scheduler(
        model=model,
        config=training_config,
        stage=stage,
    )
    if "optimizer_state_dict" in meta and meta["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(meta["optimizer_state_dict"])
    if "scheduler_state_dict" in meta and meta["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(meta["scheduler_state_dict"])

    return model, optimizer, scheduler, training_config, meta


def append_epoch_to_csv(csv_path: Path, row_data: Dict[str, Any]) -> None:
    """Append one epoch's metrics row to CSV, writing the header if the file is new."""
    file_exists = csv_path.is_file() and csv_path.stat().st_size > 0
    fieldnames = list(row_data.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def run_training(
    config: Optional[TrainingConfig] = None,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    max_weight_samples: Optional[int] = None,
    resume_checkpoint: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Execute the complete two-stage transfer learning pipeline:
    - Stage 1: Classifier warm-up (frozen backbone, head_lr=1e-3, default 5 epochs)
    - Stage 2: Fine-tuning (unfrozen backbone, backbone_lr=1e-4, head_lr=5e-4, default 20 epochs)

    If `resume_checkpoint` is provided, loads model, optimizer, and scheduler states
    and resumes from `completed_epoch + 1`.
    """
    config = config or TrainingConfig()
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    paths = resolve_output_paths(config)

    print("=== KONFIGURASI TRAINING DEEPLABV3 MOBILENETV3-LARGE (CAMVID 11 KELAS) ===")
    print(f"  Device              : {device} (CUDA available: {torch.cuda.is_available()})")
    print(f"  Image size (H x W)  : {config.image_size[0]} x {config.image_size[1]}")
    print(f"  Batch size          : {config.batch_size} | num_workers: {config.num_workers}")
    print(f"  Stage 1 (Warm-up)   : {config.warmup_epochs} epochs (head_lr={config.warmup_head_lr})")
    print(
        f"  Stage 2 (Fine-tune) : {config.finetune_epochs} epochs "
        f"(backbone_lr={config.finetune_backbone_lr}, head_lr={config.finetune_head_lr})"
    )
    print(f"  Best checkpoint out : {paths['best_checkpoint']}")
    print(f"  History CSV out     : {paths['csv_log']}")

    # 1. Create train & validation DataLoaders (test set is strictly untouched)
    train_dataset, val_dataset, train_loader, val_loader = create_dataloaders(config)
    print(f"  Train samples       : {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"  Val samples         : {len(val_dataset)} ({len(val_loader)} batches)")

    # 2. Compute class weights from training set ONLY
    if config.use_class_weights:
        print(f"\nMenghitung class weights ({config.class_weight_method}) dari training set...")
        class_weights, train_class_counts = compute_class_weights_from_dataset(
            train_dataset,
            num_classes=config.num_classes,
            method=config.class_weight_method,
            min_weight=config.class_weight_min,
            max_weight=config.class_weight_max,
            device=device,
            max_samples=max_weight_samples,
        )
        for cid, cname in enumerate(CLASS_NAMES_11):
            print(
                f"  Class {cid:2d} ({cname:<11}): "
                f"count={int(train_class_counts[cid]):11,d} px -> weight={float(class_weights[cid]):.4f}"
            )
    else:
        class_weights = None
        train_class_counts = np.zeros(config.num_classes, dtype=np.int64)

    criterion = create_loss_function(
        class_weights=class_weights,
        ignore_index=config.ignore_index,
    )

    # 3. Create or resume model
    resumed_stage: Optional[str] = None
    resumed_optimizer_state: Optional[Dict[str, Any]] = None
    resumed_scheduler_state: Optional[Dict[str, Any]] = None
    completed_epochs = 0
    best_val_miou = -1.0
    best_epoch = 0

    if resume_checkpoint is not None:
        print(f"\nMelanjutkan training dari checkpoint: {resume_checkpoint}")
        model, _, _, _, meta = load_training_checkpoint(resume_checkpoint, device=device)
        completed_epochs = int(meta.get("epoch", 0))
        best_val_miou = float(meta.get("best_miou", -1.0) or -1.0)
        best_epoch = completed_epochs
        resumed_stage = meta.get("stage") or meta.get("metrics", {}).get("stage")
        resumed_optimizer_state = meta.get("optimizer_state_dict")
        resumed_scheduler_state = meta.get("scheduler_state_dict")
    else:
        model_cfg = ModelConfig(
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
            pretrained=config.pretrained,
            aux_loss=config.aux_loss,
            freeze_backbone=True,
            default_input_size=config.image_size,
        )
        model = create_deeplabv3_mobilenet_v3_large(config=model_cfg, device=device)

    history: List[Dict[str, Any]] = []
    global_epoch = 0

    stages: List[Tuple[str, int]] = []
    if config.warmup_epochs > 0:
        stages.append(("warmup", config.warmup_epochs))
    if config.finetune_epochs > 0:
        stages.append(("finetune", config.finetune_epochs))

    for stage_name, stage_epochs in stages:
        print(f"\n================ STAGE: {stage_name.upper()} ({stage_epochs} EPOCHS) ================")
        optimizer, scheduler = configure_stage_optimizer_and_scheduler(
            model=model,
            config=config,
            stage=stage_name,
        )
        if resumed_stage == stage_name and resumed_optimizer_state is not None:
            optimizer.load_state_dict(resumed_optimizer_state)
            if resumed_scheduler_state is not None:
                scheduler.load_state_dict(resumed_scheduler_state)
            resumed_optimizer_state = None
            resumed_scheduler_state = None

        for stage_ep in range(1, stage_epochs + 1):
            global_epoch += 1
            if global_epoch <= completed_epochs:
                continue

            ep_t0 = time.perf_counter()
            bb_lr, hd_lr = get_current_lrs(optimizer)

            train_res = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                aux_loss_weight=config.aux_loss_weight,
                grad_clip_norm=config.grad_clip_norm,
                max_batches=max_train_batches,
            )

            val_res = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                aux_loss_weight=config.aux_loss_weight,
                max_batches=max_val_batches,
            )

            scheduler.step(val_res["miou"])
            ep_seconds = time.perf_counter() - ep_t0

            is_best = val_res["miou"] > best_val_miou
            if is_best:
                best_val_miou = val_res["miou"]
                best_epoch = global_epoch

            row: Dict[str, Any] = {
                "epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": stage_ep,
                "backbone_lr": bb_lr,
                "head_lr": hd_lr,
                "train_loss": round(train_res["loss"], 6),
                "val_loss": round(val_res["loss"], 6),
                "train_pixel_acc": round(train_res["pixel_accuracy"], 6),
                "val_pixel_acc": round(val_res["pixel_accuracy"], 6),
                "train_miou": round(train_res["miou"], 6),
                "val_miou": round(val_res["miou"], 6),
                "best_val_miou": round(best_val_miou, 6),
                "epoch_seconds": round(ep_seconds, 2),
            }
            for cname in CLASS_NAMES_11:
                row[f"val_iou_{cname}"] = round(val_res["per_class_iou"][cname], 6)

            history.append(row)
            append_epoch_to_csv(paths["csv_log"], row)

            ckpt_metrics = {
                "train_loss": train_res["loss"],
                "val_loss": val_res["loss"],
                "train_pixel_accuracy": train_res["pixel_accuracy"],
                "val_pixel_accuracy": val_res["pixel_accuracy"],
                "train_miou": train_res["miou"],
                "val_miou": val_res["miou"],
                "val_per_class_iou": val_res["per_class_iou"],
            }

            # Always update last checkpoint
            save_training_checkpoint(
                checkpoint_path=paths["last_checkpoint"],
                model=model,
                optimizer=optimizer,
                training_config=config,
                epoch=global_epoch,
                stage=stage_name,
                best_miou=best_val_miou,
                metrics=ckpt_metrics,
                class_weights=class_weights,
                scheduler=scheduler,
            )

            # Save best checkpoint whenever validation mIoU improves
            if is_best:
                save_training_checkpoint(
                    checkpoint_path=paths["best_checkpoint"],
                    model=model,
                    optimizer=optimizer,
                    training_config=config,
                    epoch=global_epoch,
                    stage=stage_name,
                    best_miou=best_val_miou,
                    metrics=ckpt_metrics,
                    class_weights=class_weights,
                    scheduler=scheduler,
                )

            best_tag = " [BEST]" if is_best else ""
            print(
                f"[Epoch {global_epoch:02d}/{config.total_epochs:02d} | {stage_name}] "
                f"lr(bb={bb_lr:.1e}, head={hd_lr:.1e}) | "
                f"train_loss={train_res['loss']:.4f}, val_loss={val_res['loss']:.4f} | "
                f"val_acc={val_res['pixel_accuracy'] * 100:.2f}%, "
                f"val_mIoU={val_res['miou'] * 100:.2f}% | "
                f"time={ep_seconds:.1f}s{best_tag}"
            )
            print(val_res["formatted_report"])

    summary = {
        "experiment_name": str(paths["experiment_name"]),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_miou": best_val_miou,
        "total_epochs_run": global_epoch,
        "best_checkpoint": str(paths["best_checkpoint"]),
        "last_checkpoint": str(paths["last_checkpoint"]),
        "csv_log": str(paths["csv_log"]),
        "training_config": config.to_dict(),
    }
    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary": summary,
        "history": history,
        "paths": paths,
        "class_weights": class_weights,
    }


def parse_args(
    argv: Optional[List[str]] = None,
) -> Tuple[TrainingConfig, argparse.Namespace]:
    """Parse command-line arguments into a TrainingConfig instance and raw Namespace."""
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description="Two-stage transfer learning for CamVid 11-class Semantic Segmentation."
    )
    parser.add_argument("--data-dir", type=str, default=defaults.data_dir)
    parser.add_argument("--image-height", type=int, default=defaults.image_size[0])
    parser.add_argument("--image-width", type=int, default=defaults.image_size[1])
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=defaults.seed)

    parser.add_argument("--warmup-epochs", type=int, default=defaults.warmup_epochs)
    parser.add_argument("--warmup-head-lr", type=float, default=defaults.warmup_head_lr)
    parser.add_argument("--finetune-epochs", type=int, default=defaults.finetune_epochs)
    parser.add_argument("--finetune-backbone-lr", type=float, default=defaults.finetune_backbone_lr)
    parser.add_argument("--finetune-head-lr", type=float, default=defaults.finetune_head_lr)

    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--aux-loss-weight", type=float, default=defaults.aux_loss_weight)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--class-weight-method",
        type=str,
        default=defaults.class_weight_method,
        choices=["enet", "median_freq_sqrt", "median_freq", "uniform"],
    )
    parser.add_argument("--scheduler-patience", type=int, default=defaults.scheduler_patience)
    parser.add_argument("--scheduler-factor", type=float, default=defaults.scheduler_factor)
    parser.add_argument("--no-augmentation", action="store_true")

    parser.add_argument("--models-dir", type=str, default=defaults.models_dir)
    parser.add_argument("--results-dir", type=str, default=defaults.results_dir)
    parser.add_argument("--experiment-name", type=str, default=defaults.experiment_name)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-weight-samples", type=int, default=None)

    args = parser.parse_args(argv)
    cfg = TrainingConfig(
        data_dir=args.data_dir,
        image_size=(args.image_height, args.image_width),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        device=args.device,
        seed=args.seed,
        warmup_epochs=args.warmup_epochs,
        warmup_head_lr=args.warmup_head_lr,
        finetune_epochs=args.finetune_epochs,
        finetune_backbone_lr=args.finetune_backbone_lr,
        finetune_head_lr=args.finetune_head_lr,
        weight_decay=args.weight_decay,
        aux_loss_weight=args.aux_loss_weight,
        use_class_weights=not args.no_class_weights,
        class_weight_method=args.class_weight_method,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        use_augmentation=not args.no_augmentation,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        experiment_name=args.experiment_name,
        overwrite=args.overwrite,
    )
    return cfg, args


if __name__ == "__main__":
    cfg, cli_args = parse_args()
    run_training(
        cfg,
        max_train_batches=cli_args.max_train_batches,
        max_val_batches=cli_args.max_val_batches,
        max_weight_samples=cli_args.max_weight_samples,
        resume_checkpoint=cli_args.resume,
    )
