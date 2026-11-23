"""Evaluation metrics (11x11 Confusion Matrix, Pixel Accuracy, Per-Class IoU, mIoU) for CamVid."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .config import CLASS_NAMES_11, IGNORE_INDEX, NUM_CLASSES


class SegmentationMetrics:
    """
    Accumulates an 11x11 confusion matrix across batches and computes:
    - Pixel Accuracy
    - Per-class IoU (for each of the 11 CamVid classes)
    - Mean IoU (mIoU)

    Pixels with ground-truth label `ignore_index` (255) are excluded.
    Classes with zero ground-truth and zero predicted pixels (union == 0)
    are handled deterministically without producing NaN.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or list(CLASS_NAMES_11)
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class names, got {len(self.class_names)}"
            )
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def reset(self) -> None:
        """Reset the 11x11 confusion matrix to all zeros."""
        self.confusion_matrix.fill(0)

    def update(
        self,
        preds: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """
        Update the confusion matrix with a batch of predictions and ground-truth targets.

        Args:
            preds: Either logits of shape `(B, C, H, W)` / `(C, H, W)`
                or predicted class IDs of shape `(B, H, W)` / `(H, W)`.
            targets: Ground-truth class IDs of shape `(B, H, W)` / `(H, W)`
                with values in `{0..10, 255}`.
        """
        if isinstance(preds, torch.Tensor):
            preds_t = preds.detach()
            if preds_t.ndim == 4:
                preds_t = torch.argmax(preds_t, dim=1)
            elif preds_t.ndim == 3 and preds_t.size(0) == self.num_classes and (
                isinstance(targets, torch.Tensor) and targets.ndim == 2
            ):
                preds_t = torch.argmax(preds_t, dim=0)
            pred_np = preds_t.cpu().numpy().astype(np.int64, copy=False)
        else:
            pred_np = np.asarray(preds)
            if pred_np.ndim == 4:
                pred_np = np.argmax(pred_np, axis=1)
            elif pred_np.ndim == 3 and pred_np.shape[0] == self.num_classes and (
                np.asarray(targets).ndim == 2
            ):
                pred_np = np.argmax(pred_np, axis=0)
            pred_np = pred_np.astype(np.int64, copy=False)

        if isinstance(targets, torch.Tensor):
            target_np = targets.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            target_np = np.asarray(targets, dtype=np.int64)

        if pred_np.shape != target_np.shape:
            raise ValueError(
                f"Shape mismatch between predictions {pred_np.shape} and targets {target_np.shape}"
            )

        # Ignore pixels where target == ignore_index (255) or outside valid range [0, num_classes - 1]
        valid_mask = (
            (target_np != self.ignore_index)
            & (target_np >= 0)
            & (target_np < self.num_classes)
            & (pred_np >= 0)
            & (pred_np < self.num_classes)
        )

        if not np.any(valid_mask):
            return

        valid_targets = target_np[valid_mask]
        valid_preds = pred_np[valid_mask]

        flat_indices = self.num_classes * valid_targets + valid_preds
        hist = np.bincount(flat_indices, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )
        self.confusion_matrix += hist

    def compute(self) -> Dict[str, Any]:
        """
        Compute Pixel Accuracy, Per-Class IoU, and Mean IoU (mIoU) from the 11x11 confusion matrix.

        Policy for `ignore_index=255` and zero-union classes:
            1. Pixels whose ground-truth label is `ignore_index` (255) are filtered out in
               `update()` before populating the confusion matrix. Consequently, any prediction
               made at an `ignore_index` pixel never enters the confusion matrix.
            2. For each class `c` in `0..10`:
               `Union[c] = GT[c] + Pred[c] - TP[c]` (evaluated over valid non-ignored pixels).
               - If `Union[c] > 0` (including when `GT[c] == 0` and `Pred[c] > 0` on valid pixels):
                 `IoU[c] = TP[c] / Union[c]`, `per_class_valid[c] = True`, and class `c` is
                 INCLUDED in the `mIoU` average.
               - If `Union[c] == 0` (neither ground-truth nor predictions on valid pixels contain
                 class `c`): `IoU[c]` is set to `0.0` (never `NaN`), `per_class_valid[c] = False`,
                 and class `c` is EXCLUDED from the `mIoU` denominator.
            3. If all 11 classes have `Union[c] == 0` (e.g. all pixels in the target mask are
               `ignore_index = 255`), `pixel_accuracy` and `miou` both deterministically return `0.0`.

        Returns a dictionary with:
            - `confusion_matrix`: (11, 11) int64 ndarray (rows=GT, cols=Pred)
            - `total_valid_pixels`: total evaluated pixels (excluding ignore_index=255)
            - `pixel_accuracy`: float in [0.0, 1.0]
            - `per_class_iou`: dict mapping class_name -> float IoU in [0.0, 1.0]
            - `per_class_iou_list`: list of 11 float IoUs in [0.0, 1.0]
            - `per_class_valid`: dict mapping class_name -> bool (True if union > 0)
            - `per_class_gt_pixels`: dict mapping class_name -> int GT pixel count
            - `per_class_pred_pixels`: dict mapping class_name -> int predicted pixel count
            - `num_valid_classes`: number of classes with union > 0
            - `miou`: float mean IoU over valid classes (0.0 if no valid pixels)
        """
        cm = self.confusion_matrix
        total_valid_pixels = int(cm.sum())
        tp = np.diag(cm).astype(np.float64)
        gt_per_class = cm.sum(axis=1).astype(np.float64)
        pred_per_class = cm.sum(axis=0).astype(np.float64)
        union = gt_per_class + pred_per_class - tp

        if total_valid_pixels > 0:
            pixel_accuracy = float(tp.sum() / float(total_valid_pixels))
        else:
            pixel_accuracy = 0.0

        per_class_iou_arr = np.zeros(self.num_classes, dtype=np.float64)
        valid_classes_mask = union > 0
        per_class_iou_arr[valid_classes_mask] = (
            tp[valid_classes_mask] / union[valid_classes_mask]
        )

        num_valid_classes = int(valid_classes_mask.sum())
        if num_valid_classes > 0:
            miou = float(per_class_iou_arr[valid_classes_mask].mean())
        else:
            miou = 0.0

        per_class_iou: Dict[str, float] = {}
        per_class_valid: Dict[str, bool] = {}
        per_class_gt: Dict[str, int] = {}
        per_class_pred: Dict[str, int] = {}

        for cid, cname in enumerate(self.class_names):
            per_class_iou[cname] = float(per_class_iou_arr[cid])
            per_class_valid[cname] = bool(valid_classes_mask[cid])
            per_class_gt[cname] = int(gt_per_class[cid])
            per_class_pred[cname] = int(pred_per_class[cid])

        return {
            "confusion_matrix": cm.copy(),
            "total_valid_pixels": total_valid_pixels,
            "pixel_accuracy": pixel_accuracy,
            "miou": miou,
            "num_valid_classes": num_valid_classes,
            "per_class_iou": per_class_iou,
            "per_class_iou_list": [float(v) for v in per_class_iou_arr.tolist()],
            "per_class_valid": per_class_valid,
            "per_class_gt_pixels": per_class_gt,
            "per_class_pred_pixels": per_class_pred,
        }

    def format_report(self, prefix: str = "") -> str:
        """Return a formatted multi-line report of Pixel Accuracy, per-class IoU, and mIoU."""
        res = self.compute()
        lines = [
            f"{prefix}Pixel Accuracy : {res['pixel_accuracy'] * 100:.2f}% "
            f"({res['total_valid_pixels']:,d} valid pixels)",
            f"{prefix}Mean IoU (mIoU): {res['miou'] * 100:.2f}% "
            f"({res['num_valid_classes']}/{self.num_classes} active classes)",
            f"{prefix}{'ID':>2} | {'Class Name':<12} | {'IoU (%)':>8} | {'GT Pixels':>12} | {'Pred Pixels':>12} | {'Status'}",
            f"{prefix}" + "-" * 68,
        ]
        for cid, cname in enumerate(self.class_names):
            iou_pct = res["per_class_iou"][cname] * 100.0
            gt_px = res["per_class_gt_pixels"][cname]
            pr_px = res["per_class_pred_pixels"][cname]
            status = "Active" if res["per_class_valid"][cname] else "Absent (Excluded)"
            lines.append(
                f"{prefix}{cid:2d} | {cname:<12} | {iou_pct:7.2f}% | {gt_px:12,d} | {pr_px:12,d} | {status}"
            )
        return "\n".join(lines)
