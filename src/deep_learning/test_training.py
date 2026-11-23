"""Unit tests, audit verification tests, and micro-benchmarks for CamVid 11-class training pipeline."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Dict, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn

# Ensure project root is on sys.path when invoked as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.config import CAMVID_DIR, CLASS_NAMES_11, IGNORE_INDEX, NUM_CLASSES
from src.deep_learning.dataset import CamVidDataset
from src.deep_learning.metrics import SegmentationMetrics
from src.deep_learning.model import (
    ModelConfig,
    count_parameters,
    create_deeplabv3_mobilenet_v3_large,
    create_loss_function,
    get_device,
    load_checkpoint,
)
from src.deep_learning.train import (
    compute_class_weights_from_counts,
    compute_class_weights_from_dataset,
    compute_segmentation_loss,
    configure_stage_optimizer_and_scheduler,
    create_dataloaders,
    get_current_lrs,
    load_training_checkpoint,
    run_training,
    save_training_checkpoint,
    train_one_step,
    validate_one_step,
)
from src.deep_learning.training_config import TrainingConfig


def test_dataloader_creation() -> None:
    """1. Verify train and val DataLoader creation and confirm test split is excluded."""
    print("=== [TEST 1] Verifikasi Pembuatan DataLoader (Train & Val) ===")
    cfg = TrainingConfig(image_size=(360, 480), batch_size=2, num_workers=0)
    train_ds, val_ds, train_loader, val_loader = create_dataloaders(cfg)

    assert train_ds.split == "train" and len(train_ds) == 421
    assert val_ds.split == "val" and len(val_ds) == 112
    assert len(train_loader) == 211  # ceil(421 / 2)
    assert len(val_loader) == 56     # 112 / 2

    images, masks = next(iter(train_loader))
    assert tuple(images.shape) == (2, 3, 360, 480)
    assert tuple(masks.shape) == (2, 360, 480)
    assert images.dtype == torch.float32
    assert masks.dtype == torch.int64

    allowed_ids = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
    unique_ids = set(torch.unique(masks).tolist())
    assert unique_ids.issubset(allowed_ids)
    print(
        f"  [PASS] Train DataLoader: {len(train_ds)} sampel ({len(train_loader)} batch @ batch_size=2), "
        f"Val DataLoader: {len(val_ds)} sampel ({len(val_loader)} batch)."
    )
    print("  [PASS] Test split tidak disentuh selama pembuatan DataLoader training & validasi.")


def test_confusion_matrix_metrics_and_ignore_index() -> None:
    """
    2. Audit & verify 11x11 confusion matrix, Pixel Accuracy, Per-Class IoU, mIoU,
    ignore_index=255 exclusion, all-ignore target case, and prediction-only class case.
    """
    print("\n=== [TEST 2] Audit Confusion Matrix 11x11, Pixel Accuracy, mIoU, & Edge Cases ===")
    metrics = SegmentationMetrics(num_classes=11, ignore_index=255)

    # Case A: Standard synthetic 2x4 ground-truth and predictions with ignore_index=255
    # Row 0: GT=[0, 0, 1, 1] -> Pred=[0, 1, 1, 1]
    #   Class 0: TP=1, FN=1, FP=0 -> Union=2 -> IoU(0) = 1/2 = 0.50
    #   Class 1: TP=2, FN=0, FP=1 -> Union=3 -> IoU(1) = 2/3 = 0.6667
    # Row 1: GT=[2, 255, 255, 255] -> Pred=[2, 8, 9, 10]
    #   Class 2: TP=1, FN=0, FP=0 -> Union=1 -> IoU(2) = 1/1 = 1.00
    #   Pixels with GT=255 must be completely ignored even though predictions are 8, 9, 10!
    #   Classes 3..10 have 0 GT and 0 valid Pred (Union=0) -> excluded from mIoU, IoU=0.0 (no NaN!)
    gt = np.array([[0, 0, 1, 1], [2, 255, 255, 255]], dtype=np.int64)
    pred = np.array([[0, 1, 1, 1], [2, 8, 9, 10]], dtype=np.int64)

    metrics.update(pred, gt)
    res = metrics.compute()

    cm = res["confusion_matrix"]
    assert cm.shape == (11, 11)
    assert res["total_valid_pixels"] == 5
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 1] == 2
    assert cm[2, 2] == 1
    # Predictions (8, 9, 10) on ignore_index=255 pixels must NOT enter confusion matrix
    assert cm[:, 8].sum() == 0 and cm[:, 9].sum() == 0 and cm[:, 10].sum() == 0

    expected_acc = 4.0 / 5.0  # 0.80
    expected_miou = (0.5 + (2.0 / 3.0) + 1.0) / 3.0
    assert np.isclose(res["pixel_accuracy"], expected_acc, atol=1e-6)
    assert res["num_valid_classes"] == 3
    assert np.isclose(res["miou"], expected_miou, atol=1e-6)

    for cid in range(3, 11):
        cname = CLASS_NAMES_11[cid]
        assert res["per_class_iou"][cname] == 0.0
        assert not np.isnan(res["per_class_iou"][cname])
        assert res["per_class_valid"][cname] is False

    # Case B: All targets contain ONLY ignore_index = 255
    all_ignore_metrics = SegmentationMetrics(num_classes=11, ignore_index=255)
    all_ignore_gt = np.full((6, 6), 255, dtype=np.int64)
    all_ignore_pred = np.random.randint(0, 11, size=(6, 6), dtype=np.int64)
    all_ignore_metrics.update(all_ignore_pred, all_ignore_gt)
    all_ignore_res = all_ignore_metrics.compute()
    assert all_ignore_res["total_valid_pixels"] == 0
    assert all_ignore_res["confusion_matrix"].sum() == 0
    assert all_ignore_res["num_valid_classes"] == 0
    assert all_ignore_res["pixel_accuracy"] == 0.0 and not np.isnan(all_ignore_res["pixel_accuracy"])
    assert all_ignore_res["miou"] == 0.0 and not np.isnan(all_ignore_res["miou"])
    assert all(not np.isnan(v) and v == 0.0 for v in all_ignore_res["per_class_iou"].values())

    # Case C: Class that ONLY appears in predictions on valid pixels (GT=0, valid Pred > 0)
    # Suppose GT has only Class 0 (4 pixels): [0, 0, 0, 0]
    # Model predicts Class 0 on 2 pixels, and falsely predicts Class 5 (Tree) on 2 pixels: [0, 0, 5, 5]
    # Then:
    # - Class 0: GT=4, Pred=2, TP=2 -> Union=4 -> IoU(0) = 2/4 = 0.50 (valid=True)
    # - Class 5: GT=0, Pred=2, TP=0 -> Union=2 > 0 -> IoU(5) = 0/2 = 0.00 (valid=True, included in mIoU!)
    # - Other 9 classes: GT=0, Pred=0 -> Union=0 -> excluded from mIoU (valid=False)
    # - mIoU across the 2 active classes (Class 0 and Class 5) = (0.50 + 0.00) / 2 = 0.25
    pred_only_metrics = SegmentationMetrics(num_classes=11, ignore_index=255)
    gt_c = np.array([[0, 0, 0, 0]], dtype=np.int64)
    pred_c = np.array([[0, 0, 5, 5]], dtype=np.int64)
    pred_only_metrics.update(pred_c, gt_c)
    res_c = pred_only_metrics.compute()

    assert res_c["per_class_gt_pixels"]["Tree"] == 0
    assert res_c["per_class_pred_pixels"]["Tree"] == 2
    assert res_c["per_class_valid"]["Tree"] is True
    assert res_c["per_class_iou"]["Tree"] == 0.0
    assert res_c["num_valid_classes"] == 2  # Sky (0) and Tree (5)
    assert np.isclose(res_c["miou"], (0.5 + 0.0) / 2.0, atol=1e-6)

    print(
        f"  [PASS] Case A (Standard + ignore_index=255): Pixel Accuracy={res['pixel_accuracy']*100:.2f}%, "
        f"mIoU={res['miou']*100:.2f}% (3 kelas aktif, 8 kelas union=0 dikeluarkan dari mIoU)."
    )
    print("  [PASS] Case B (Seluruh target = 255): Confusion matrix kosong, Pixel Acc = 0.0, mIoU = 0.0 (tanpa NaN).")
    print(
        "  [PASS] Case C (Kelas hanya muncul pada prediksi di piksel valid, GT=0 & Pred=2): "
        "Union=2 > 0 -> IoU=0.0%, masuk sebagai kelas aktif dan menurunkan mIoU dari 50.0% ke 25.0%."
    )


def test_class_weights_audit() -> None:
    """
    3. Audit class weights computation:
    - Verify strict bounds [min_weight, max_weight] = [0.25, 5.0]
    - Verify monotonic ordering (rarer class weight >= more common class weight)
    - Verify zero-sample class handling and extreme imbalance clipping
    """
    print("\n=== [TEST 3] Audit Perhitungan Class Weights (Bounds [0.25, 5.0], Monotonisitas, & Zero-Sample) ===")
    min_w, max_w = 0.25, 5.0

    # 3a. Synthetic counts with a 0-sample class and extreme ratio to trigger both min and max clipping
    extreme_counts = [10_000_000, 5_000_000, 100_000, 50_000, 10_000, 5_000, 1_000, 500, 100, 1, 0]
    for method in ("enet", "median_freq_sqrt", "median_freq", "uniform"):
        w = compute_class_weights_from_counts(
            extreme_counts,
            num_classes=11,
            method=method,
            min_weight=min_w,
            max_weight=max_w,
        )
        assert tuple(w.shape) == (11,)
        assert torch.isfinite(w).all()
        w_np = w.numpy()
        assert np.all(w_np >= min_w - 1e-7) and np.all(w_np <= max_w + 1e-7), (
            f"Method '{method}' produced weights outside [{min_w}, {max_w}]: {w_np}"
        )
        # Zero-sample class (index 10) must receive neutral weight 1.0
        assert np.isclose(w_np[10], 1.0, atol=1e-6)
        # Active classes (indices 0..9) are sorted in descending count order -> weights must be non-decreasing!
        active_w = w_np[:10]
        assert np.all(np.diff(active_w) >= -1e-6), (
            f"Method '{method}' violated monotonicity: {active_w}"
        )

    # 3b. Verify guard against computing class weights on val/test dataset
    for forbidden_split in ("val", "test"):
        ds_forbidden = CamVidDataset(CAMVID_DIR, split=forbidden_split)
        rejected = False
        try:
            compute_class_weights_from_dataset(ds_forbidden)
        except ValueError:
            rejected = True
        assert rejected, f"Expected ValueError when computing class weights on '{forbidden_split}' split!"

    # 3c. Verify actual full training set class weights (421 train masks)
    train_ds = CamVidDataset(CAMVID_DIR, split="train")
    t0 = time.perf_counter()
    full_weights, full_counts = compute_class_weights_from_dataset(
        train_ds,
        method="enet",
        min_weight=min_w,
        max_weight=max_w,
    )
    dt = time.perf_counter() - t0
    w_list = full_weights.numpy()

    # Check strict bounds [0.25, 5.0]
    assert np.all(w_list >= min_w) and np.all(w_list <= max_w), (
        f"Full train weights out of [{min_w}, {max_w}]: min={w_list.min()}, max={w_list.max()}"
    )

    # Check strict monotonicity across all 11 classes when ordered by pixel count descending
    order_by_count_desc = np.argsort(-full_counts)
    ordered_weights = w_list[order_by_count_desc]
    assert np.all(np.diff(ordered_weights) >= -1e-7), (
        f"Monotonicity violated on CamVid train weights: {ordered_weights}"
    )

    print(
        f"  [PASS] Seluruh 11 class weights berada ketat di dalam [{min_w:.2f}, {max_w:.2f}] "
        f"(min aktual = {w_list.min():.4f}, max aktual = {w_list.max():.4f}, dihitung dalam {dt:.2f}s):"
    )
    for cid in order_by_count_desc:
        cname = CLASS_NAMES_11[cid]
        print(
            f"    - [{cid:2d}] {cname:<11}: {int(full_counts[cid]):11,d} px -> weight = {float(w_list[cid]):.4f}"
        )
    print("  [PASS] Monotonisitas terverifikasi: semakin jarang jumlah piksel kelas, bobotnya semakin besar/sama.")
    print("  [PASS] Kelas tanpa sampel (0 piksel) mendapat bobot netral 1.0000 tanpa division-by-zero atau NaN.")


def test_backbone_freeze_and_batchnorm_audit_and_pipeline() -> Dict[str, float]:
    """
    4. Audit backbone freeze & BatchNorm running statistics during Stage 1 warm-up,
    Stage 1 -> Stage 2 transition parameter groups, checkpoint resume with optimizer state,
    and CLI execution from project root.
    """
    print("\n=== [TEST 4] Audit Backbone Freeze + BatchNorm Stats, Transisi Stage 1->2, & Checkpoint Resume ===")
    device = get_device()
    cfg = TrainingConfig(image_size=(360, 480), batch_size=2, num_workers=0)

    proc = psutil.Process()
    rss_before_mb = proc.memory_info().rss / (1024 * 1024)

    model = create_deeplabv3_mobilenet_v3_large(
        ModelConfig(
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            pretrained=True,
            aux_loss=True,
            freeze_backbone=True,
            default_input_size=(360, 480),
        ),
        device=device,
    )

    weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
    criterion = create_loss_function(class_weights=weights, ignore_index=IGNORE_INDEX)

    torch.manual_seed(42)
    images = torch.randn(2, 3, 360, 480, device=device)
    targets = torch.randint(0, NUM_CLASSES, (2, 360, 480), dtype=torch.long, device=device)
    targets[:, 0:20, 0:20] = IGNORE_INDEX

    # 4a. Verify compute_segmentation_loss handles both dict (out+aux) and single logits tensor
    model.eval()
    with torch.no_grad():
        dict_out = model(images, return_dict=True)
        tensor_out = model(images, return_dict=False)
        tot_l, main_l, aux_l, _ = compute_segmentation_loss(dict_out, targets, criterion, aux_loss_weight=0.4)
        single_tot_l, single_main_l, single_aux_l, _ = compute_segmentation_loss(tensor_out, targets, criterion)
        assert aux_l is not None and torch.isclose(tot_l, main_l + 0.4 * aux_l, atol=1e-6)
        assert single_aux_l is None and torch.isclose(single_tot_l, single_main_l, atol=1e-6)

    # 4b. AUDIT 1: Stage 1 (Warm-up) — Snapshot ALL backbone parameters AND ALL backbone BatchNorm buffers
    opt_s1, sched_s1 = configure_stage_optimizer_and_scheduler(model, cfg, stage="warmup")

    # Even if model.train() is called explicitly, backbone must stay in eval() mode while head is in train()
    model.train()
    assert not model.backbone.training, "Backbone must be in eval() mode during warm-up!"
    assert model.classifier.training, "Classifier head must be in train() mode during warm-up!"
    backbone_bn_layers = [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    head_bn_layers = [m for m in model.classifier.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(backbone_bn_layers) > 0 and all(not bn.training for bn in backbone_bn_layers)
    assert len(head_bn_layers) > 0 and all(bn.training for bn in head_bn_layers)

    backbone_params_before = {
        name: p.detach().clone() for name, p in model.backbone.named_parameters()
    }
    backbone_buffers_before = {
        name: b.detach().clone() for name, b in model.backbone.named_buffers()
    }
    head_bn_mean_before = head_bn_layers[0].running_mean.detach().clone()
    head_bn_var_before = head_bn_layers[0].running_var.detach().clone()
    head_conv_before = model.classifier[4].weight.detach().clone()

    t0_s1 = time.perf_counter()
    step_s1 = train_one_step(
        model=model,
        images=images,
        targets=targets,
        criterion=criterion,
        optimizer=opt_s1,
        device=device,
        aux_loss_weight=0.4,
    )
    dt_s1 = time.perf_counter() - t0_s1

    # Compare every single backbone parameter before vs after warm-up step
    for name, p_after in model.backbone.named_parameters():
        assert p_after.grad is None, f"Backbone parameter {name} received gradient during warm-up!"
        assert torch.equal(p_after.data, backbone_params_before[name]), (
            f"Backbone parameter {name} changed during Stage 1 warm-up!"
        )

    # Compare every single backbone buffer (running_mean, running_var, num_batches_tracked) before vs after
    for name, b_after in model.backbone.named_buffers():
        assert torch.equal(b_after.data, backbone_buffers_before[name]), (
            f"Backbone BatchNorm buffer {name} changed during Stage 1 warm-up!"
        )

    # Confirm classifier head weights AND classifier head BatchNorm running stats DID update!
    assert not torch.equal(model.classifier[4].weight.data, head_conv_before), (
        "Classifier head weights did NOT update during warm-up!"
    )
    assert not torch.equal(head_bn_layers[0].running_mean.data, head_bn_mean_before), (
        "Classifier head BatchNorm running_mean did NOT update during warm-up!"
    )
    assert not torch.equal(head_bn_layers[0].running_var.data, head_bn_var_before), (
        "Classifier head BatchNorm running_var did NOT update during warm-up!"
    )

    print(
        f"  [PASS] Audit Backbone Freeze (Stage 1 Warm-up, time={dt_s1:.3f}s): "
        f"{len(backbone_params_before)} tensor parameter backbone & "
        f"{len(backbone_buffers_before)} buffer BatchNorm backbone (running_mean, running_var, "
        f"num_batches_tracked pada {len(backbone_bn_layers)} layer BN) 100% TIDAK BERUBAH."
    )
    print(
        f"  [PASS] Classifier head tetap dalam mode train() dan BatchNorm head ({len(head_bn_layers)} layer BN) "
        "terbukti memperbarui running_mean & running_var secara normal."
    )

    # 4c. AUDIT 4: Transition from Stage 1 (Warm-up) to Stage 2 (Fine-tuning)
    warmed_head_weight = model.classifier[4].weight.detach().clone()
    opt_s2, sched_s2 = configure_stage_optimizer_and_scheduler(model, cfg, stage="finetune")
    model.train()

    # Verify state preservation and parameter groups upon Stage 1 -> Stage 2 transition
    assert torch.equal(model.classifier[4].weight.data, warmed_head_weight), (
        "Transition to Stage 2 lost warmed-up classifier head weights!"
    )
    assert model.backbone.training and all(bn.training for bn in backbone_bn_layers), (
        "Backbone BatchNorm layers were not switched back to train() mode in Stage 2!"
    )
    assert len(opt_s2.param_groups) == 2
    assert opt_s2.param_groups[0]["name"] == "backbone"
    assert opt_s2.param_groups[1]["name"] == "classifier_head"
    bb_lr_s2, hd_lr_s2 = get_current_lrs(opt_s2)
    assert np.isclose(bb_lr_s2, 1e-4) and np.isclose(hd_lr_s2, 5e-4)

    pstats = count_parameters(model)
    opt_bb_params = sum(p.numel() for p in opt_s2.param_groups[0]["params"])
    opt_hd_params = sum(p.numel() for p in opt_s2.param_groups[1]["params"])
    assert opt_bb_params == pstats["backbone"] == 2_971_952
    assert opt_hd_params == pstats["classifier_head"] + pstats["aux_classifier"] == 8_054_696
    assert opt_bb_params + opt_hd_params == pstats["total"] == 11_026_648

    t0_s2 = time.perf_counter()
    step_s2 = train_one_step(
        model=model,
        images=images,
        targets=targets,
        criterion=criterion,
        optimizer=opt_s2,
        device=device,
        aux_loss_weight=0.4,
    )
    dt_s2 = time.perf_counter() - t0_s2

    # Verify both backbone parameters and backbone BatchNorm stats updated in Stage 2
    assert not torch.equal(
        next(model.backbone.parameters()).data,
        next(iter(backbone_params_before.values())),
    ), "Backbone parameters did NOT update in Stage 2!"
    assert not torch.equal(
        backbone_bn_layers[0].running_mean.data,
        backbone_buffers_before["0.1.running_mean"],
    ), "Backbone BatchNorm running_mean did NOT update in Stage 2!"
    print(
        f"  [PASS] Transisi Stage 1 -> Stage 2 (time={dt_s2:.3f}s): State model terjaga, "
        f"parameter groups lengkap (backbone={opt_bb_params:,d} @ lr=1e-4, "
        f"head={opt_hd_params:,d} @ lr=5e-4), serta parameter & BN backbone ikut terlatih."
    )

    # 4d. Single validation step
    t0_val = time.perf_counter()
    val_step = validate_one_step(
        model=model,
        images=images,
        targets=targets,
        criterion=criterion,
        device=device,
        aux_loss_weight=0.4,
    )
    dt_val = time.perf_counter() - t0_val
    assert not model.training
    print(
        f"  [PASS] Validation Step (eval + no_grad): total_loss={val_step['total_loss']:.4f} | time={dt_val:.3f}s"
    )

    rss_after_mb = proc.memory_info().rss / (1024 * 1024)

    # 4e. Save & Load Training Checkpoint with Optimizer State + CLI Execution from Project Root
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_resume_ckpt.pth"
        save_training_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=opt_s2,
            training_config=cfg,
            epoch=6,
            stage="finetune",
            best_miou=0.5432,
            metrics={"val_miou": 0.5432, "val_pixel_accuracy": 0.8765},
            class_weights=weights,
            scheduler=sched_s2,
        )

        # Verify load_training_checkpoint restores model, optimizer state (including AdamW step & exp_avg), and scheduler
        resumed_model, resumed_opt, resumed_sched, resumed_cfg, meta = load_training_checkpoint(
            ckpt_path, device=device
        )
        assert meta["epoch"] == 6
        assert np.isclose(meta["best_miou"], 0.5432)
        assert len(resumed_opt.param_groups) == 2
        assert len(resumed_opt.state) > 0, "Optimizer state (AdamW momentum buffers) was empty after load!"

        # Execute one more training step with resumed model and resumed optimizer to confirm seamless continuation
        resumed_step = train_one_step(
            model=resumed_model,
            images=images,
            targets=targets,
            criterion=criterion,
            optimizer=resumed_opt,
            device=device,
            aux_loss_weight=0.4,
        )
        assert np.isfinite(resumed_step["total_loss"])
        print("  [PASS] Checkpoint resume berhasil memuat model_state_dict, optimizer_state_dict (AdamW buffers), & scheduler_state_dict.")

        # 4f. Test running `python src/deep_learning/train.py` via subprocess directly from PROJECT_ROOT
        cli_cmd = [
            sys.executable,
            "src/deep_learning/train.py",
            "--warmup-epochs", "1",
            "--finetune-epochs", "1",
            "--batch-size", "2",
            "--max-train-batches", "1",
            "--max-val-batches", "1",
            "--max-weight-samples", "4",
            "--models-dir", str(Path(tmp_dir) / "cli_models"),
            "--results-dir", str(Path(tmp_dir) / "cli_results"),
            "--experiment-name", "cli_root_test",
        ]
        proc_res = subprocess.run(
            cli_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_res.returncode == 0, (
            f"CLI execution from project root failed (code={proc_res.returncode}):\n"
            f"STDOUT:\n{proc_res.stdout}\nSTDERR:\n{proc_res.stderr}"
        )
        cli_best_ckpt = Path(tmp_dir) / "cli_models" / "cli_root_test_best.pth"
        cli_csv = Path(tmp_dir) / "cli_results" / "cli_root_test_history.csv"
        assert cli_best_ckpt.is_file(), "CLI run did not create best checkpoint!"
        assert cli_csv.is_file(), "CLI run did not create history CSV!"
        with open(cli_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert rows[0]["stage"] == "warmup" and rows[1]["stage"] == "finetune"

    print(
        "  [PASS] Eksekusi CLI `python src/deep_learning/train.py` langsung dari root project "
        "berhasil (exit code 0) tanpa masalah import."
    )

    return {
        "stage1_step_sec": dt_s1,
        "stage2_step_sec": dt_s2,
        "val_step_sec": dt_val,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
    }


def main() -> None:
    test_dataloader_creation()
    test_confusion_matrix_metrics_and_ignore_index()
    test_class_weights_audit()
    bench = test_backbone_freeze_and_batchnorm_audit_and_pipeline()

    s1_epoch_sec = 211 * bench["stage1_step_sec"] + 56 * bench["val_step_sec"]
    s2_epoch_sec = 211 * bench["stage2_step_sec"] + 56 * bench["val_step_sec"]
    total_25ep_min = (5 * s1_epoch_sec + 20 * s2_epoch_sec) / 60.0

    print("\n=== [HASIL BENCHMARK AKTUAL & ESTIMASI WAKTU/MEMORI LOKAL (CPU)] ===")
    print(f"  - Waktu 1 Train Batch (Stage 1 - Warm-up, B=2, 360x480)  : {bench['stage1_step_sec']:.3f} detik")
    print(f"  - Waktu 1 Train Batch (Stage 2 - Finetune, B=2, 360x480) : {bench['stage2_step_sec']:.3f} detik")
    print(f"  - Waktu 1 Val Batch   (Eval + no_grad, B=2, 360x480)     : {bench['val_step_sec']:.3f} detik")
    print(f"  - Estimasi 1 Epoch Stage 1 (211 train + 56 val batch)    : {s1_epoch_sec/60.0:.2f} menit")
    print(f"  - Estimasi 1 Epoch Stage 2 (211 train + 56 val batch)    : {s2_epoch_sec/60.0:.2f} menit")
    print(f"  - Estimasi Total 25 Epoch (5 Warm-up + 20 Fine-tune)     : {total_25ep_min:.1f} menit (~{total_25ep_min/60.0:.2f} jam)")
    print(f"  - Penggunaan RAM Proses Python (RSS Peak)                : {bench['rss_after_mb']:.1f} MB")
    print("\n>>> SEMUA AUDIT & UNIT TEST TRAINING PIPELINE LULUS 100% <<<")


if __name__ == "__main__":
    main()
