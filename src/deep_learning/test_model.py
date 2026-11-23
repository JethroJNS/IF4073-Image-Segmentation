"""Validation tests for DeepLabV3 MobileNetV3-Large 11-class CamVid model."""

from pathlib import Path
import sys
import tempfile
import time

import torch
import torchvision
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

# Ensure project root is on sys.path when invoked as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.config import IGNORE_INDEX, NUM_CLASSES
from src.deep_learning.model import (
    CamVidDeepLabV3,
    ModelConfig,
    count_parameters,
    create_deeplabv3_mobilenet_v3_large,
    create_loss_function,
    get_device,
    load_checkpoint,
    save_checkpoint,
)


def test_environment_and_pretrained_weights() -> CamVidDeepLabV3:
    """Verify PyTorch/torchvision versions, model creation, and official pretrained weights."""
    print("=== [TEST 1] Verifikasi Environment, Pembuatan Model, & Pretrained Weights ===")
    device = get_device()
    print(f"  PyTorch version     : {torch.__version__}")
    print(f"  Torchvision version : {torchvision.__version__}")
    print(f"  Selected device     : {device} (cuda.is_available={torch.cuda.is_available()})")

    cfg = ModelConfig(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        pretrained=True,
        aux_loss=True,
        default_input_size=(360, 480),
    )
    model = create_deeplabv3_mobilenet_v3_large(config=cfg, device=device)
    print(f"  Pretrained source   : {model.pretrained_source}")

    # Compare backbone and ASPP weights against raw official torchvision model to prove pretrained loading
    ref = deeplabv3_mobilenet_v3_large(
        weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    ).to(device)

    for (name_a, param_a), (name_b, param_b) in zip(
        model.backbone.named_parameters(), ref.backbone.named_parameters()
    ):
        assert name_a == name_b
        assert torch.equal(param_a.data, param_b.data), (
            f"Backbone weight mismatch at {name_a}: pretrained weights were not preserved!"
        )

    # Also verify ASPP module (classifier[0]) and 3x3 conv (classifier[1]) preserved pretrained weights
    for idx in (0, 1, 2):
        for (name_a, param_a), (name_b, param_b) in zip(
            model.classifier[idx].named_parameters(), ref.classifier[idx].named_parameters()
        ):
            assert name_a == name_b
            assert torch.equal(param_a.data, param_b.data), (
                f"DeepLabHead pretrained weight mismatch at classifier[{idx}].{name_a}"
            )

    # Verify final 1x1 classifier head was adapted from 21 classes to 11 classes
    head_conv = model.classifier[4]
    assert isinstance(head_conv, torch.nn.Conv2d)
    assert head_conv.in_channels == 256
    assert head_conv.out_channels == NUM_CLASSES == 11
    assert head_conv.kernel_size == (1, 1)

    aux_conv = model.aux_classifier[4]
    assert isinstance(aux_conv, torch.nn.Conv2d)
    assert aux_conv.in_channels == 10
    assert aux_conv.out_channels == NUM_CLASSES == 11
    assert aux_conv.kernel_size == (1, 1)

    param_stats = count_parameters(model)
    print("  [PASS] Model berhasil dibuat dan bobot pretrained resmi terbukti identik pada backbone & ASPP.")
    print("  [PASS] Final classifier head berhasil disesuaikan ke 11 kelas (Conv2d(256, 11, kernel_size=1)).")
    print("\n=== [STATISTIK PARAMETER MODEL] ===")
    for k, v in param_stats.items():
        print(f"  - {k:<18}: {v:11,d} parameter")

    return model


def test_forward_pass(model: CamVidDeepLabV3) -> None:
    """Verify forward pass at resolution 360x480 with batch_size=1 and batch_size=2."""
    print("\n=== [TEST 2] Pengujian Forward Pass (Resolusi 360x480, Batch Size 1 & 2) ===")
    device = next(model.parameters()).device
    h, w = 360, 480

    torch.manual_seed(42)
    model.eval()
    with torch.no_grad():
        for batch_size in (1, 2):
            dummy_input = torch.randn(batch_size, 3, h, w, device=device)
            t0 = time.perf_counter()
            logits = model(dummy_input)
            dt = (time.perf_counter() - t0) * 1000.0

            assert isinstance(logits, torch.Tensor), f"Expected Tensor output, got {type(logits)}"
            assert tuple(logits.shape) == (batch_size, NUM_CLASSES, h, w), (
                f"Expected shape {(batch_size, NUM_CLASSES, h, w)}, got {tuple(logits.shape)}"
            )
            assert logits.dtype == torch.float32
            assert torch.isfinite(logits).all(), "Logits contain NaN or Inf!"

            print(
                f"  [PASS] Eval Forward (batch_size={batch_size}): "
                f"input={tuple(dummy_input.shape)} -> output={tuple(logits.shape)} "
                f"| min={float(logits.min()):.4f}, max={float(logits.max()):.4f} "
                f"| time={dt:.1f} ms"
            )

    # Also verify forward pass in train() mode for both batch_size=1 and batch_size=2
    model.train()
    for batch_size in (1, 2):
        dummy_input = torch.randn(batch_size, 3, h, w, device=device)
        out_dict = model(dummy_input, return_dict=True)
        assert "out" in out_dict and "aux" in out_dict
        assert tuple(out_dict["out"].shape) == (batch_size, NUM_CLASSES, h, w)
        assert tuple(out_dict["aux"].shape) == (batch_size, NUM_CLASSES, h, w)
        assert torch.isfinite(out_dict["out"]).all()
        assert torch.isfinite(out_dict["aux"]).all()
        print(
            f"  [PASS] Train Forward (batch_size={batch_size}): "
            f"out={tuple(out_dict['out'].shape)}, aux={tuple(out_dict['aux'].shape)}"
        )


def test_loss_and_backward_pass(model: CamVidDeepLabV3) -> None:
    """Verify CrossEntropyLoss with ignore_index=255 and backward pass gradients."""
    print("\n=== [TEST 3] Pengujian CrossEntropyLoss (ignore_index=255) & Backward Pass ===")
    device = next(model.parameters()).device
    h, w = 360, 480
    criterion = create_loss_function(ignore_index=IGNORE_INDEX)

    torch.manual_seed(123)
    for batch_size in (1, 2):
        model.train()
        model.zero_grad(set_to_none=True)

        dummy_images = torch.randn(batch_size, 3, h, w, device=device, requires_grad=True)
        # Create dummy target mask with class IDs 0..10 and ignore_index=255 (~10% of pixels)
        dummy_targets = torch.randint(0, NUM_CLASSES, (batch_size, h, w), dtype=torch.long, device=device)
        ignore_mask = torch.rand(batch_size, h, w, device=device) < 0.10
        dummy_targets[ignore_mask] = IGNORE_INDEX

        t0 = time.perf_counter()
        outputs = model(dummy_images, return_dict=True)
        logits = outputs["out"]
        aux_logits = outputs["aux"]

        main_loss = criterion(logits, dummy_targets)
        aux_loss = criterion(aux_logits, dummy_targets)
        total_loss = main_loss + 0.4 * aux_loss

        assert torch.isfinite(main_loss), f"Main loss is not finite: {main_loss}"
        assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss}"
        assert float(main_loss.item()) > 0.0

        total_loss.backward()
        dt = (time.perf_counter() - t0) * 1000.0

        # Check gradients on classifier head, aux head, and backbone
        head_grad = model.classifier[4].weight.grad
        assert head_grad is not None, "Missing gradient on classifier[4].weight!"
        assert torch.isfinite(head_grad).all(), "Non-finite gradient on classifier[4].weight!"
        head_grad_norm = float(head_grad.norm().item())
        assert head_grad_norm > 0.0, "Zero gradient on classifier[4].weight!"

        backbone_grad_norms = [
            float(p.grad.norm().item())
            for p in model.backbone.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(backbone_grad_norms) > 0, "No gradients flowed to backbone!"
        assert all(g >= 0.0 for g in backbone_grad_norms)
        max_bb_grad = max(backbone_grad_norms)
        assert max_bb_grad > 0.0, "All backbone gradients are zero!"

        print(
            f"  [PASS] Loss & Backward (batch_size={batch_size}): "
            f"main_loss={float(main_loss.item()):.4f}, aux_loss={float(aux_loss.item()):.4f}, "
            f"total_loss={float(total_loss.item()):.4f} | "
            f"head_grad_norm={head_grad_norm:.4f}, max_backbone_grad_norm={max_bb_grad:.4f} | "
            f"time={dt:.1f} ms"
        )


def test_checkpoint_roundtrip(model: CamVidDeepLabV3) -> None:
    """Verify save_checkpoint and load_checkpoint using a temporary directory (no fake files left)."""
    print("\n=== [TEST 4] Verifikasi Save & Load Checkpoint (Temporary Directory) ===")
    device = next(model.parameters()).device
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "temp_test_ckpt.pth"
        save_checkpoint(
            ckpt_path,
            model=model,
            epoch=0,
            best_miou=0.0,
            metrics={"note": "unit_test_roundtrip_only"},
        )
        assert ckpt_path.is_file()

        loaded_model, meta = load_checkpoint(ckpt_path, device=device)
        loaded_model.eval()

        assert meta["num_classes"] == NUM_CLASSES == 11
        assert meta["ignore_index"] == IGNORE_INDEX == 255

        x = torch.randn(1, 3, 360, 480, device=device)
        with torch.no_grad():
            y1 = model(x)
            y2 = loaded_model(x)
        assert torch.allclose(y1, y2, atol=1e-6), "Loaded checkpoint outputs differ from original model!"

    print("  [PASS] Fungsi save_checkpoint dan load_checkpoint bekerja identik (file temporer sudah dihapus).")


def main() -> None:
    model = test_environment_and_pretrained_weights()
    test_forward_pass(model)
    test_loss_and_backward_pass(model)
    test_checkpoint_roundtrip(model)
    print("\n>>> SEMUA PENGUJIAN ARSITEKTUR MODEL DEEPLABV3 MOBILENETV3-LARGE LULUS 100% <<<")


if __name__ == "__main__":
    main()
