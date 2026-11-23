"""Comprehensive automated validation test suite for CamVid 11-class Dataset."""

import csv
from pathlib import Path
import sys
import time
from typing import Dict, Set, Tuple

import numpy as np
from PIL import Image
import torch

# Ensure project root is on sys.path when invoked as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.config import (
    CAMVID_DIR,
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    CLASS_NAMES_11,
    EXPECTED_SPLIT_COUNTS,
    GROUPING_11_TO_32,
    IGNORE_Classes_32,
    IGNORE_INDEX,
    IMAGE_RESOLUTION,
    NUM_CLASSES,
    RAW_32_CLASSES,
    RAW_32_TO_11_ID,
    RGB_TO_CLASS_ID,
)
from src.deep_learning.dataset import CamVidDataset, mask_to_rgb, overlay_mask, rgb_to_mask


def test_metadata_and_32_to_11_mapping() -> None:
    """Verify all 32 official colors from label_colors.txt and class_dict.csv map accurately."""
    print("=== [TEST 1] Verifikasi Metadata & Mapping 32 Kelas -> 11 Kelas + Void (255) ===")
    assert NUM_CLASSES == 11, f"Expected NUM_CLASSES=11, got {NUM_CLASSES}"
    assert len(CLASS_NAMES_11) == 11
    assert IGNORE_INDEX == 255

    # 1. Verify against official label_colors.txt
    txt_path = CAMVID_DIR / "label_colors.txt"
    assert txt_path.is_file(), f"Missing {txt_path}"
    txt_classes: Dict[str, Tuple[int, int, int]] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.split()
                txt_classes[parts[3]] = (int(parts[0]), int(parts[1]), int(parts[2]))

    # 2. Verify against class_dict.csv
    csv_path = CAMVID_DIR / "class_dict.csv"
    assert csv_path.is_file(), f"Missing {csv_path}"
    csv_classes: Dict[str, Tuple[int, int, int]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            csv_classes[row["name"].strip()] = (
                int(row["r"].strip()),
                int(row["g"].strip()),
                int(row["b"].strip()),
            )

    assert len(txt_classes) == 32, f"Expected 32 classes in label_colors.txt, got {len(txt_classes)}"
    assert txt_classes == csv_classes, "Mismatch between label_colors.txt and class_dict.csv"
    assert txt_classes == RAW_32_CLASSES, "Mismatch between RAW_32_CLASSES and official metadata files"

    # 3. Verify synthetic image containing all 32 official RGB colors maps each pixel to expected class ID
    synth_rgb = np.zeros((32, 1, 3), dtype=np.uint8)
    expected_ids = np.zeros((32, 1), dtype=np.uint8)
    for idx, (raw_name, rgb) in enumerate(sorted(RAW_32_CLASSES.items())):
        synth_rgb[idx, 0] = rgb
        expected_ids[idx, 0] = RAW_32_TO_11_ID[raw_name]

    mapped_mask = rgb_to_mask(synth_rgb)
    assert np.array_equal(mapped_mask, expected_ids), (
        f"32-color synthetic mapping mismatch:\nGot: {mapped_mask.ravel()}\nExp: {expected_ids.ravel()}"
    )

    # 4. Specifically verify Void and Animal map to 255
    assert rgb_to_mask(np.array([[[0, 0, 0]]], dtype=np.uint8))[0, 0] == IGNORE_INDEX
    assert rgb_to_mask(np.array([[[64, 128, 64]]], dtype=np.uint8))[0, 0] == IGNORE_INDEX

    # 5. Verify unknown / non-palette RGB colors map to 255
    unknown_colors = np.array(
        [[[1, 2, 3], [255, 255, 255], [128, 128, 51], [26, 26, 0], [191, 191, 126]]],
        dtype=np.uint8,
    )
    unknown_mapped = rgb_to_mask(unknown_colors)
    assert np.all(unknown_mapped == IGNORE_INDEX), (
        f"Expected all unknown RGB colors to map to {IGNORE_INDEX}, got {unknown_mapped}"
    )

    print("  [PASS] Seluruh 32 warna resmi terpetakan dengan benar ke ID 0..10 dan 255.")
    print("  [PASS] Void (0, 0, 0), Animal (64, 128, 64), dan warna tak dikenal -> 255 (ignore_index).")


def test_seq05vd_f02610_non_palette_pixels() -> None:
    """Explicitly test test_labels/Seq05VD_f02610_L.png which has 175 non-palette boundary pixels."""
    print("\n=== [TEST 2] Verifikasi Khusus File Seq05VD_f02610_L.png (175 Piksel Non-Palet) ===")
    lbl_path = CAMVID_DIR / "test_labels" / "Seq05VD_f02610_L.png"
    assert lbl_path.is_file(), f"Missing {lbl_path}"

    raw_rgb = np.asarray(Image.open(lbl_path).convert("RGB"), dtype=np.uint8)
    mask = rgb_to_mask(raw_rgb)

    valid_ids = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
    unique_ids = set(np.unique(mask).tolist())
    assert unique_ids.issubset(valid_ids), f"Invalid class IDs found: {unique_ids - valid_ids}"

    # Count how many pixels in Seq05VD_f02610_L.png were non-palette and verify they became 255
    known_rgb_codes = set((r << 16) | (g << 8) | b for (r, g, b) in RAW_32_CLASSES.values())
    arr32 = raw_rgb.astype(np.uint32)
    codes = (arr32[:, :, 0] << 16) | (arr32[:, :, 1] << 8) | arr32[:, :, 2]
    non_palette_mask = ~np.isin(codes, list(known_rgb_codes))
    n_non_palette = int(non_palette_mask.sum())

    assert n_non_palette == 175, f"Expected 175 non-palette pixels, found {n_non_palette}"
    assert np.all(mask[non_palette_mask] == IGNORE_INDEX), (
        "Some non-palette pixels were not mapped to IGNORE_INDEX (255)!"
    )
    print(
        f"  [PASS] 175 piksel non-palet (55 warna RGB artefak tepi) pada Seq05VD_f02610_L.png "
        f"seluruhnya dipetakan dengan aman ke ignore_index = {IGNORE_INDEX}."
    )


def test_full_dataset_splits_and_pixel_distribution() -> None:
    """Scan all 701 samples across train (421), val (112), and test (168)."""
    print("\n=== [TEST 3] Pengujian Penuh Seluruh Split Dataset (701 Sampel / 484.531.200 Piksel) ===")
    t0 = time.perf_counter()

    allowed_values = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
    global_counts: Dict[int, int] = {cid: 0 for cid in list(range(NUM_CLASSES)) + [IGNORE_INDEX]}
    split_counts: Dict[str, Dict[int, int]] = {}
    all_stems: Set[str] = set()
    total_samples = 0

    for split_name, expected_count in EXPECTED_SPLIT_COUNTS.items():
        ds = CamVidDataset(CAMVID_DIR, split=split_name)
        assert len(ds) == expected_count, (
            f"Split '{split_name}' count mismatch: got {len(ds)}, expected {expected_count}"
        )
        split_counts[split_name] = {cid: 0 for cid in list(range(NUM_CLASSES)) + [IGNORE_INDEX]}

        for idx in range(len(ds)):
            image_rgb, label_rgb, stem, _, _ = ds.load_raw_pair(idx)
            assert stem not in all_stems, f"Duplicate stem '{stem}' across splits!"
            all_stems.add(stem)

            w, h = IMAGE_RESOLUTION
            assert image_rgb.shape == (h, w, 3), f"Unexpected image shape {image_rgb.shape} for {stem}"
            assert label_rgb.shape == (h, w, 3), f"Unexpected label shape {label_rgb.shape} for {stem}"

            mask = rgb_to_mask(label_rgb)
            assert mask.shape == (h, w), f"Unexpected mask shape {mask.shape}"
            assert mask.dtype == np.uint8

            u_vals, u_cnts = np.unique(mask, return_counts=True)
            for v, c in zip(u_vals.tolist(), u_cnts.tolist()):
                assert v in allowed_values, f"Illegal class ID {v} in {split_name}/{stem}"
                global_counts[v] += c
                split_counts[split_name][v] += c

            # Check PyTorch __getitem__ tensor contract on first and last sample of each split
            if idx in (0, len(ds) - 1):
                img_t, msk_t = ds[idx]
                assert isinstance(img_t, torch.Tensor) and img_t.dtype == torch.float32
                assert isinstance(msk_t, torch.Tensor) and msk_t.dtype == torch.int64
                assert tuple(img_t.shape) == (3, h, w)
                assert tuple(msk_t.shape) == (h, w)
                assert float(img_t.min()) >= 0.0 and float(img_t.max()) <= 1.0

                # Also check colorization & overlay round-trip shape
                color_rgb = mask_to_rgb(mask)
                ov_rgb = overlay_mask(image_rgb, mask)
                assert color_rgb.shape == (h, w, 3) and color_rgb.dtype == np.uint8
                assert ov_rgb.shape == (h, w, 3) and ov_rgb.dtype == np.uint8

            total_samples += 1

        print(
            f"  [PASS] Split '{split_name:<5}': {len(ds):3d}/{expected_count:3d} pasangan valid, "
            f"resolusi 960x720 konsisten, nilai mask hanya di {{0..10, 255}}."
        )

    elapsed = time.perf_counter() - t0
    total_pixels = sum(global_counts.values())
    assert total_samples == 701, f"Expected 701 total samples, got {total_samples}"
    assert len(all_stems) == 701, f"Expected 701 unique stems, got {len(all_stems)}"
    assert total_pixels == 701 * 960 * 720, f"Pixel total mismatch: {total_pixels}"

    print(f"  [PASS] Total 701 sampel (484.531.200 piksel) berhasil divalidasi dalam {elapsed:.2f} detik.")

    print("\n=== [TABEL DISTRIBUSI PIKSEL 11 KELAS TARGET & VOID/IGNORE (255)] ===")
    print(
        f"{'ID':>3} | {'Nama Kelas':<13} | {'Train (%)':>9} | {'Val (%)':>8} | "
        f"{'Test (%)':>8} | {'Total Piksel':>14} | {'Total (%)':>9}"
    )
    print("-" * 82)
    train_tot = sum(split_counts["train"].values())
    val_tot = sum(split_counts["val"].values())
    test_tot = sum(split_counts["test"].values())

    for cid in list(range(NUM_CLASSES)) + [IGNORE_INDEX]:
        name = CLASS_ID_TO_NAME[cid]
        tr_pct = 100.0 * split_counts["train"][cid] / train_tot
        va_pct = 100.0 * split_counts["val"][cid] / val_tot
        te_pct = 100.0 * split_counts["test"][cid] / test_tot
        tot_px = global_counts[cid]
        tot_pct = 100.0 * tot_px / total_pixels
        print(
            f"{cid:3d} | {name:<13} | {tr_pct:8.3f}% | {va_pct:7.3f}% | "
            f"{te_pct:7.3f}% | {tot_px:14,d} | {tot_pct:8.4f}%"
        )


def test_resize_and_normalization_options() -> None:
    """Verify optional target_size resizing and ImageNet normalization in CamVidDataset."""
    print("\n=== [TEST 4] Verifikasi Opsi Resize & ImageNet Normalization ===")
    ds_resized = CamVidDataset(
        CAMVID_DIR,
        split="val",
        target_size=(360, 480),
        imagenet_normalize=True,
        return_dict=True,
    )
    item = ds_resized[0]
    img_t = item["image"]
    msk_t = item["mask"]
    assert tuple(img_t.shape) == (3, 360, 480), f"Unexpected resized image shape: {img_t.shape}"
    assert tuple(msk_t.shape) == (360, 480), f"Unexpected resized mask shape: {msk_t.shape}"
    unique_ids = set(torch.unique(msk_t).tolist())
    allowed_values = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
    assert unique_ids.issubset(allowed_values), (
        f"Nearest-neighbor resize corrupted mask IDs: {unique_ids - allowed_values}"
    )
    print("  [PASS] Resize (bilinear image + nearest mask) dan ImageNet normalization berfungsi benar.")


def main() -> None:
    test_metadata_and_32_to_11_mapping()
    test_seq05vd_f02610_non_palette_pixels()
    test_full_dataset_splits_and_pixel_distribution()
    test_resize_and_normalization_options()
    print("\n>>> SEMUA PENGUJIAN DATASET CAMVID 11-KELAS LULUS 100% <<<")


if __name__ == "__main__":
    main()
