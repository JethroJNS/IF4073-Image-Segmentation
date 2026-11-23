"""PyTorch Dataset and fast vectorized label conversion for CamVid (11 classes)."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .config import (
    CAMVID_DIR,
    ID_TO_RGB_LUT,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RGB_TO_ID_LUT,
    SPLIT_DIRS,
)


def rgb_to_mask(rgb_label: np.ndarray) -> np.ndarray:
    """
    Convert an RGB color-coded label array of shape (H, W, 3) and dtype uint8
    into a 2D class ID mask of shape (H, W) and dtype uint8.

    Values in the returned mask are in {0..10, 255}, where 255 is IGNORE_INDEX
    (covering Void, Animal, and any unknown/non-palette RGB colors).
    Uses precomputed 3D lookup table indexing with zero Python loops.
    """
    if rgb_label.ndim != 3 or rgb_label.shape[2] != 3:
        raise ValueError(
            f"Expected RGB label of shape (H, W, 3), got {rgb_label.shape}"
        )
    if rgb_label.dtype != np.uint8:
        rgb_label = rgb_label.astype(np.uint8, copy=False)

    return RGB_TO_ID_LUT[rgb_label[:, :, 0], rgb_label[:, :, 1], rgb_label[:, :, 2]]


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convert a 2D class ID mask of shape (H, W) with values in {0..10, 255}
    into an RGB color image of shape (H, W, 3) and dtype uint8 using the
    11-class color palette (255 -> black / Void).
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D class mask of shape (H, W), got {mask.shape}")
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    return ID_TO_RGB_LUT[mask_u8]


def overlay_mask(
    image_rgb: np.ndarray,
    mask_or_rgb: np.ndarray,
    alpha: float = 0.45,
    ignore_transparent: bool = True,
) -> np.ndarray:
    """
    Create an alpha-blended overlay of an RGB image and either a 2D class ID mask
    or a 3D colorized RGB mask.

    If `ignore_transparent=True` and a 2D class mask is provided, pixels with
    `IGNORE_INDEX` (255) keep their original image appearance without darkening.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected image_rgb of shape (H, W, 3), got {image_rgb.shape}")

    if mask_or_rgb.ndim == 2:
        color_mask = mask_to_rgb(mask_or_rgb)
        valid_region = (mask_or_rgb != IGNORE_INDEX) if ignore_transparent else None
    elif mask_or_rgb.ndim == 3 and mask_or_rgb.shape[2] == 3:
        color_mask = mask_or_rgb
        valid_region = None
    else:
        raise ValueError(
            f"Expected mask_or_rgb of shape (H, W) or (H, W, 3), got {mask_or_rgb.shape}"
        )

    img_f = image_rgb.astype(np.float32)
    msk_f = color_mask.astype(np.float32)
    blended = (1.0 - alpha) * img_f + alpha * msk_f

    if valid_region is not None:
        out = np.where(valid_region[:, :, None], blended, img_f)
    else:
        out = blended

    return np.clip(out, 0, 255).astype(np.uint8)


class CamVidDataset(Dataset):
    """
    PyTorch Dataset for CamVid 11-class Semantic Segmentation.

    Supports splits: 'train' (421), 'val' (112), 'test' (168).
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = CAMVID_DIR,
        split: str = "train",
        target_size: Optional[Tuple[int, int]] = None,
        imagenet_normalize: bool = False,
        transform: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
        return_dict: bool = False,
    ) -> None:
        """
        Args:
            root_dir: Path to `data/CamVid`.
            split: One of `'train'`, `'val'`, or `'test'`.
            target_size: Optional `(height, width)` to resize image (bilinear) and mask (nearest).
            imagenet_normalize: If True, normalize image tensor using ImageNet mean & std.
            transform: Optional callable `(image_np, mask_np) -> (image_np, mask_np)`.
            return_dict: If True, `__getitem__` returns a dictionary with tensors and metadata;
                otherwise returns `(image_tensor, mask_tensor)`.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower().strip()
        if self.split not in SPLIT_DIRS:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of {list(SPLIT_DIRS.keys())}"
            )

        self.target_size = target_size  # (H, W)
        self.imagenet_normalize = imagenet_normalize
        self.transform = transform
        self.return_dict = return_dict

        img_subdir, lbl_subdir = SPLIT_DIRS[self.split]
        self.image_dir = self.root_dir / img_subdir
        self.label_dir = self.root_dir / lbl_subdir

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.samples: List[Tuple[Path, Path, str]] = self._discover_and_validate_pairs()

        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def _discover_and_validate_pairs(self) -> List[Tuple[Path, Path, str]]:
        """Discover and verify 1-to-1 pairing between images `<stem>.png` and labels `<stem>_L.png`."""
        img_files = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        )
        lbl_files = sorted(
            p for p in self.label_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        )

        img_map: Dict[str, Path] = {p.stem: p for p in img_files}
        lbl_map: Dict[str, Path] = {}
        for p in lbl_files:
            stem = p.stem[:-2] if p.stem.endswith("_L") else p.stem
            lbl_map[stem] = p

        missing_labels = sorted(set(img_map.keys()) - set(lbl_map.keys()))
        missing_images = sorted(set(lbl_map.keys()) - set(img_map.keys()))
        if missing_labels or missing_images:
            raise RuntimeError(
                f"Split '{self.split}' has mismatched pairs: "
                f"missing_labels={missing_labels[:5]}, missing_images={missing_images[:5]}"
            )

        samples = [
            (img_map[stem], lbl_map[stem], stem)
            for stem in sorted(img_map.keys())
        ]
        if not samples:
            raise RuntimeError(f"No valid image-label pairs found in {self.image_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def load_raw_pair(self, index: int) -> Tuple[np.ndarray, np.ndarray, str, Path, Path]:
        """Load raw RGB image `(H, W, 3)` and raw RGB label `(H, W, 3)` as uint8 NumPy arrays."""
        img_path, lbl_path, stem = self.samples[index]
        with Image.open(img_path) as im:
            image_rgb = np.array(im.convert("RGB"), dtype=np.uint8, copy=True)
        with Image.open(lbl_path) as lb:
            label_rgb = np.array(lb.convert("RGB"), dtype=np.uint8, copy=True)

        if image_rgb.shape != label_rgb.shape:
            raise ValueError(
                f"Shape mismatch for sample '{stem}': "
                f"image={image_rgb.shape} vs label={label_rgb.shape}"
            )
        return image_rgb, label_rgb, stem, img_path, lbl_path

    def get_raw_sample(self, index: int) -> Dict[str, Any]:
        """
        Return full unnormalized NumPy arrays and metadata for visualization and inspection.
        """
        image_rgb, raw_label_rgb, stem, img_path, lbl_path = self.load_raw_pair(index)
        mask = rgb_to_mask(raw_label_rgb)
        mapped_label_rgb = mask_to_rgb(mask)
        overlay_rgb = overlay_mask(image_rgb, mask, alpha=0.45)

        return {
            "index": index,
            "split": self.split,
            "stem": stem,
            "image_path": str(img_path),
            "label_path": str(lbl_path),
            "image": image_rgb,
            "raw_label_rgb": raw_label_rgb,
            "mask": mask,
            "mapped_label_rgb": mapped_label_rgb,
            "overlay": overlay_rgb,
        }

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        image_rgb, raw_label_rgb, stem, img_path, lbl_path = self.load_raw_pair(index)
        mask = rgb_to_mask(raw_label_rgb)

        if self.target_size is not None:
            target_h, target_w = self.target_size
            if (image_rgb.shape[0], image_rgb.shape[1]) != (target_h, target_w):
                image_pil = Image.fromarray(image_rgb)
                mask_pil = Image.fromarray(mask)
                image_rgb = np.array(
                    image_pil.resize((target_w, target_h), resample=Image.BILINEAR),
                    dtype=np.uint8,
                    copy=True,
                )
                mask = np.array(
                    mask_pil.resize((target_w, target_h), resample=Image.NEAREST),
                    dtype=np.uint8,
                    copy=True,
                )

        if self.transform is not None:
            image_rgb, mask = self.transform(image_rgb, mask)

        # Convert image (H, W, 3) uint8 -> (3, H, W) float32 in [0.0, 1.0]
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image_rgb))
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        if self.imagenet_normalize:
            image_tensor = (image_tensor - self._mean) / self._std

        # Convert mask (H, W) uint8 -> (H, W) int64 for CrossEntropyLoss(ignore_index=255)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).to(dtype=torch.long)

        if self.return_dict:
            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "stem": stem,
                "split": self.split,
                "image_path": str(img_path),
                "label_path": str(lbl_path),
            }
        return image_tensor, mask_tensor
