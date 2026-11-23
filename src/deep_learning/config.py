"""Configuration and 32-to-11 class mapping for the CamVid dataset."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CAMVID_DIR = DATA_DIR / "CamVid"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Split configurations (fixed native CamVid split: 421 / 112 / 168)
SPLIT_DIRS: Dict[str, Tuple[str, str]] = {
    "train": ("train", "train_labels"),
    "val": ("val", "val_labels"),
    "test": ("test", "test_labels"),
}
EXPECTED_SPLIT_COUNTS: Dict[str, int] = {
    "train": 421,
    "val": 112,
    "test": 168,
}
IMAGE_RESOLUTION: Tuple[int, int] = (960, 720)  # (width, height)

# Target 11 classes specification (IDs 0 to 10) and ignore_index = 255
NUM_CLASSES: int = 11
IGNORE_INDEX: int = 255
VOID_COLOR: Tuple[int, int, int] = (0, 0, 0)

CLASS_NAMES_11: List[str] = [
    "Sky",         # 0
    "Building",    # 1
    "Pole",        # 2
    "Road",        # 3
    "Pavement",    # 4
    "Tree",        # 5
    "SignSymbol",  # 6
    "Fence",       # 7
    "Car",         # 8
    "Pedestrian",  # 9
    "Bicyclist",   # 10
]

CLASS_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES_11)}
CLASS_ID_TO_NAME: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES_11)}
CLASS_ID_TO_NAME[IGNORE_INDEX] = "Void/Ignore"

# Official representative RGB colors for the 11 target classes
CLASS_COLORS_11: List[Tuple[int, int, int]] = [
    (128, 128, 128),  # 0: Sky
    (128, 0, 0),      # 1: Building
    (192, 192, 128),  # 2: Pole
    (128, 64, 128),   # 3: Road
    (0, 0, 192),      # 4: Pavement
    (128, 128, 0),    # 5: Tree
    (192, 128, 128),  # 6: SignSymbol
    (64, 64, 128),    # 7: Fence
    (64, 0, 128),     # 8: Car
    (64, 64, 0),      # 9: Pedestrian
    (0, 128, 192),    # 10: Bicyclist
]

CLASS_ID_TO_COLOR: Dict[int, Tuple[int, int, int]] = {
    idx: color for idx, color in enumerate(CLASS_COLORS_11)
}
CLASS_ID_TO_COLOR[IGNORE_INDEX] = VOID_COLOR

# Official 32 original CamVid classes and their RGB values (from label_colors.txt & class_dict.csv)
RAW_32_CLASSES: Dict[str, Tuple[int, int, int]] = {
    "Animal": (64, 128, 64),
    "Archway": (192, 0, 128),
    "Bicyclist": (0, 128, 192),
    "Bridge": (0, 128, 64),
    "Building": (128, 0, 0),
    "Car": (64, 0, 128),
    "CartLuggagePram": (64, 0, 192),
    "Child": (192, 128, 64),
    "Column_Pole": (192, 192, 128),
    "Fence": (64, 64, 128),
    "LaneMkgsDriv": (128, 0, 192),
    "LaneMkgsNonDriv": (192, 0, 64),
    "Misc_Text": (128, 128, 64),
    "MotorcycleScooter": (192, 0, 192),
    "OtherMoving": (128, 64, 64),
    "ParkingBlock": (64, 192, 128),
    "Pedestrian": (64, 64, 0),
    "Road": (128, 64, 128),
    "RoadShoulder": (128, 128, 192),
    "Sidewalk": (0, 0, 192),
    "SignSymbol": (192, 128, 128),
    "Sky": (128, 128, 128),
    "SUVPickupTruck": (64, 128, 192),
    "TrafficCone": (0, 0, 64),
    "TrafficLight": (0, 64, 64),
    "Train": (192, 64, 128),
    "Tree": (128, 128, 0),
    "Truck_Bus": (192, 128, 192),
    "Tunnel": (64, 0, 64),
    "VegetationMisc": (192, 192, 0),
    "Void": (0, 0, 0),
    "Wall": (64, 192, 0),
}

# Grouping from 11 target classes to the 32 raw CamVid classes
GROUPING_11_TO_32: Dict[str, List[str]] = {
    "Sky": ["Sky"],
    "Building": ["Building", "Wall", "Bridge", "Tunnel", "Archway"],
    "Pole": ["Column_Pole", "TrafficCone"],
    "Road": ["Road", "LaneMkgsDriv", "LaneMkgsNonDriv"],
    "Pavement": ["Sidewalk", "ParkingBlock", "RoadShoulder"],
    "Tree": ["Tree", "VegetationMisc"],
    "SignSymbol": ["SignSymbol", "Misc_Text", "TrafficLight"],
    "Fence": ["Fence"],
    "Car": ["Car", "SUVPickupTruck", "Truck_Bus", "Train", "OtherMoving"],
    "Pedestrian": ["Pedestrian", "Child", "CartLuggagePram"],
    "Bicyclist": ["Bicyclist", "MotorcycleScooter"],
}

# Classes explicitly mapped to Void/Ignore (255)
IGNORE_Classes_32: List[str] = ["Void", "Animal"]

# Build reverse lookup maps: Raw 32 class name -> Target 11 class name & ID
RAW_32_TO_11_NAME: Dict[str, str] = {}
RAW_32_TO_11_ID: Dict[str, int] = {}

for target_name, raw_list in GROUPING_11_TO_32.items():
    target_id = CLASS_NAME_TO_ID[target_name]
    for raw_name in raw_list:
        if raw_name not in RAW_32_CLASSES:
            raise KeyError(f"Unknown raw class '{raw_name}' in GROUPING_11_TO_32")
        if raw_name in RAW_32_TO_11_ID:
            raise ValueError(f"Duplicate mapping for raw class '{raw_name}'")
        RAW_32_TO_11_NAME[raw_name] = target_name
        RAW_32_TO_11_ID[raw_name] = target_id

for raw_name in IGNORE_Classes_32:
    if raw_name not in RAW_32_CLASSES:
        raise KeyError(f"Unknown raw class '{raw_name}' in IGNORE_Classes_32")
    if raw_name in RAW_32_TO_11_ID:
        raise ValueError(f"Duplicate mapping for raw class '{raw_name}'")
    RAW_32_TO_11_NAME[raw_name] = "Void/Ignore"
    RAW_32_TO_11_ID[raw_name] = IGNORE_INDEX

if len(RAW_32_TO_11_ID) != 32:
    raise AssertionError(
        f"Expected all 32 raw classes to be mapped, got {len(RAW_32_TO_11_ID)}"
    )

# Map RGB tuple -> target class ID (0..10 or 255)
RGB_TO_CLASS_ID: Dict[Tuple[int, int, int], int] = {
    rgb: RAW_32_TO_11_ID[raw_name] for raw_name, rgb in RAW_32_CLASSES.items()
}


def build_rgb_lookup_table() -> np.ndarray:
    """
    Build a 256x256x256 uint8 lookup table mapping any (R, G, B) triplet
    in O(1) time to its target class ID (0..10), with default IGNORE_INDEX (255)
    for Void, Animal, and any unknown/non-palette RGB colors.
    """
    lut = np.full((256, 256, 256), IGNORE_INDEX, dtype=np.uint8)
    for (r, g, b), class_id in RGB_TO_CLASS_ID.items():
        lut[r, g, b] = class_id
    return lut


def build_id_to_rgb_table() -> np.ndarray:
    """
    Build a 256x3 uint8 lookup table mapping class IDs (0..10 and 255)
    to their visualization RGB colors.
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for class_id, (r, g, b) in CLASS_ID_TO_COLOR.items():
        lut[class_id] = (r, g, b)
    return lut


# Precomputed lookup tables for fast vectorized conversion
RGB_TO_ID_LUT: np.ndarray = build_rgb_lookup_table()
ID_TO_RGB_LUT: np.ndarray = build_id_to_rgb_table()

# Standard ImageNet normalization constants for pretrained CNNs
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
