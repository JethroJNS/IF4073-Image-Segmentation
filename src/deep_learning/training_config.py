"""Configuration dataclass for two-stage transfer learning on CamVid (11 classes)."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import CAMVID_DIR, IGNORE_INDEX, MODELS_DIR, NUM_CLASSES, RESULTS_DIR


@dataclass
class TrainingConfig:
    """
    Configuration for two-stage DeepLabV3 MobileNetV3-Large training on CamVid.

    Stage 1 (Classifier Warm-Up):
        - Backbone frozen
        - Head learning rate = 1e-3
        - Default epochs = 5

    Stage 2 (Fine-Tuning):
        - Backbone unfrozen
        - Backbone learning rate = 1e-4
        - Head learning rate = 5e-4
        - Default epochs = 20
    """

    # Dataset & hardware configuration
    data_dir: str = str(CAMVID_DIR)
    image_size: Tuple[int, int] = (360, 480)  # (height, width)
    batch_size: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    device: str = "auto"  # 'auto', 'cpu', or 'cuda'
    seed: int = 42

    # Model configuration
    num_classes: int = NUM_CLASSES
    ignore_index: int = IGNORE_INDEX
    pretrained: bool = True
    aux_loss: bool = True
    aux_loss_weight: float = 0.4

    # Stage 1: Classifier warm-up
    warmup_epochs: int = 5
    warmup_head_lr: float = 1e-3

    # Stage 2: Full fine-tuning
    finetune_epochs: int = 20
    finetune_backbone_lr: float = 1e-4
    finetune_head_lr: float = 5e-4

    # Optimizer, regularizer & scheduler
    weight_decay: float = 1e-4
    use_class_weights: bool = True
    class_weight_method: str = "enet"  # 'enet', 'median_freq_sqrt', or 'uniform'
    class_weight_min: float = 0.25
    class_weight_max: float = 5.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    min_lr: float = 1e-6
    grad_clip_norm: Optional[float] = 5.0
    use_augmentation: bool = True

    # Output & checkpoint configuration
    models_dir: str = str(MODELS_DIR)
    results_dir: str = str(RESULTS_DIR / "training")
    experiment_name: str = "deeplabv3_mobilenetv3_camvid11"
    overwrite: bool = False

    @property
    def total_epochs(self) -> int:
        return self.warmup_epochs + self.finetune_epochs

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["image_size"] = list(self.image_size)
        d["total_epochs"] = self.total_epochs
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        if "image_size" in filtered and isinstance(filtered["image_size"], list):
            filtered["image_size"] = tuple(filtered["image_size"])
        return cls(**filtered)
