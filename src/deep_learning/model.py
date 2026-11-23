"""DeepLabV3 with MobileNetV3-Large backbone for 11-class CamVid Semantic Segmentation."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

from .config import CLASS_NAMES_11, IGNORE_INDEX, NUM_CLASSES


@dataclass
class ModelConfig:
    """Configuration for DeepLabV3 MobileNetV3-Large semantic segmentation model."""

    architecture: str = "deeplabv3_mobilenet_v3_large"
    num_classes: int = NUM_CLASSES
    ignore_index: int = IGNORE_INDEX
    pretrained: bool = True
    aux_loss: bool = True
    freeze_backbone: bool = False
    default_input_size: Tuple[int, int] = (360, 480)  # (height, width)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Select CUDA device automatically if available, otherwise fallback to CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CamVidDeepLabV3(nn.Module):
    """
    DeepLabV3 with MobileNetV3-Large backbone adapted for 11-class CamVid segmentation.

    By default, `forward(x)` returns the segmentation logits tensor of shape `(B, 11, H, W)`.
    If `return_dict=True`, it returns a dictionary `{"out": logits, "aux": aux_logits}`.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.config.num_classes}")

        if self.config.pretrained:
            try:
                seg_weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                backbone_weights = MobileNet_V3_Large_Weights.DEFAULT
                base_model = deeplabv3_mobilenet_v3_large(
                    weights=seg_weights,
                    weights_backbone=backbone_weights,
                    aux_loss=self.config.aux_loss,
                )
                self.pretrained_source = (
                    f"{seg_weights} ({seg_weights.url}) | "
                    f"Backbone: {backbone_weights} ({backbone_weights.url})"
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load official torchvision pretrained weights for "
                    "deeplabv3_mobilenet_v3_large. Refusing to silently fallback to "
                    f"random initialization. Original error: {exc}"
                ) from exc
        else:
            base_model = deeplabv3_mobilenet_v3_large(
                weights=None,
                weights_backbone=None,
                aux_loss=self.config.aux_loss,
            )
            self.pretrained_source = "None (random initialization)"

        # Replace final 1x1 conv in DeepLabHead classifier (256 -> num_classes)
        old_head_conv: nn.Conv2d = base_model.classifier[4]
        in_channels = old_head_conv.in_channels
        new_head_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.config.num_classes,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        nn.init.kaiming_normal_(new_head_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_head_conv.bias is not None:
            nn.init.zeros_(new_head_conv.bias)
        base_model.classifier[4] = new_head_conv

        # Replace final 1x1 conv in FCNHead aux_classifier (10 -> num_classes) if present
        if base_model.aux_classifier is not None:
            old_aux_conv: nn.Conv2d = base_model.aux_classifier[4]
            aux_in_channels = old_aux_conv.in_channels
            new_aux_conv = nn.Conv2d(
                in_channels=aux_in_channels,
                out_channels=self.config.num_classes,
                kernel_size=1,
                stride=1,
                bias=True,
            )
            nn.init.kaiming_normal_(new_aux_conv.weight, mode="fan_out", nonlinearity="relu")
            if new_aux_conv.bias is not None:
                nn.init.zeros_(new_aux_conv.bias)
            base_model.aux_classifier[4] = new_aux_conv

        self.model = base_model
        if self.config.freeze_backbone:
            self.set_backbone_trainable(False)

    @property
    def backbone(self) -> nn.Module:
        return self.model.backbone

    @property
    def classifier(self) -> nn.Module:
        return self.model.classifier

    @property
    def aux_classifier(self) -> Optional[nn.Module]:
        return self.model.aux_classifier

    def set_backbone_trainable(self, trainable: bool = True) -> None:
        """
        Freeze or unfreeze MobileNetV3-Large backbone parameters and BatchNorm statistics.

        When `trainable=False` (frozen during Stage 1 warm-up):
        - All backbone parameters have `requires_grad = False`.
        - The backbone submodule is forced into `.eval()` mode so its `BatchNorm2d`
          `running_mean`, `running_var`, and `num_batches_tracked` statistics do not
          change when forward passes run.
        - The classifier head and aux_classifier remain unaffected.
        """
        self.config.freeze_backbone = not trainable
        for param in self.model.backbone.parameters():
            param.requires_grad = trainable

        if not trainable:
            self.model.backbone.eval()
        elif self.training:
            self.model.backbone.train(True)

    def train(self, mode: bool = True) -> "CamVidDeepLabV3":
        """
        Override `nn.Module.train(mode)` so that when `self.config.freeze_backbone` is True,
        `self.model.backbone` remains in `.eval()` mode (preserving BatchNorm running stats),
        while `self.model.classifier` and `self.model.aux_classifier` enter `.train(mode)`.
        """
        super().train(mode)
        if mode and self.config.freeze_backbone:
            self.model.backbone.eval()
        return self

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """
        Return parameter groups with differential learning rates for backbone vs segmentation heads.
        """
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.model.classifier.parameters() if p.requires_grad]
        if self.model.aux_classifier is not None:
            head_params.extend(
                p for p in self.model.aux_classifier.parameters() if p.requires_grad
            )

        groups: List[Dict[str, Any]] = []
        if backbone_params:
            groups.append(
                {
                    "name": "backbone",
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "weight_decay": weight_decay,
                }
            )
        if head_params:
            groups.append(
                {
                    "name": "classifier_head",
                    "params": head_params,
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                }
            )
        return groups

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run forward pass on batch tensor `x` of shape `(B, 3, H, W)`.

        Note: In DeepLabV3's `ASPPPooling`, `AdaptiveAvgPool2d(1)` produces a `(B, 256, 1, 1)`
        tensor before `BatchNorm2d`. When `self.training` is True and `B == 1`, PyTorch's
        `BatchNorm2d` would fail on a `1x1` spatial map unless temporarily evaluated with
        running stats. We handle that automatically so `batch_size=1` works in both
        `.train()` and `.eval()` modes.
        """
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(f"Expected input tensor of shape (B, 3, H, W), got {tuple(x.shape)}")

        aspp_pool_bn: nn.Module = self.model.classifier[0].convs[4][2]
        restore_bn_train = False
        if self.training and x.size(0) == 1 and aspp_pool_bn.training:
            aspp_pool_bn.eval()
            restore_bn_train = True

        try:
            outputs: Dict[str, torch.Tensor] = self.model(x)
        finally:
            if restore_bn_train:
                aspp_pool_bn.train()

        if return_dict:
            return outputs
        return outputs["out"]


def create_deeplabv3_mobilenet_v3_large(
    config: Optional[ModelConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> CamVidDeepLabV3:
    """Factory function to create the 11-class DeepLabV3 MobileNetV3-Large model."""
    if config is None:
        config = ModelConfig(**kwargs)
    elif kwargs:
        cfg_dict = config.to_dict()
        cfg_dict.update(kwargs)
        config = ModelConfig(**cfg_dict)

    model = CamVidDeepLabV3(config=config)
    if device is not None:
        model = model.to(torch.device(device))
    return model


def create_loss_function(
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
) -> nn.CrossEntropyLoss:
    """Create CrossEntropyLoss configured with `ignore_index=255`."""
    return nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


def count_parameters(model: CamVidDeepLabV3) -> Dict[str, int]:
    """Count total, trainable, frozen, and per-submodule parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    backbone_total = sum(p.numel() for p in model.backbone.parameters())
    classifier_total = sum(p.numel() for p in model.classifier.parameters())
    aux_total = (
        sum(p.numel() for p in model.aux_classifier.parameters())
        if model.aux_classifier is not None
        else 0
    )
    final_head_conv = sum(p.numel() for p in model.classifier[4].parameters())

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "backbone": backbone_total,
        "classifier_head": classifier_total,
        "final_1x1_conv": final_head_conv,
        "aux_classifier": aux_total,
    }


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: CamVidDeepLabV3,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_miou: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model state_dict, configuration, and training metadata to a checkpoint file."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "architecture": model.config.architecture,
        "num_classes": model.config.num_classes,
        "ignore_index": model.config.ignore_index,
        "class_names": CLASS_NAMES_11,
        "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, path)
    return path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[CamVidDeepLabV3] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Tuple[CamVidDeepLabV3, Dict[str, Any]]:
    """
    Load a trained CamVidDeepLabV3 model from a checkpoint file.

    Supports both full checkpoint dictionaries (containing `'model_state_dict'`)
    and plain state_dict files.
    """
    target_device = torch.device(device) if device is not None else get_device()
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

    if model is None:
        cfg_dict = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        # When loading from a local checkpoint, we do not need to re-download torchvision weights
        cfg_dict["pretrained"] = False
        config = ModelConfig(**cfg_dict) if cfg_dict else ModelConfig(pretrained=False)
        model = CamVidDeepLabV3(config=config)

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    # Allow loading either wrapped keys ('model.backbone...') or raw torchvision keys ('backbone...')
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    if loaded_keys and not (loaded_keys & model_keys):
        if all(not k.startswith("model.") for k in loaded_keys):
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)
    model = model.to(target_device)

    metadata = (
        {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        if isinstance(checkpoint, dict)
        else {}
    )
    return model, metadata
