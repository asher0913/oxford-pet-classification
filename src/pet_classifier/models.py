"""Model definitions.

PetResNet is the custom CNN for custom-model track - residual blocks with SE
channel attention, ~2.7M params. It is kept small because the training
set is only ~3k images after the val split.

The transfer learning helpers wrap torchvision models for transfer-learning track:
swap the classifier for a 37-way layer, optionally freeze the backbone,
and split params into backbone vs head groups so the head can use a
larger LR (10x by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torchvision import models


TRANSFER_MODELS = {
    # Only list models that the rest of the replacement code knows how to edit.
    "resnet18", "resnet34", "resnet50",
    "vgg16", "vgg19",
    "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
}

MODEL_CHOICES = ["custom", *sorted(TRANSFER_MODELS)]


@dataclass(frozen=True)
class ModelInfo:
    """The built model plus the choices used to make it."""

    # what build_model returns
    model: nn.Module
    model_name: str
    pretrained: bool
    freeze_backbone: bool
    num_classes: int


# ----- building blocks for PetResNet -----


class SqueezeExcite(nn.Module):
    """SE channel attention block used in the deeper custom stages."""

    # Channel attention. GAP -> tiny MLP -> sigmoid -> per-channel weight.
    # reduction=16 is the value from the paper.

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Set up the tiny MLP that produces per-channel weights."""

        super().__init__()
        # Keep at least a few hidden units so small channel counts still work.
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel weights to the incoming feature map."""

        batch, channels, _, _ = x.shape
        squeeze = self.pool(x).view(batch, channels)  # [B,C,H,W] -> [B,C]
        excitation = self.fc(squeeze).view(batch, channels, 1, 1)
        return x * excitation


class BasicResBlock(nn.Module):
    """Two-conv residual block used throughout PetResNet."""

    # Conv-BN-ReLU-Conv-BN, then add the input back and ReLU.
    # If shapes change the shortcut needs a 1x1 conv.

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Build the main path and the shortcut path."""

        super().__init__()

        # bias=False since BN already has a learnable shift
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual connection and final ReLU."""

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        return self.relu(out)


# ---------------------------------------------------------------------------
# PetResNet: the custom CNN (custom-model track)
# ---------------------------------------------------------------------------


class PetResNet(nn.Module):
    """Custom CNN for 37-way pet classification.
    
        For a 160×160 input, the model starts with a 
        simple convolutional stem and then goes through three stages 
        that progressively reduce spatial resolution while increasing channel depth. 
        SE (channel attention) blocks are added in the deeper stages. 
        The network ends with global average pooling followed by 
        a linear layer that outputs the 37 classes.
    """

    def __init__(self, num_classes: int = 37, dropout: float = 0.3) -> None:
        """Assemble the custom CNN stages and classifier head."""

        super().__init__()

        # stride-1 stem keeps the full input resolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # downsample lives inside the first ResBlock of each stage
        self.stage1 = nn.Sequential(
            BasicResBlock(32, 64, stride=2),
            BasicResBlock(64, 64, stride=1),
        )

        # SE only in the deeper stages - it didn't help the shallow ones
        self.stage2 = nn.Sequential(
            BasicResBlock(64, 128, stride=2),
            BasicResBlock(128, 128, stride=1),
            SqueezeExcite(128, reduction=16),
        )

        self.stage3 = nn.Sequential(
            # Final stage feeds both the classifier and the Grad-CAM target.
            BasicResBlock(128, 256, stride=2),
            BasicResBlock(256, 256, stride=1),
            SqueezeExcite(256, reduction=16),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout only on the FC head. Early dropout in the conv stack
        # dropped val accuracy by a couple of points.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise layers with the settings used for all custom runs."""

        # Kaiming for conv/linear, BN at gamma=1, beta=0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from image tensor to class logits."""

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        return self.classifier(x)

    def gradcam_target_layer(self) -> nn.Module:
        """Return the layer used for Grad-CAM on the custom model."""

        # last conv stage - that's where Grad-CAM normally hooks in
        return self.stage3[-1]


# ---------------------------------------------------------------------------
# Transfer learning wrappers (transfer-learning track)
# ---------------------------------------------------------------------------


_WEIGHTS_TABLE = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet34": models.ResNet34_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
    "vgg16": models.VGG16_Weights.DEFAULT,
    "vgg19": models.VGG19_Weights.DEFAULT,
    "mobilenet_v2": models.MobileNet_V2_Weights.DEFAULT,
    "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.DEFAULT,
    "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
}


def _weights_for(model_name: str, pretrained: bool):
    """Return torchvision default weights when pretrained is requested."""

    return _WEIGHTS_TABLE[model_name] if pretrained else None


def _freeze_all_parameters(model: nn.Module) -> None:
    """Freeze a pretrained backbone for feature extraction."""

    # feature-extraction style transfer: backbone frozen, only the new head trains
    for parameter in model.parameters():
        parameter.requires_grad = False


def _replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    """Swap the final classifier for a 37-way head."""

    # ResNet / VGG / MobileNet store the head differently, so dispatch on name
    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported transfer model: {model_name}")


def _classifier_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Find the new head for the transfer-learning LR split."""

    # used by build_param_groups to find the head for the LR multiplier
    if model_name.startswith("resnet"):
        return model.fc
    if model_name.startswith("vgg") or model_name.startswith("mobilenet"):
        return model.classifier[-1]
    raise ValueError(f"Unsupported transfer model: {model_name}")


def build_model(
    model_name: str,
    num_classes: int = 37,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
) -> ModelInfo:
    """Build either the custom CNN or a torchvision transfer model."""

    model_name = model_name.lower()

    if model_name == "custom":
        # The custom CNN is always trained from scratch for custom-model track.
        model = PetResNet(num_classes=num_classes, dropout=dropout)
        return ModelInfo(
            model=model,
            model_name=model_name,
            pretrained=False,
            freeze_backbone=False,
            num_classes=num_classes,
        )

    if model_name not in TRANSFER_MODELS:
        raise ValueError(f"Unknown model {model_name}. Valid choices: {', '.join(MODEL_CHOICES)}")

    weights = _weights_for(model_name, pretrained)
    # torchvision exposes each architecture as a factory function with this name.
    factory = getattr(models, model_name)
    model = factory(weights=weights)

    # freeze before replacing the classifier - new head still has requires_grad=True
    if freeze_backbone:
        _freeze_all_parameters(model)

    model = _replace_classifier(model, model_name=model_name, num_classes=num_classes)

    return ModelInfo(
        model=model,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        num_classes=num_classes,
    )


# ----- differential LR for transfer learning -----


def build_param_groups(
    model: nn.Module,
    model_name: str,
    base_lr: float,
    head_lr_multiplier: float = 10.0,
) -> list[dict]:
    """Create optimiser parameter groups, with a bigger LR for the head."""

    # Two groups for transfer: backbone at base_lr, head at 10x.
    # The pretrained backbone shouldn't move much; the new head needs
    # to catch up.

    if model_name == "custom" or model_name not in TRANSFER_MODELS:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr}]

    head = _classifier_layer(model, model_name)
    # Parameter object ids are the safest way to separate head vs backbone.
    head_param_ids = {id(p) for p in head.parameters()}

    head_params = [p for p in head.parameters() if p.requires_grad]
    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in head_param_ids
    ]

    # If only the head trains the multiplier doesn't make sense
    if not backbone_params:
        if not head_params:
            raise RuntimeError("No trainable parameters, check freeze_backbone and classifier replacement.")
        return [{"params": head_params, "lr": base_lr}]

    groups: list[dict] = [{"params": backbone_params, "lr": base_lr}]
    if head_params:
        groups.append({"params": head_params, "lr": base_lr * head_lr_multiplier})
    return groups


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters for logging."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def iter_trainable(model: nn.Module) -> Iterable[nn.Parameter]:
    """Yield only parameters that the optimiser should update."""

    return (p for p in model.parameters() if p.requires_grad)
