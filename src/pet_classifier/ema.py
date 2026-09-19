"""Exponential moving average support for model weights."""

from __future__ import annotations

import torch
from torch import nn


class ModelEma:
    """Maintain bias-corrected exponential moving averages of model state."""

    BIAS_WARMUP = 10

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the shadow state after one optimizer step."""
        self.num_updates += 1
        effective_decay = min(
            self.decay,
            (1.0 + self.num_updates) / (self.BIAS_WARMUP + self.num_updates),
        )
        for name, tensor in model.state_dict().items():
            shadow_tensor = self.shadow[name]
            if tensor.dtype.is_floating_point:
                shadow_tensor.mul_(effective_decay).add_(
                    tensor.detach(), alpha=1.0 - effective_decay
                )
            else:
                shadow_tensor.copy_(tensor)

    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Apply EMA state to a model and return its original state."""
        backup = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        model.load_state_dict(self.shadow)
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore state returned by :meth:`apply_to`."""
        model.load_state_dict(backup)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the averaged model state for checkpointing."""
        return self.shadow
