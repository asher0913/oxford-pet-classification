"""Grad-CAM implementation, written following the Selvaraju paper.

The idea: hook the last conv layer to grab its activations and gradients,
take the spatial mean of gradients per channel as weights, multiply by
activations and ReLU. Then upsample to image size and overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from pet_classifier.data import denormalize_for_display


class GradCAM:
    """Tiny Grad-CAM helper that owns the hooks for one target layer."""

    # Usage:
    #   cam = GradCAM(model, model.layer4[-1])
    #   heatmap, pred, prob = cam(image_tensor)
    # Call remove_hooks() afterwards to free the handles.

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Register hooks on the layer used for the explanation."""

        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # forward hook -> save activations, backward hook -> save gradients
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module: nn.Module, _inputs, output: torch.Tensor) -> None:
        """Store activations from the forward hook."""

        # detach so autograd doesn't hold on to anything
        self.activations = output.detach()

    def _save_gradients(self, _module: nn.Module, _grad_in, grad_out) -> None:
        """Store gradients from the backward hook."""

        # grad_out[0] is dL/d(layer_output)
        self.gradients = grad_out[0].detach()

    def remove_hooks(self) -> None:
        """Remove hooks so repeated visualisations do not leak handles."""

        self._forward_handle.remove()
        self._backward_handle.remove()

    def __call__(self, image: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, int, float]:
        """Return the heatmap, predicted class index and probability."""

        # image must be [1, C, H, W] and already on the right device
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError("GradCAM expects a single image with shape [1, C, H, W].")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(image)
        probabilities = F.softmax(logits, dim=1)

        if class_index is None:
            # Default explanation is for the model's own top prediction.
            class_index = int(torch.argmax(logits, dim=1).item())

        # Scalar to back-propagate through: the chosen class logit.
        score = logits[0, class_index]
        score.backward()

        if self.activations is None or self.gradients is None:
            # Usually means the chosen layer was not used by this model path.
            raise RuntimeError("Activations or gradients were not captured. "
                               "Did the target layer run during forward?")

        # alpha_k = spatial mean of gradient on channel k
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # weighted sum across channels then ReLU
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # bring the heatmap up to image resolution
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)

        # normalise to [0,1] so it plots with sensible colours
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        heatmap = cam.squeeze().cpu().numpy()
        probability = float(probabilities[0, class_index].item())
        return heatmap, class_index, probability


def overlay_heatmap(image_tensor: torch.Tensor, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a Grad-CAM heatmap with the original image."""

    # de-normalise so the image colours are right, then blend with the jet heatmap
    image = denormalize_for_display(image_tensor).permute(1, 2, 0).cpu().numpy()

    colour_map = plt.get_cmap("jet")
    # matplotlib returns RGBA; imshow only needs RGB here.
    coloured_heatmap = colour_map(heatmap)[..., :3]  # drop alpha channel

    blended = (1.0 - alpha) * image + alpha * coloured_heatmap
    return np.clip(blended, 0.0, 1.0)


def save_gradcam_grid(
    output_path: str | Path,
    samples: Iterable[tuple[torch.Tensor, int, str]],
    model: nn.Module,
    target_layer: nn.Module,
    class_names: list[str],
    device: torch.device,
    ncols: int = 4,
) -> None:
    """Save raw images and Grad-CAM overlays in one figure."""

    # samples is a list of (image, true_label, caption). For each one,
    # the raw image and the Grad-CAM overlay are plotted side by side.
    samples = list(samples)
    if not samples:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cam = GradCAM(model, target_layer)

    # 2 subplots per sample (image + overlay)
    rows = (len(samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(rows, ncols * 2, figsize=(ncols * 4.5, rows * 3.0))
    axes = np.atleast_2d(axes)

    for idx, (image_tensor, true_label, caption) in enumerate(samples):
        # Each sample takes two neighbouring axes: raw image then overlay.
        row = idx // ncols
        col = (idx % ncols) * 2

        image_on_device = image_tensor.unsqueeze(0).to(device)
        heatmap, pred_idx, prob = cam(image_on_device)
        overlay = overlay_heatmap(image_tensor, heatmap, alpha=0.4)

        axes[row, col].imshow(denormalize_for_display(image_tensor).permute(1, 2, 0).cpu().numpy())
        axes[row, col].axis("off")
        axes[row, col].set_title(
            f"true: {class_names[true_label]}",
            fontsize=9,
        )

        axes[row, col + 1].imshow(overlay)
        axes[row, col + 1].axis("off")
        status = "OK" if pred_idx == true_label else "WRONG"
        axes[row, col + 1].set_title(
            f"{status}: {class_names[pred_idx]} ({prob:.2f})",
            fontsize=9,
            color=("green" if pred_idx == true_label else "red"),
        )

    # blank out unused trailing cells
    total_slots = axes.shape[0] * axes.shape[1]
    used_slots = len(samples) * 2
    for leftover in range(used_slots, total_slots):
        r = leftover // axes.shape[1]
        c = leftover % axes.shape[1]
        axes[r, c].axis("off")

    if caption:
        fig.suptitle(caption, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    cam.remove_hooks()
