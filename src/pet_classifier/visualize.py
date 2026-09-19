"""Run a saved checkpoint and produce two figures: a prediction grid
(true vs predicted, green/red) and a Grad-CAM grid for the same images.

Pass --filter correct or --filter incorrect to focus the grid on either
the model's wins or its mistakes - the latter is used in the discussion
section.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pet_classifier.data import build_dataloaders, denormalize_for_display
from pet_classifier.evaluate import strip_compile_prefix
from pet_classifier.gradcam import save_gradcam_grid
from pet_classifier.models import PetResNet, build_model
from pet_classifier.utils import get_device


def parse_args() -> argparse.Namespace:
    """Parse options for checkpoint visualisation."""

    parser = argparse.ArgumentParser(description="Render prediction grids and Grad-CAM overlays for a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=16,
                        help="How many images to show in each grid.")
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--filter",
                        choices=["any", "correct", "incorrect"], default="any",
                        help="Restrict the samples to correct or incorrect predictions.")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def pick_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Choose the convolutional layer Grad-CAM should hook."""

    # Rule of thumb: the last conv stage before the classifier
    if isinstance(model, PetResNet):
        return model.gradcam_target_layer()

    if model_name.startswith("resnet"):
        return model.layer4[-1]
    if model_name.startswith("vgg"):
        # VGG.features ends with a pool, so walk backwards for the last conv
        for layer in reversed(model.features):
            if isinstance(layer, torch.nn.Conv2d):
                return layer
        raise RuntimeError("No conv layer found in VGG.features")
    if model_name.startswith("mobilenet"):
        return model.features[-1]

    raise ValueError(f"No Grad-CAM target rule for model {model_name}")


def collect_filtered_samples(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    filter_mode: str,
    num_samples: int,
):
    """Collect examples matching the requested correctness filter."""

    # Walk the loader and stop once enough samples have been collected that
    # match filter_mode (correct / incorrect / any)
    collected: list[tuple[torch.Tensor, int, int]] = []
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            # Only move the batch copy used for inference; keep originals for plotting.
            images_gpu = images.to(device, non_blocking=True)
            logits = model(images_gpu)
            predictions = torch.argmax(logits, dim=1).cpu()

            for image, label, pred in zip(images, labels.tolist(), predictions.tolist()):
                # Filtering here is cheaper than collecting everything first.
                if filter_mode == "correct" and pred != label:
                    continue
                if filter_mode == "incorrect" and pred == label:
                    continue
                collected.append((image, int(label), int(pred)))
                if len(collected) >= num_samples:
                    return collected
    return collected


def save_prediction_grid(
    output_path: str | Path,
    samples: list[tuple[torch.Tensor, int, int]],
    class_names: list[str],
    ncols: int = 4,
) -> None:
    """Save a grid of images with true and predicted labels."""

    # Each cell shows the image with true/predicted label. Green = correct, red = wrong.
    if not samples:
        print("No samples collected, skipping prediction grid.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = (len(samples) + ncols - 1) // ncols
    # np.atleast_2d below handles the one-row case cleanly.
    fig, axes = plt.subplots(rows, ncols, figsize=(ncols * 3.2, rows * 3.2))
    axes = np.atleast_2d(axes)

    for idx, (image, true_label, pred_label) in enumerate(samples):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].imshow(denormalize_for_display(image).permute(1, 2, 0).cpu().numpy())
        axes[row, col].axis("off")

        correct = true_label == pred_label
        title = f"T: {class_names[true_label]}\nP: {class_names[pred_label]}"
        axes[row, col].set_title(
            title, fontsize=8,
            color=("green" if correct else "red"),
        )

    # Blank out unused grid cells.
    total = rows * ncols
    for leftover in range(len(samples), total):
        axes[leftover // ncols, leftover % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Load a checkpoint and write the two visualisation grids."""

    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = SimpleNamespace(**checkpoint["args"])

    data_dir = args.data_dir or saved_args.data_dir
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "visualisation"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = build_dataloaders(
        data_dir=data_dir,
        image_size=saved_args.image_size,
        batch_size=saved_args.batch_size,
        val_fraction=saved_args.val_fraction,
        num_workers=2,
        augmentation="none",
        seed=saved_args.seed,
        download=args.download,
    )

    model_info = build_model(
        model_name=saved_args.model,
        num_classes=saved_args.num_classes,
        pretrained=False,
        freeze_backbone=False,
        dropout=getattr(saved_args, "dropout", 0.3),
    )
    model = model_info.model
    model.load_state_dict(strip_compile_prefix(checkpoint["model_state"]))

    device = get_device(args.device)
    model.to(device).eval()

    loader = data.val_loader if args.split == "val" else data.test_loader

    # Reuse the exact same samples for prediction grid and Grad-CAM.
    samples = collect_filtered_samples(
        loader=loader, model=model, device=device,
        filter_mode=args.filter, num_samples=args.num_samples,
    )

    # Prediction grid
    save_prediction_grid(
        output_path=output_dir / f"predictions_grid_{args.split}_{args.filter}.png",
        samples=samples,
        class_names=checkpoint["class_names"],
        ncols=args.ncols,
    )

    # Grad-CAM grid on the same samples so the two figures correspond.
    gradcam_samples = [
        (image, true_label, f"{args.split} / {args.filter}")
        for image, true_label, _ in samples
    ]
    target_layer = pick_target_layer(model, saved_args.model)
    save_gradcam_grid(
        output_path=output_dir / f"gradcam_grid_{args.split}_{args.filter}.png",
        samples=gradcam_samples,
        model=model,
        target_layer=target_layer,
        class_names=checkpoint["class_names"],
        device=device,
        ncols=args.ncols,
    )

    print(f"Saved prediction and Grad-CAM grids to {output_dir}")


if __name__ == "__main__":
    main()
