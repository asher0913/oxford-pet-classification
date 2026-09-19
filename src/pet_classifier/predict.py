"""Run inference on a single image or a folder of images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from pet_classifier.data import build_image_transform
from pet_classifier.evaluate import strip_compile_prefix
from pet_classifier.models import build_model
from pet_classifier.utils import get_device


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse options for single-image or folder prediction."""

    parser = argparse.ArgumentParser(description="Predict pet breeds for one image or a whole folder.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory of images.")
    parser.add_argument("--output-csv", default="predictions.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    """Collect supported image files from a path."""

    # single file or directory - either way return a sorted list of image paths
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            file for file in path.rglob("*")
            if file.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Input does not exist: {path}")


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    transform,
    device: torch.device,
    top_k: int,
    class_names: list[str],
) -> dict[str, str]:
    """Run one image through the model and format the top-k outputs."""

    # convert("RGB") handles grayscale PNGs and RGBA images
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        scores, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))

    row: dict[str, str] = {"image": str(image_path)}
    for rank, (score, index) in enumerate(zip(scores.cpu().tolist(), indices.cpu().tolist()), start=1):
        row[f"top{rank}_label"] = class_names[index]
        row[f"top{rank}_probability"] = f"{score:.6f}"
    return row


def main() -> None:
    """Load a checkpoint, predict all requested images and write a CSV."""

    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = SimpleNamespace(**checkpoint["args"])
    # Use the class order saved with the checkpoint, not a fresh guess.
    class_names = checkpoint["class_names"]

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

    transform = build_image_transform(saved_args.image_size)
    images = collect_images(Path(args.input))
    if not images:
        raise FileNotFoundError(f"No images found at {args.input}")

    # One row per image keeps the CSV easy to inspect in Excel.
    rows = [predict_image(model, image, transform, device, args.top_k, class_names)
            for image in images]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {output_csv}")
    print("Sample row:", rows[0])


if __name__ == "__main__":
    main()
