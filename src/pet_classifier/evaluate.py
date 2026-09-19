"""Re-score a saved checkpoint on val + test.

Training already prints val numbers, so this script is mainly for fresh
confusion matrices on the test set or for moving a checkpoint to another
machine. It pulls the original args out of the checkpoint so the val
split is identical to the training run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from pet_classifier.data import build_dataloaders
from pet_classifier.models import build_model
from pet_classifier.train import run_one_epoch, run_tta_evaluation
from pet_classifier.utils import (
    get_device,
    save_confusion_outputs,
    save_json,
    save_per_class_accuracy_bar,
)


def parse_args() -> argparse.Namespace:
    """Parse options for re-scoring an existing checkpoint."""

    parser = argparse.ArgumentParser(description="Evaluate an Oxford-IIIT Pet checkpoint on val and test splits.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt or last_model.pt.")
    parser.add_argument("--data-dir", default=None,
                        help="Override the data dir baked into the checkpoint.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to put fresh evaluation artifacts.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from training time.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true",
                        help="Download the dataset if it is missing.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Only re-score the validation split.")
    parser.add_argument("--tta", action="store_true",
                        help="Also evaluate the test split with horizontal-flip TTA "
                             "(averages logits over original + flipped). Recorded as "
                             "test_acc_tta alongside the plain test_acc.")
    return parser.parse_args()


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove torch.compile prefixes from saved state_dict keys."""

    # torch.compile adds _orig_mod. to every key. Strip it before loading
    # into a normal (non-compiled) model.
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}


def main() -> None:
    """Load a checkpoint, rebuild the data split and save fresh metrics."""

    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    # weights_only=False so the stored args/indices/class_names are loaded too
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = SimpleNamespace(**checkpoint["args"])

    # CLI overrides are only for moving a checkpoint to another machine.
    data_dir = args.data_dir or saved_args.data_dir
    batch_size = args.batch_size or saved_args.batch_size
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild loaders using the saved args so the val split matches.
    # augmentation="none" so the numbers are deterministic.
    data = build_dataloaders(
        data_dir=data_dir,
        image_size=saved_args.image_size,
        batch_size=batch_size,
        val_fraction=saved_args.val_fraction,
        num_workers=args.num_workers,
        augmentation="none",
        seed=saved_args.seed,
        download=args.download,
    )

    model_info = build_model(
        # Build the same architecture first, then load trained weights.
        model_name=saved_args.model,
        num_classes=saved_args.num_classes,
        pretrained=False,         # the weights come from the checkpoint
        freeze_backbone=False,    # not relevant for inference
        dropout=getattr(saved_args, "dropout", 0.3),
    )
    model = model_info.model
    model.load_state_dict(strip_compile_prefix(checkpoint["model_state"]))

    device = get_device(args.device)
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss()

    # --- Validation ---
    with torch.no_grad():
        val_loss, val_acc, val_targets, val_predictions = run_one_epoch(
            model=model, loader=data.val_loader,
            criterion=criterion, optimizer=None,
            device=device, scaler=None, use_amp=False,
            grad_clip=0.0, epoch=int(checkpoint.get("epoch", 0)),
            phase="val(reeval)",
        )

    save_confusion_outputs(output_dir, val_targets, val_predictions,
                           checkpoint["class_names"], prefix="validation")
    save_per_class_accuracy_bar(output_dir, val_targets, val_predictions,
                                checkpoint["class_names"], prefix="validation")

    results: dict[str, object] = {
        # Save the core numbers in a small file for later comparison.
        "checkpoint": str(checkpoint_path),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_samples": len(val_targets),
    }

    # --- Test ---
    test_acc_tta: float | None = None
    if not args.skip_test:
        with torch.no_grad():
            test_loss, test_acc, test_targets, test_predictions = run_one_epoch(
                model=model, loader=data.test_loader,
                criterion=criterion, optimizer=None,
                device=device, scaler=None, use_amp=False,
                grad_clip=0.0, epoch=0, phase="test",
            )
        save_confusion_outputs(output_dir, test_targets, test_predictions,
                               checkpoint["class_names"], prefix="test")
        save_per_class_accuracy_bar(output_dir, test_targets, test_predictions,
                                    checkpoint["class_names"], prefix="test")
        results.update({
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_samples": len(test_targets),
        })

        if args.tta:
            # Plain and TTA metrics are both kept so the delta is easy to compare.
            tta_loss, test_acc_tta, tta_targets, tta_predictions = run_tta_evaluation(
                model=model, loader=data.test_loader,
                criterion=criterion,
                device=device, use_amp=False,
            )
            save_confusion_outputs(output_dir, tta_targets, tta_predictions,
                                   checkpoint["class_names"], prefix="test_tta")
            save_per_class_accuracy_bar(output_dir, tta_targets, tta_predictions,
                                        checkpoint["class_names"], prefix="test_tta")
            results.update({
                "test_loss_tta": float(tta_loss),
                "test_acc_tta": float(test_acc_tta),
            })

    save_json(output_dir / "metrics.json", results)

    print(f"Validation loss / acc: {val_loss:.4f} / {val_acc:.4f}")
    if not args.skip_test:
        print(f"Test loss / acc:       {test_loss:.4f} / {test_acc:.4f}")
    if test_acc_tta is not None:
        print(f"Test (TTA) loss / acc: {tta_loss:.4f} / {test_acc_tta:.4f}")
    print(f"Evaluation artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
