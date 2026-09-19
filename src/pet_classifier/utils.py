"""Shared helpers: metrics, plotting, checkpoint packaging, seeding."""

from __future__ import annotations

import csv
import json
import random
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

import matplotlib

# Agg backend so matplotlib works on the headless GPU server
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402


class AverageMeter:
    """Track a weighted average across batches."""

    # Weighted running mean - weighted by batch size because the last
    # batch is usually smaller and skews a plain mean.

    def __init__(self) -> None:
        """Start with an empty meter."""

        self.reset()

    def reset(self) -> None:
        """Clear the accumulated total and count."""

        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        """Add one batch value, weighted by batch size."""

        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        """Return the current average, or 0 before any updates."""

        return self.total / self.count if self.count else 0.0


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for repeatable experiment runs."""

    # Seed every RNG. cudnn.benchmark=True is faster but non-deterministic;
    # Determinism is only forced via --deterministic for the final report runs.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(requested: str) -> torch.device:
    """Pick the requested device, with auto preferring CUDA."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    """Count correct predictions in one batch."""

    # returns (correct, total) for one batch
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == targets).sum().item()
    return int(correct), int(targets.numel())


def namespace_to_dict(args: Namespace | dict[str, Any]) -> dict[str, Any]:
    """Convert argparse settings into JSON-friendly values."""

    # Path objects aren't JSON-serialisable, so convert them to strings here
    if isinstance(args, Namespace):
        raw = vars(args)
    else:
        raw = args
    output: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a small JSON artifact, creating folders if needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_history_csv(path: str | Path, history: list[dict[str, float]]) -> None:
    """Write the per-epoch history table."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(path: str | Path, history: list[dict[str, float]]) -> None:
    """Save the loss/accuracy curves used in the report."""

    # two-panel plot: loss curves on the left, accuracy curves on the right
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss and accuracy share epochs but need different y-axes.
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="Train accuracy")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="Val accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training / Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_confusion_outputs(
    output_dir: str | Path,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str],
    prefix: str = "validation",
) -> None:
    """Save confusion matrix artifacts for one split."""

    # Save CM as CSV + a heatmap PNG, plus sklearn's classification_report
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))
    # Passing labels keeps empty classes in the matrix instead of shrinking it.
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(output_dir / f"{prefix}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    # scale figure size by class count so 37 labels stay readable
    fig_width = max(12, len(class_names) * 0.45)
    fig_height = max(10, len(class_names) * 0.38)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        cm, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, square=False, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{prefix.capitalize()} Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=220)
    plt.close(fig)

    report = classification_report(
        y_true, y_pred, labels=labels,
        target_names=class_names, zero_division=0,
    )
    (output_dir / f"{prefix}_classification_report.txt").write_text(report, encoding="utf-8")


def save_per_class_accuracy_bar(
    output_dir: str | Path,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str],
    prefix: str = "validation",
) -> None:
    """Save a worst-first per-class accuracy bar chart."""

    # Horizontal bar chart of per-class accuracy. Sorted worst-first so
    # the problem breeds are easy to spot quickly.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    accuracies = []
    for class_index in range(len(class_names)):
        # Some filtered plots may have no examples for a class.
        mask = y_true_arr == class_index
        if mask.sum() == 0:
            accuracies.append(0.0)
        else:
            accuracies.append(float((y_pred_arr[mask] == class_index).mean()))

    order = np.argsort(accuracies)
    sorted_names = [class_names[i] for i in order]
    sorted_values = [accuracies[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(6, len(class_names) * 0.25)))
    ax.barh(range(len(class_names)), sorted_values, color="steelblue")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_title(f"Per-Class Accuracy ({prefix})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_per_class_accuracy.png", dpi=200)
    plt.close(fig)


def checkpoint_payload(
    model_state: dict[str, torch.Tensor],
    args: Namespace,
    class_names: list[str],
    train_indices: list[int],
    val_indices: list[int],
    epoch: int,
    best_val_acc: float,
) -> dict[str, Any]:
    """Package the checkpoint fields needed for later evaluation."""

    # Save the args + split indices with the weights so evaluate.py can
    # rebuild the same val split later without guessing.
    return {
        # These fields are enough to rebuild the model and exact val split.
        "model_state": model_state,
        "args": namespace_to_dict(args),
        "class_names": class_names,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
    }
