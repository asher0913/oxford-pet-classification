"""Training script for the from-scratch custom CNN and the transfer-learning models.

Same loop is used for the custom CNN and the transfer learning models,
the difference is just how the model gets built. All settings come from
CLI args so each ablation row can be driven from one script.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from tqdm import tqdm

from pet_classifier.data import build_dataloaders
from pet_classifier.ema import ModelEma
from pet_classifier.models import MODEL_CHOICES, build_model, build_param_groups, count_parameters
from pet_classifier.utils import (
    AverageMeter,
    accuracy_from_logits,
    checkpoint_payload,
    get_device,
    plot_training_curves,
    save_confusion_outputs,
    save_json,
    save_per_class_accuracy_bar,
    seed_everything,
    write_history_csv,
)

def parse_args() -> argparse.Namespace:
    """Collect all CLI settings for one training run."""

    parser = argparse.ArgumentParser(description="Train models for the pet-classification project.")

    # --- Experiment book-keeping ---
    parser.add_argument("--experiment-name", default="pet_experiment",
                        help="Used to name the output directory.")
    parser.add_argument("--data-dir", default="./data", help="Dataset root.")
    parser.add_argument("--output-dir", default="./outputs",
                        help="Where run artifacts land.")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Write to output-dir/experiment-name without appending a timestamp.")

    # --- Model ---
    parser.add_argument("--model", default="custom", choices=MODEL_CHOICES,
                        help="Model architecture to train.")
    # default False so a from-scratch run can never accidentally end up pretrained
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Use ImageNet weights for transfer models. Ignored for custom.")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze the pretrained feature extractor (feature-extraction strategy).")
    parser.add_argument("--num-classes", type=int, default=37,
                        help="Oxford-IIIT Pet has 37 breeds.")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout on the final FC layer (custom CNN only).")

    # --- Data ---
    parser.add_argument("--image-size", type=int, default=160,
                        help="160 for custom, 224 for transfer models.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--augmentation", choices=["none", "basic", "strong"], default="basic")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--download", action="store_true",
                        help="Download the dataset if it is missing.")

    # --- Optimiser and schedule ---
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Only used by SGD.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler",
                        choices=["none", "step", "cosine", "plateau"],
                        default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup for the first N epochs. Only combines with cosine; "
                             "ignored for other schedulers. 0 disables warmup.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Cross-entropy label smoothing factor. 0.1 is a reasonable starting point.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Beta distribution parameter for Mixup. 0 disables Mixup; 0.2 is typical.")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Max grad norm. 0 disables clipping.")
    parser.add_argument("--head-lr-mult", type=float, default=10.0,
                        help="Multiplier on the new classifier head's LR for transfer fine-tuning.")
    parser.add_argument("--min-lr", type=float, default=0.0,
                        help="Cosine LR floor (eta_min). Set to 1e-5 to avoid the LR hitting 0 at the end.")
    parser.add_argument("--ema", action="store_true",
                        help="Use EMA of weights for val + checkpointing.")
    parser.add_argument("--ema-decay", type=float, default=0.9999,
                        help="EMA decay. The actual decay is bias-corrected (see ModelEma).")
    parser.add_argument("--tta", action="store_true",
                        help="Test-time augmentation (horizontal flip avg) at --test-at-end.")

    # --- System ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic cuDNN (slower).")
    parser.add_argument("--device", default="auto",
                        help="auto | cuda | cuda:0 | cpu.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training on CUDA.")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap the model in torch.compile (needs PyTorch 2.x).")
    parser.add_argument("--test-at-end", action="store_true",
                        help="Run the best model on the official test split once training finishes.")
    return parser.parse_args()


def create_run_dir(output_dir: str | Path, experiment_name: str, no_timestamp: bool) -> Path:
    """Create the output folder for this experiment."""

    # Timestamp on the folder name so re-runs don't overwrite each other
    base = Path(output_dir)
    if no_timestamp:
        run_dir = base / experiment_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / f"{experiment_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> Optimizer:
    """Build the requested optimiser for the trainable params."""

    # For transfer models the param groups split backbone vs head so they
    # can use different LRs (head needs to catch up faster).
    groups = build_param_groups(
        model=model,
        model_name=args.model,
        base_lr=args.lr,
        head_lr_multiplier=args.head_lr_mult,
    )

    if args.optimizer == "adam":
        return torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    # SGD
    return torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


def build_scheduler(args: argparse.Namespace, optimizer: Optimizer):
    """Build the learning-rate schedule selected on the command line."""

    # Cosine + warmup is the main schedule used here. Warmup matters most for
    # the from-scratch CNN since the BN stats are noisy at the start.
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    if args.scheduler == "cosine":
        warmup = max(int(args.warmup_epochs), 0)
        cosine_epochs = max(args.epochs - warmup, 1)
        # min_lr keeps the tail just above zero so the last few epochs
        # still do something. With eta_min=0 the last epoch had LR ~ 0
        # and the model basically froze.
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=float(args.min_lr))
        if warmup == 0:
            return cosine
        # ramp LR from 0.1*base to base over the warmup window
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        return SequentialLR(optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup])
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup to one batch and return both target labels."""

    # Mixup builds a fake training image by linearly mixing two real ones.
    # Return both labels and lam, then mix the losses in the training
    # loop because CrossEntropyLoss only takes integer targets.
    if alpha <= 0.0:
        return images, targets, targets, 1.0

    # Beta(alpha, alpha) gives the random mixing strength for this batch.
    lam = float(np.random.beta(alpha, alpha))
    # Pair each image with another image from the same batch.
    permutation = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[permutation]
    return mixed, targets, targets[permutation], lam


def unwrap_compiled(model: nn.Module) -> nn.Module:
    """Get the original module if torch.compile wrapped it."""

    # torch.compile adds an _orig_mod wrapper; saving the wrapped state_dict
    # leaves _orig_mod. prefixes in the keys and breaks loading later
    return getattr(model, "_orig_mod", model)


def run_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    grad_clip: float,
    epoch: int,
    phase: str,
    mixup_alpha: float = 0.0,
    ema: ModelEma | None = None,
) -> tuple[float, float, list[int], list[int]]:
    """Run one train/val/test pass and return loss, acc and labels."""

    # Pass optimizer=None to use this for validation/test as well.
    # Mixup only kicks in during training so val/test numbers stay
    # comparable across rows.
    is_train = optimizer is not None
    use_mixup = is_train and mixup_alpha > 0.0
    model.train(mode=is_train)

    loss_meter = AverageMeter()
    correct_total = 0
    sample_total = 0
    # Stored for confusion matrices and per-class plots after the pass.
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(loader, desc=f"{phase} epoch {epoch}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # Mix before autocast so interpolation stays in fp32
        if use_mixup:
            mixed_images, targets_a, targets_b, lam = mixup_batch(images, targets, mixup_alpha)
        else:
            mixed_images, targets_a, targets_b, lam = images, targets, targets, 1.0

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Evaluation uses the same path, just with gradients disabled.
                logits = model(mixed_images)
                if use_mixup:
                    # Mix the losses, not the labels (CE expects ints)
                    loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
                else:
                    loss = criterion(logits, targets_a)

            if is_train:
                if scaler is not None and use_amp:
                    # AMP scales the loss so fp16 grads don't underflow
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in model.parameters() if p.requires_grad),
                            max_norm=grad_clip,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in model.parameters() if p.requires_grad),
                            max_norm=grad_clip,
                        )
                    optimizer.step()

                if ema is not None:
                    # EMA follows the training weights after each real update.
                    ema.update(unwrap_compiled(model))

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)

        correct, total = accuracy_from_logits(logits.detach(), targets)
        correct_total += correct
        sample_total += total

        predictions = torch.argmax(logits.detach(), dim=1)
        # Keep CPU lists so later plotting code does not need tensors/devices.
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

        progress.set_postfix(
            loss=f"{loss_meter.average:.4f}",
            acc=f"{correct_total / max(sample_total, 1):.4f}",
        )

    accuracy = correct_total / max(sample_total, 1)
    return loss_meter.average, accuracy, all_targets, all_predictions


def run_tta_evaluation(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate by averaging original and horizontally flipped logits."""

    # Test-time augmentation: predict on the image and its horizontal
    # flip, average the two logits. Pets are roughly left-right symmetric
    # so this is a safe averaging trick.
    model.eval()
    loss_meter = AverageMeter()
    correct_total = 0
    sample_total = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="test (TTA)", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits_main = model(images)
                logits_flip = model(torch.flip(images, dims=[3]))
                logits = (logits_main + logits_flip) * 0.5
                loss = criterion(logits, targets)

            loss_meter.update(loss.item(), targets.size(0))
            predictions = torch.argmax(logits, dim=1)
            correct_total += (predictions == targets).sum().item()
            sample_total += targets.size(0)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    accuracy = correct_total / max(sample_total, 1)
    return loss_meter.average, accuracy, all_targets, all_predictions


def main() -> None:
    """Full training run: data, model, epochs, checkpoints and summary."""

    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    device = get_device(args.device)

    # AMP only matters on CUDA
    use_amp = bool(args.amp and device.type == "cuda")
    run_dir = create_run_dir(args.output_dir, args.experiment_name, args.no_timestamp)

    if args.model != "custom" and args.image_size != 224:
        print("Note: transfer models usually want 224x224 input. "
              f"Running with {args.image_size} instead.")

    data = build_dataloaders(
        # This call fixes the train/val split for the whole run.
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
        seed=args.seed,
        download=args.download,
    )

    model_info = build_model(
        # The flags are ignored for the custom CNN so it stays from scratch.
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=bool(args.pretrained and args.model != "custom"),
        freeze_backbone=bool(args.freeze_backbone and args.model != "custom"),
        dropout=args.dropout,
    )
    model = model_info.model.to(device)

    if args.compile and hasattr(torch, "compile"):
        # Compile is optional because it can be slower on some CPU setups.
        model = torch.compile(model)

    total_params, trainable_params = count_parameters(model)
    print(f"Run directory:    {run_dir}")
    print(f"Device:           {device}")
    print(f"Classes:          {len(data.class_names)}")
    print(f"Train / val size: {len(data.train_indices)} / {len(data.val_indices)}")
    print(f"Test size:        {len(data.test_loader.dataset)}")
    print(f"Model:            {args.model}  pretrained={model_info.pretrained}  "
          f"freeze={model_info.freeze_backbone}")
    print(f"Parameters:       total={total_params:,}  trainable={trainable_params:,}")

    save_json(
        run_dir / "run_config.json",
        {
            "args": vars(args),
            "class_names": data.class_names,
            "train_size": len(data.train_indices),
            "val_size": len(data.val_indices),
            "test_size": len(data.test_loader.dataset),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        },
    )

    # Label smoothing is a very cheap regulariser that helps on
    # fine-grained problems. It pushes the model away from putting
    # all probability mass on one class, which reduces overfitting.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    # New torch.amp API. The old torch.cuda.amp.* is deprecated.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # EMA is built against the unwrapped model so its state-dict keys
    # match the keys later loaded into the model (torch.compile adds an
    # _orig_mod. prefix that would otherwise have to be stripped).
    ema: ModelEma | None = None
    if args.ema:
        ema = ModelEma(unwrap_compiled(model), decay=args.ema_decay)
        print(f"EMA enabled with decay={args.ema_decay}. "
              f"Validation and best_model.pt will use the EMA weights.")

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_model.pt"

    # Wall-clock timing so the report can quote training cost per model.
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, _, _ = run_one_epoch(
            model=model, loader=data.train_loader,
            criterion=criterion, optimizer=optimizer,
            device=device, scaler=scaler, use_amp=use_amp,
            grad_clip=args.grad_clip,
            epoch=epoch, phase="train",
            mixup_alpha=args.mixup_alpha,
            ema=ema,
        )

        # If EMA is on, validation runs twice (raw + EMA). EMA is what gets
        # checkpointed but raw is logged too, so history.csv shows whether
        # the shadow is lagging.
        val_loss_raw: float | None = None
        val_acc_raw: float | None = None
        if ema is not None:
            with torch.no_grad():
                val_loss_raw, val_acc_raw, _, _ = run_one_epoch(
                    model=model, loader=data.val_loader,
                    criterion=criterion, optimizer=None,
                    device=device, scaler=None, use_amp=use_amp,
                    grad_clip=0.0,
                    epoch=epoch, phase="val(raw)",
                )

        ema_backup: dict[str, torch.Tensor] | None = None
        if ema is not None:
            # Swap EMA weights in only for the validation pass.
            ema_backup = ema.apply_to(unwrap_compiled(model))
        try:
            with torch.no_grad():
                val_loss, val_acc, val_targets, val_predictions = run_one_epoch(
                    model=model, loader=data.val_loader,
                    criterion=criterion, optimizer=None,
                    device=device, scaler=None, use_amp=use_amp,
                    grad_clip=0.0,
                    epoch=epoch, phase="val",
                )
        finally:
            if ema is not None and ema_backup is not None:
                ema.restore(unwrap_compiled(model), ema_backup)
        epoch_seconds = time.time() - epoch_start

        # plateau needs val_loss; others step blindly
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(current_lr),
            "epoch_seconds": float(epoch_seconds),
        }
        # log raw val too when EMA is on so it can be compared in the report
        if ema is not None:
            row["val_loss_raw"] = float(val_loss_raw) if val_loss_raw is not None else 0.0
            row["val_acc_raw"] = float(val_acc_raw) if val_acc_raw is not None else 0.0
        history.append(row)
        write_history_csv(run_dir / "history.csv", history)
        plot_training_curves(run_dir / "training_curves.png", history)

        # EMA shadow goes into best_model.pt so evaluate.py just loads it
        # normally. Raw weights are also saved as best_model_raw.pt as a
        # safety net because EMA underperformed on this dataset in some runs.
        if ema is not None:
            weights_for_checkpoint = ema.state_dict()
        else:
            weights_for_checkpoint = unwrap_compiled(model).state_dict()

        payload = checkpoint_payload(
            model_state=weights_for_checkpoint,
            args=args,
            class_names=data.class_names,
            train_indices=data.train_indices,
            val_indices=data.val_indices,
            epoch=epoch,
            best_val_acc=max(best_val_acc, val_acc),
        )
        # last_model overwritten every epoch, best_model only on improvement
        torch.save(payload, last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(payload, best_path)
            if ema is not None:
                raw_payload = checkpoint_payload(
                    model_state=unwrap_compiled(model).state_dict(),
                    args=args,
                    class_names=data.class_names,
                    train_indices=data.train_indices,
                    val_indices=data.val_indices,
                    epoch=epoch,
                    best_val_acc=max(best_val_acc, val_acc),
                )
                torch.save(raw_payload, run_dir / "best_model_raw.pt")
            save_confusion_outputs(run_dir, val_targets, val_predictions,
                                   data.class_names, prefix="validation")
            save_per_class_accuracy_bar(run_dir, val_targets, val_predictions,
                                        data.class_names, prefix="validation")
            print(f"Epoch {epoch}: new best val acc = {val_acc:.4f}")

        if ema is not None and val_acc_raw is not None:
            print(f"Epoch {epoch:03d}/{args.epochs:03d}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} (ema)  "
                  f"val_acc_raw={val_acc_raw:.4f}  "
                  f"lr={current_lr:.6f}  time={epoch_seconds:.1f}s")
        else:
            print(f"Epoch {epoch:03d}/{args.epochs:03d}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                  f"lr={current_lr:.6f}  time={epoch_seconds:.1f}s")

    total_training_seconds = time.time() - training_start
    mean_epoch_seconds = (
        sum(row["epoch_seconds"] for row in history) / len(history) if history else 0.0
    )

    # Final test eval on the held-out split (only touched here).
    # Reload the best checkpoint first so the numbers are for the model
    # that would be quoted in the report.
    test_acc: float | None = None
    test_acc_tta: float | None = None
    if args.test_at_end:
        print("\nEvaluating the best checkpoint on the test split...")
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        unwrap_compiled(model).load_state_dict(checkpoint["model_state"])
        with torch.no_grad():
            test_loss, test_acc, test_targets, test_predictions = run_one_epoch(
                model=model, loader=data.test_loader,
                criterion=criterion, optimizer=None,
                device=device, scaler=None, use_amp=use_amp,
                grad_clip=0.0, epoch=0, phase="test",
            )
        save_confusion_outputs(run_dir, test_targets, test_predictions,
                               data.class_names, prefix="test")
        save_per_class_accuracy_bar(run_dir, test_targets, test_predictions,
                                    data.class_names, prefix="test")
        print(f"Test loss={test_loss:.4f}  test_acc={test_acc:.4f}")

        # Same checkpoint, just two forward passes per image now
        if args.tta:
            print("Running test-time augmentation (horizontal flip average)...")
            tta_loss, test_acc_tta, tta_targets, tta_predictions = run_tta_evaluation(
                model=model, loader=data.test_loader,
                criterion=criterion,
                device=device, use_amp=use_amp,
            )
            save_confusion_outputs(run_dir, tta_targets, tta_predictions,
                                   data.class_names, prefix="test_tta")
            save_per_class_accuracy_bar(run_dir, tta_targets, tta_predictions,
                                        data.class_names, prefix="test_tta")
            print(f"Test (TTA) loss={tta_loss:.4f}  test_acc_tta={test_acc_tta:.4f}  "
                  f"(delta vs no TTA: {(test_acc_tta - test_acc) * 100:+.2f} pp)")

    # Save key numbers separately so the sweep script can aggregate them
    # without re-reading history.csv.
    summary = {
        "experiment_name": args.experiment_name,
        "model": args.model,
        "pretrained": bool(args.pretrained and args.model != "custom"),
        "freeze_backbone": bool(args.freeze_backbone and args.model != "custom"),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "learning_rate": args.lr,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "augmentation": args.augmentation,
        "ema": bool(args.ema),
        "ema_decay": float(args.ema_decay) if args.ema else 0.0,
        "min_lr": float(args.min_lr),
        "tta": bool(args.tta),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "best_val_acc": float(best_val_acc),
        "best_model": str(best_path),
        "mean_epoch_seconds": float(mean_epoch_seconds),
        "total_training_seconds": float(total_training_seconds),
    }
    if test_acc is not None:
        summary["test_acc"] = float(test_acc)
    if test_acc_tta is not None:
        summary["test_acc_tta"] = float(test_acc_tta)
    save_json(run_dir / "summary.json", summary)

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")
    if test_acc is not None:
        print(f"Test accuracy (held-out split): {test_acc:.4f}")
    if test_acc_tta is not None:
        print(f"Test accuracy with TTA:         {test_acc_tta:.4f}")
    print(f"Mean epoch time: {mean_epoch_seconds:.1f}s  "
          f"(total training: {total_training_seconds/60:.1f} min)")
    print(f"All artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
