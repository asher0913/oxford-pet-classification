"""Run the full ablation sweep reported in the project.

custom-model track (rows A-G): progressive single-variable ablation on the custom CNN,
starting from a bare baseline and adding one technique per row.
transfer-learning track (rows H-J): frozen vs fine-tuned ResNet-18, plus the best-effort
fine-tune+EMA run for the headline number.

80 epochs for custom-model track, and 15 for transfer-learning track.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


TRAIN_MODULE = "pet_classifier.train"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OXFORD_PET_HOST = "www.robots.ox.ac.uk"


OFFLINE_DATA_HELP = """Oxford-IIIT Pet is not present locally and this machine cannot resolve the dataset host.

Prepare the dataset manually on a machine with Internet connectivity,
then transfer it to this machine. The torchvision loader will see the files and skip downloading:
  1. Download:
       https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
       https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

  2. On the Linux GPU server, put both archives under:
       {base_folder}

  3. Extract them there:
       mkdir -p {base_folder}
       tar -xzf images.tar.gz -C {base_folder}
       tar -xzf annotations.tar.gz -C {base_folder}

  4. Expected final structure:
       {base_folder}/images/*.jpg
       {base_folder}/annotations/trainval.txt
       {base_folder}/annotations/test.txt

After that, rerun this script. The torchvision loader will see the files and skip downloading.
"""


def project_path(path: str) -> str:
    """Resolve a CLI path relative to the project root."""

    # resolve relative paths against the project root so the script works from any cwd
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def subprocess_env() -> dict[str, str]:
    """Build the environment used for child training processes."""

    # Make pet_classifier importable even without pip install -e by prepending src/ to PYTHONPATH
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_DIR) if not existing else f"{SRC_DIR}{os.pathsep}{existing}"
    return env


def oxford_pet_ready(data_dir: str) -> bool:
    """Check whether the Oxford-IIIT Pet files already exist locally."""

    # torchvision extracts the dataset into this fixed subfolder name
    base_folder = Path(data_dir) / "oxford-iiit-pet"
    required = [
        base_folder / "images",
        base_folder / "annotations",
        base_folder / "annotations" / "trainval.txt",
        base_folder / "annotations" / "test.txt",
    ]
    return all(path.exists() for path in required)


def can_resolve_dataset_host() -> bool:
    """Check DNS before letting torchvision try the dataset download."""

    # DNS failure is the common offline machine case, so catch it early
    try:
        socket.getaddrinfo(OXFORD_PET_HOST, 443, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return True


def preflight_dataset(data_dir: str) -> None:
    """Give a clearer message if the dataset cannot be downloaded."""

    # If the files are already there, the training scripts can run fully offline.
    if oxford_pet_ready(data_dir):
        print(f"Dataset found under: {Path(data_dir) / 'oxford-iiit-pet'}")
        return

    # Otherwise let torchvision handle the download when the host is reachable.
    if can_resolve_dataset_host():
        print("Dataset not found locally; torchvision will download Oxford-IIIT Pet on the first run.")
        return

    base_folder = Path(data_dir) / "oxford-iiit-pet"
    raise SystemExit(OFFLINE_DATA_HELP.format(base_folder=base_folder))


def build_experiments(
    data_dir: str,
    output_dir: str,
    num_workers: int,
    custom_epochs: int,
    transfer_epochs: int,
) -> list[dict]:
    """Build the experiment list used for the full ablation sweep."""

    # Each dict is {"name": str, "args": list[str]}.
    # Settings shared across all custom runs - only the variable under test changes.
    common_custom = [
        "--model", "custom",
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--image-size", "224",
        "--batch-size", "64",
        "--epochs", str(custom_epochs),
        "--optimizer", "adamw",
        "--lr", "1e-3",
        "--dropout", "0.3",
        "--num-workers", str(num_workers),
        "--download",
        "--amp",
        "--test-at-end",
    ]

    common_transfer = [
        # Transfer runs share a stronger default recipe than the scratch baseline.
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--image-size", "224",
        "--batch-size", "32",
        "--epochs", str(transfer_epochs),
        "--augmentation", "basic",
        "--optimizer", "adamw",
        "--weight-decay", "1e-4",
        "--scheduler", "cosine",
        "--warmup-epochs", "1",
        "--label-smoothing", "0.1",
        "--num-workers", str(num_workers),
        "--download",
        "--amp",
        "--test-at-end",
    ]

    return [
        # ---- custom-model track: progressive single-variable ablation ----
        {
            "name": "custom_baseline",
            "args": [
                "--experiment-name", "custom_baseline",
                *common_custom,
                "--augmentation", "none",
                "--scheduler", "none",
                "--weight-decay", "0.0",
                "--label-smoothing", "0.0",
                "--mixup-alpha", "0.0",
            ],
        },
        {
            "name": "custom_aug",
            "args": [
                "--experiment-name", "custom_aug",
                *common_custom,
                "--augmentation", "strong",
                "--scheduler", "none",
                "--weight-decay", "0.0",
                "--label-smoothing", "0.0",
                "--mixup-alpha", "0.0",
            ],
        },
        {
            "name": "custom_aug_sched",
            "args": [
                "--experiment-name", "custom_aug_sched",
                *common_custom,
                "--augmentation", "strong",
                "--scheduler", "cosine",
                "--warmup-epochs", "3",
                "--weight-decay", "0.0",
                "--label-smoothing", "0.0",
                "--mixup-alpha", "0.0",
            ],
        },
        {
            "name": "custom_aug_sched_wd",
            "args": [
                "--experiment-name", "custom_aug_sched_wd",
                *common_custom,
                "--augmentation", "strong",
                "--scheduler", "cosine",
                "--warmup-epochs", "3",
                "--weight-decay", "1e-4",
                "--label-smoothing", "0.0",
                "--mixup-alpha", "0.0",
            ],
        },
        {
            "name": "custom_aug_sched_wd_ls",
            "args": [
                "--experiment-name", "custom_aug_sched_wd_ls",
                *common_custom,
                "--augmentation", "strong",
                "--scheduler", "cosine",
                "--warmup-epochs", "3",
                "--weight-decay", "1e-4",
                "--label-smoothing", "0.1",
                "--mixup-alpha", "0.0",
            ],
        },
        {
            "name": "custom_full",
            "args": [
                "--experiment-name", "custom_full",
                *common_custom,
                "--augmentation", "strong",
                "--scheduler", "cosine",
                "--warmup-epochs", "3",
                "--weight-decay", "1e-4",
                "--label-smoothing", "0.1",
                "--mixup-alpha", "0.2",
            ],
        },
        {
            # EMA + min_lr + TTA layered on top of row E (not F/Mixup).
            # Mixup + EMA would hurt accuracy on this dataset: the blended
            # training distribution clashes with clean-image val, and EMA
            # compounds that bias. Dropping Mixup for this row fixed it.
            "name": "custom_full_ema",
            "args": [
                "--experiment-name", "custom_full_ema",
                *common_custom,
                "--augmentation", "strong",
                "--scheduler", "cosine",
                "--warmup-epochs", "3",
                "--weight-decay", "1e-4",
                "--label-smoothing", "0.1",
                # No Mixup. The original plan was to keep it 
                # and see if EMA could smooth out the noise, 
                # but it just made things worse.
                "--mixup-alpha", "0.0",
                "--ema",
                "--ema-decay", "0.9999",
                "--min-lr", "1e-5",
                "--tta",
            ],
        },
        # ---- transfer-learning track: freeze vs fine-tune on ResNet-18 ----
        {
            "name": "transfer_resnet18_frozen",
            "args": [
                "--experiment-name", "transfer_resnet18_frozen",
                "--model", "resnet18",
                "--pretrained",
                "--freeze-backbone",
                *common_transfer,
                "--lr", "1e-3",
            ],
        },
        {
            "name": "transfer_resnet18_finetune",
            "args": [
                "--experiment-name", "transfer_resnet18_finetune",
                "--model", "resnet18",
                "--pretrained",
                *common_transfer,
                "--lr", "1e-4",
                "--head-lr-mult", "10",
            ],
        },
        {
            # EMA + min_lr + TTA on top of the fine-tune recipe - same idea as row G.
            "name": "transfer_resnet18_finetune_ema",
            "args": [
                "--experiment-name", "transfer_resnet18_finetune_ema",
                "--model", "resnet18",
                "--pretrained",
                *common_transfer,
                "--lr", "1e-4",
                "--head-lr-mult", "10",
                "--ema",
                "--ema-decay", "0.999",
                "--min-lr", "1e-5",
                "--tta",
            ],
        },
    ]


def parse_args() -> argparse.Namespace:
    """Parse the small set of options for the sweep driver."""

    parser = argparse.ArgumentParser(
        description="Run the full ablation as a sequence of training jobs.",
    )
    parser.add_argument("--data-dir", default="./data",
                        help="Where the Oxford-IIIT Pet dataset lives (or will be downloaded to).")
    parser.add_argument("--output-dir", default="./outputs",
                        help="Parent directory for per-experiment result folders.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes per run.")
    parser.add_argument("--custom-epochs", type=int, default=80,
                        help="Epochs for each custom-model track run. 30 epochs left the from-scratch "
                             "CNN clearly underfit, so I raised the budget to 80.")
    parser.add_argument("--transfer-epochs", type=int, default=15,
                        help="Epochs for each transfer-learning track run. Pretrained features converge fast.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Subset of experiment names to run (default: all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would be executed and exit.")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort the whole sweep if any single run exits non-zero.")
    parser.add_argument("--skip-dataset-check", action="store_true",
                        help="Skip the local / DNS preflight check for the Oxford-IIIT Pet dataset.")
    return parser.parse_args()


def format_command(args: list[str]) -> str:
    """Format a training command for printing in logs."""

    return " ".join([sys.executable, "-m", TRAIN_MODULE, *args])


def run_single_experiment(args: list[str], env: dict[str, str]) -> int:
    """Run one experiment and return its process exit code."""

    # stdout/stderr inherited so tqdm bars stream live
    command = [sys.executable, "-m", TRAIN_MODULE, *args]
    process = subprocess.run(command, check=False, env=env)
    return process.returncode


def find_latest_run_dir(output_dir: Path, experiment_name: str) -> Path | None:
    """Find the newest timestamped run folder for one experiment."""

    # Match <name>_YYYYMMDD_HHMMSS exactly with a regex instead of glob,
    # because "custom_aug" is a prefix of "custom_aug_sched" and a glob
    # would silently match the wrong folder.
    pattern = re.compile(rf"^{re.escape(experiment_name)}_\d{{8}}_\d{{6}}$")
    matches = sorted(
        entry for entry in output_dir.iterdir()
        if entry.is_dir() and pattern.match(entry.name)
    )
    return matches[-1] if matches else None


VIS_MODULE = "pet_classifier.visualize"

# Only render grids for the two headline models (best custom-model track and best transfer-learning track).
VISUALISATION_TARGETS = (
    "custom_full_ema",
    "transfer_resnet18_finetune_ema",
)

# Correct + incorrect on both splits.
VISUALISATION_VIEWS = (
    ("val", "incorrect"),
    ("val", "correct"),
    ("test", "incorrect"),
    ("test", "correct"),
)


def run_visualisations(
    output_dir: Path,
    experiments: list[dict],
    env: dict[str, str],
) -> None:
    """Create the prediction and Grad-CAM grids after training finishes."""

    # Run after the sweep so figures use the final checkpoints.
    # Failures are logged but don't abort - training results are still valid.
    ran_names = {experiment["name"] for experiment in experiments}
    for target in VISUALISATION_TARGETS:
        if target not in ran_names:
            continue
        run_dir = find_latest_run_dir(output_dir, target)
        if run_dir is None:
            print(f"  [visualise] No run folder found for {target}, skipping.")
            continue
        checkpoint = run_dir / "best_model.pt"
        if not checkpoint.exists():
            print(f"  [visualise] {checkpoint} missing, skipping.")
            continue

        for split, filter_mode in VISUALISATION_VIEWS:
            print(f"  [visualise] {target} / {split} / {filter_mode}")
            command = [
                sys.executable, "-m", VIS_MODULE,
                "--checkpoint", str(checkpoint),
                "--split", split,
                "--filter", filter_mode,
                "--num-samples", "16",
                "--ncols", "4",
            ]
            result = subprocess.run(command, check=False, env=env)
            if result.returncode != 0:
                print(
                    f"  [visualise] Grid failed for {target} "
                    f"({split}/{filter_mode}); continuing with the rest."
                )


def plot_ablation_chart(rows: list[dict], output_dir: Path) -> Path | None:
    """Plot the summary CSV numbers as a compact bar chart."""

    # Grouped bar chart of val/test accuracy for each row in the ablation.
    # matplotlib import is deferred so a missing install doesn't block training.
    if not rows:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [chart] matplotlib unavailable ({exc}); skipping ablation_chart.png.")
        return None

    # preserve definition order so the chart reads A→G→transfer-learning track, same as the report
    ordered = [row for row in rows if "experiment_name" in row]
    if not ordered:
        return None

    names = [row["experiment_name"] for row in ordered]
    val_accs = [float(row.get("best_val_acc") or 0.0) * 100.0 for row in ordered]
    test_accs = [float(row.get("test_acc") or 0.0) * 100.0 for row in ordered]
    # only show TTA bars if any row actually recorded a TTA number
    tta_accs = [float(row.get("test_acc_tta") or 0.0) * 100.0 for row in ordered]
    has_tta = any(value > 0.0 for value in tta_accs)

    import numpy as np  # local import to match the matplotlib pattern above

    x = np.arange(len(names))
    width = 0.26 if has_tta else 0.38

    fig, ax = plt.subplots(figsize=(max(9.0, 1.4 * len(names)), 5.2))
    if has_tta:
        bars_val = ax.bar(x - width, val_accs, width,
                          label="Validation accuracy", color="#3b7dd8")
        bars_test = ax.bar(x, test_accs, width,
                           label="Test accuracy", color="#d86b3b")
        bars_tta = ax.bar(x + width, tta_accs, width,
                          label="Test accuracy (TTA)", color="#2ca06c")
        bar_groups = (bars_val, bars_test, bars_tta)
    else:
        bars_val = ax.bar(x - width / 2, val_accs, width,
                          label="Validation accuracy", color="#3b7dd8")
        bars_test = ax.bar(x + width / 2, test_accs, width,
                           label="Test accuracy", color="#d86b3b")
        bar_groups = (bars_val, bars_test)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ablation: validation vs held-out test accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    for bar_group in bar_groups:
        for bar in bar_group:
            height = bar.get_height()
            if height <= 0.0:
                continue
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    fig.tight_layout()
    chart_path = output_dir / "ablation_chart.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return chart_path


def aggregate_summaries(
    output_dir: Path,
    experiments: list[dict],
) -> tuple[Path | None, list[dict]]:
    """Collect finished run summaries into one CSV and JSON."""

    # Collect summary.json from every completed run and write a single CSV + JSON.
    rows: list[dict] = []
    for experiment in experiments:
        run_dir = find_latest_run_dir(output_dir, experiment["name"])
        if run_dir is None:
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["run_dir"] = str(run_dir)
        rows.append(summary)

    if not rows:
        return None, []

    # union of keys so a missing field in one run doesn't silently drop a column
    all_keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    csv_path = output_dir / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in all_keys})

    json_path = output_dir / "ablation_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    return csv_path, rows


def main() -> None:
    """Run the selected experiments, then aggregate and visualise results."""

    cli = parse_args()

    data_dir = project_path(cli.data_dir)
    output_dir = Path(project_path(cli.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # all_experiments is the full list; experiments is the filtered subset.
    # Keeping both means aggregation still picks up old runs when --only is used.
    all_experiments = build_experiments(
        data_dir=data_dir,
        output_dir=str(output_dir),
        num_workers=cli.num_workers,
        custom_epochs=cli.custom_epochs,
        transfer_epochs=cli.transfer_epochs,
    )
    experiments = all_experiments

    if cli.only:
        # Check names before starting any long GPU job.
        wanted = set(cli.only)
        known = {experiment["name"] for experiment in all_experiments}
        unknown = wanted - known
        if unknown:
            raise SystemExit(
                f"Unknown experiment name(s): {sorted(unknown)}. "
                f"Choose from: {sorted(known)}"
            )
        experiments = [experiment for experiment in all_experiments if experiment["name"] in wanted]

    if not cli.skip_dataset_check and not cli.dry_run:
        preflight_dataset(data_dir)

    print(f"Planning to run {len(experiments)} experiment(s):")
    for experiment in experiments:
        print(f"  - {experiment['name']}")
    print()

    if cli.dry_run:
        # Useful for checking the exact commands before submitting the sweep.
        for experiment in experiments:
            print(f"[dry-run] {experiment['name']}")
            print(f"    {format_command(experiment['args'])}")
        return

    env = subprocess_env()
    results: list[tuple[str, int, float]] = []
    sweep_start = time.time()

    for experiment in experiments:
        # Keep the loop simple so a failed run is easy to spot in the terminal.
        name = experiment["name"]
        print("=" * 78)
        print(f"  Starting experiment: {name}")
        print("=" * 78)
        print(format_command(experiment["args"]))
        print()

        run_start = time.time()
        exit_code = run_single_experiment(experiment["args"], env)
        elapsed = time.time() - run_start
        results.append((name, exit_code, elapsed))

        print()
        print(f"  Finished {name}: exit_code={exit_code}, elapsed={elapsed/60:.1f} min")
        print()

        if exit_code != 0 and cli.stop_on_error:
            print(f"  Aborting sweep because {name} failed and --stop-on-error is set.")
            break

    total_elapsed = time.time() - sweep_start

    # aggregate over the full list so re-running a single row still produces a complete CSV
    aggregate_path, aggregate_rows = aggregate_summaries(output_dir, all_experiments)
    chart_path = plot_ablation_chart(aggregate_rows, output_dir)

    # use the full experiment list here too so --only doesn't skip visualisation
    print()
    print("=" * 78)
    print("  Rendering visualisations for the headline models")
    print("=" * 78)
    run_visualisations(output_dir, all_experiments, env)

    print()
    print("=" * 78)
    print("  Sweep summary")
    print("=" * 78)
    for name, exit_code, elapsed in results:
        status = "OK" if exit_code == 0 else f"FAILED ({exit_code})"
        print(f"  {name:<32s} {status:<14s} {elapsed/60:6.1f} min")
    print(f"  Total wall-clock: {total_elapsed/60:.1f} min")
    print(f"  Result folders under: {output_dir}")
    if aggregate_path is not None:
        print(f"  Aggregated table:     {aggregate_path}")
    if chart_path is not None:
        print(f"  Ablation bar chart:   {chart_path}")

    # non-zero exit so shell chaining / CI can detect failures
    if any(code != 0 for _, code, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
