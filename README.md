# Oxford-IIIT Pet Classification

An end-to-end PyTorch project for fine-grained classification across 37 cat
and dog breeds. It compares a custom residual CNN trained from scratch with
transfer-learning baselines, and includes reproducible ablations, checkpoint
evaluation, batch inference, and Grad-CAM visualizations.

Two modeling tracks share the same data and evaluation pipeline:

* **Transfer learning.** Pretrained torchvision networks
  (ResNet, VGG, MobileNet). Two strategies are supported: feature
  extraction with a frozen backbone, and full fine-tuning with
  differential learning rates.
* **Custom model.** `PetResNet`, a residual network with
  Squeeze-and-Excite channel attention designed specifically for
  this dataset. Trained from scratch, no pretrained weights.

The reported experiment used a deterministic 80/20 split of the official
training data and kept the official test split isolated until final
evaluation. The best fine-tuned ResNet-18 reached **89.8% test accuracy**
with EMA and horizontal-flip TTA. The best custom model reached **49.7% test
accuracy**, providing a clear from-scratch baseline for the transfer-learning
comparison.

## Project layout

```
oxford-pet-classification/
├── README.md
├── pyproject.toml
├── requirements.txt
├── scripts/
│   └── run_recommended_experiments.py
└── src/pet_classifier/
    ├── __init__.py
    ├── data.py          # dataset, train/val split, transforms, test loader
    ├── ema.py           # bias-corrected exponential moving average
    ├── models.py        # PetResNet, transfer wrappers, differential LR helper
    ├── train.py         # CLI training entry point
    ├── evaluate.py      # re-score a checkpoint on val + test
    ├── predict.py       # top-k inference on files or a folder
    ├── visualize.py     # prediction grids + Grad-CAM grids
    ├── gradcam.py       # from-scratch Grad-CAM implementation
    └── utils.py         # seeding, metrics, plotting, checkpoint helpers
```

## Setup

From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-compatible PyTorch build that matches the target
machine. The command below works on CUDA 12.1; substitute the right
index URL from https://pytorch.org/get-started/locally/ for other
CUDA versions.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

Quick sanity check:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

If `torch.cuda.is_available()` is `False`, reinstall `torch` with
the CUDA build that matches the installed driver.

### Dataset setup on servers without internet

The recommended script passes `--download`, so torchvision will try
to fetch Oxford-IIIT Pet automatically. If the GPU machine has no
internet access or DNS, the training process will raise an error
of the form:

```text
Temporary failure in name resolution
urllib.error.URLError
```

That is a server network problem, not a training-code problem. In
that case, download the two archives on another machine and copy
them to the server:

```text
https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```

Then prepare the expected torchvision folder:

```bash
cd oxford-pet-classification
mkdir -p data/oxford-iiit-pet

# Put images.tar.gz and annotations.tar.gz in data/oxford-iiit-pet first.
tar -xzf data/oxford-iiit-pet/images.tar.gz -C data/oxford-iiit-pet
tar -xzf data/oxford-iiit-pet/annotations.tar.gz -C data/oxford-iiit-pet
```

Expected final structure:

```text
data/oxford-iiit-pet/images/*.jpg
data/oxford-iiit-pet/annotations/trainval.txt
data/oxford-iiit-pet/annotations/test.txt
```

Once those files exist, rerun:

```bash
python scripts/run_recommended_experiments.py
```

## Run the experiment suite

One Python driver runs the full custom-model track ablation (seven custom-CNN
runs, 80 epochs each) and all three transfer-learning track transfer runs
(frozen / fine-tune / fine-tune + EMA, 15 epochs each). Ten runs in
total. On an RTX 3090 it is roughly 85 minutes; on older cards it
takes longer.

The last custom-CNN row (`custom_full_ema`) and the last transfer
row (`transfer_resnet18_finetune_ema`) add three extra training /
inference tricks on top of the previous row:

* **EMA** — an exponential moving average of model weights (Polyak
  averaging). The shadow weights are used for validation and saved
  as `best_model.pt`. The decay uses
  `eff_decay = min(decay, (1+n)/(10+n))` so during the first few
  hundred updates the shadow is not dominated by the random Kaiming
  init. The raw training weights are also saved to
  `best_model_raw.pt` whenever EMA is on, as a backup. When EMA is
  active `history.csv` logs `val_acc` (shadow) and `val_acc_raw`
  (raw training weights) side by side, so any per-epoch divergence
  between them is visible.
* **`min_lr = 1e-5`** — non-zero floor on the cosine schedule so the
  last few epochs still make small updates instead of effectively
  stopping.
* **Horizontal-flip TTA** — at the final `--test-at-end` evaluation,
  predictions are averaged over the image and its left-right flip.
  Training itself is unchanged. `summary.json` records both
  `test_acc` and `test_acc_tta` so the with-vs-without-TTA
  comparison is available afterwards.

None of EMA, TTA or `min_lr` adds pretrained weights or changes the
architecture, so custom-model track stays a from-scratch custom CNN and transfer-learning track
still uses one of the permitted torchvision architectures.

```bash
python scripts/run_recommended_experiments.py
```

Override locations, worker count, or run only part of the sweep:

```bash
python scripts/run_recommended_experiments.py \
    --data-dir /mnt/data \
    --output-dir /mnt/out \
    --num-workers 8

python scripts/run_recommended_experiments.py \
    --only custom_baseline custom_aug

python scripts/run_recommended_experiments.py --dry-run
```

Epoch counts can also be overridden on the driver, for a quick
smoke test with fewer epochs or for a longer run than the defaults:

```bash
python scripts/run_recommended_experiments.py \
    --custom-epochs 80 --transfer-epochs 15
```

The driver runs `python -m pet_classifier.train` as a subprocess for each
experiment, so it uses exactly the same training code as the
individual commands shown below.

## Generated artifacts

A run named `foo` creates `outputs/foo_<timestamp>/` with:

| File                                     | What it is                                            |
| ---------------------------------------- | ----------------------------------------------------- |
| `best_model.pt`                          | Checkpoint with the highest validation accuracy       |
| `last_model.pt`                          | Final checkpoint after the last epoch                 |
| `run_config.json`                        | All CLI args plus class names and split sizes         |
| `history.csv`                            | Per-epoch loss, accuracy and learning rate            |
| `training_curves.png`                    | Loss and accuracy curves                              |
| `validation_confusion_matrix.png` / csv  | Best-epoch confusion matrix on the validation split   |
| `validation_classification_report.txt`   | Per-class precision / recall / F1 (validation)        |
| `validation_per_class_accuracy.png`      | Per-class accuracy bar chart (validation)             |
| `test_confusion_matrix.png` / csv        | Confusion matrix on the official test split          |
| `test_classification_report.txt`         | Per-class precision / recall / F1 (test)              |
| `test_per_class_accuracy.png`            | Per-class accuracy bar chart (test)                   |
| `test_tta_confusion_matrix.png` / csv    | TTA confusion matrix (only when `--tta` was used)     |
| `test_tta_classification_report.txt`     | Per-class report under TTA (only when `--tta` was used) |
| `test_tta_per_class_accuracy.png`        | Per-class accuracy bar chart under TTA                |
| `summary.json`                           | Best val accuracy, test accuracy (and TTA test accuracy), best model path |

Test-set artifacts only appear if the run used `--test-at-end`,
which the recommended driver does for every experiment.

Once every run in the sweep has finished, the driver also writes
three project-level artifacts plus a visualisation folder next to
each headline checkpoint:

| Artifact                                   | What it is                                           |
| ------------------------------------------ | ---------------------------------------------------- |
| `outputs/ablation_summary.csv`             | One row per experiment with every headline number   |
| `outputs/ablation_summary.json`            | Same information in JSON, easier to parse in notebooks |
| `outputs/ablation_chart.png`               | Grouped bar chart of val / test accuracy across the full sweep |
| `outputs/custom_full_ema_*/visualisation/`     | Prediction grids + Grad-CAM overlays for the custom-model track best model (val/test × correct/incorrect) |
| `outputs/transfer_resnet18_finetune_ema_*/visualisation/` | Same four grids for the transfer-learning track best model |

Between these files the run folders give per-experiment confusion
matrices and training curves, `ablation_chart.png` gives the overall
val/test comparison, and the `visualisation/` folders contain the
Grad-CAM grids for the two best models.

## Individual commands

### Custom model progressive ablation

The runs below add one technique at a time. Each row only changes
one thing compared with the previous row, so any accuracy change
between rows can be matched to that single technique.

```bash
# A. Baseline: nothing but the architecture.
python -m pet_classifier.train \
  --experiment-name custom_baseline \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation none --scheduler none \
  --weight-decay 0.0 --label-smoothing 0.0 --mixup-alpha 0.0 \
  --download --amp --test-at-end

# B. + strong augmentation. This is the first ablation step,
#    isolating the effect of data augmentation.
python -m pet_classifier.train \
  --experiment-name custom_aug \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler none \
  --weight-decay 0.0 --label-smoothing 0.0 --mixup-alpha 0.0 \
  --download --amp --test-at-end

# C. + warmup + cosine LR schedule.
python -m pet_classifier.train \
  --experiment-name custom_aug_sched \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler cosine --warmup-epochs 3 \
  --weight-decay 0.0 --label-smoothing 0.0 --mixup-alpha 0.0 \
  --download --amp --test-at-end

# D. + L2 weight decay.
python -m pet_classifier.train \
  --experiment-name custom_aug_sched_wd \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler cosine --warmup-epochs 3 \
  --weight-decay 1e-4 --label-smoothing 0.0 --mixup-alpha 0.0 \
  --download --amp --test-at-end

# E. + label smoothing.
python -m pet_classifier.train \
  --experiment-name custom_aug_sched_wd_ls \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler cosine --warmup-epochs 3 \
  --weight-decay 1e-4 --label-smoothing 0.1 --mixup-alpha 0.0 \
  --download --amp --test-at-end

# F. + Mixup.
python -m pet_classifier.train \
  --experiment-name custom_full \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler cosine --warmup-epochs 3 \
  --weight-decay 1e-4 --label-smoothing 0.1 --mixup-alpha 0.2 \
  --download --amp --test-at-end

# G. + EMA + cosine min_lr + TTA (best custom-model configuration).
#    EMA and min_lr change training; TTA only runs at --test-at-end
#    and gives an extra ``test_acc_tta`` number.
#    This row is built on top of E (``custom_aug_sched_wd_ls``), not F
#    (``custom_full``), because Mixup hurt val accuracy on this
#    dataset and did not combine well with EMA. So ``--mixup-alpha``
#    is set back to 0 here.
python -m pet_classifier.train \
  --experiment-name custom_full_ema \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler cosine --warmup-epochs 3 \
  --weight-decay 1e-4 --label-smoothing 0.1 --mixup-alpha 0.0 \
  --ema --ema-decay 0.9999 --min-lr 1e-5 --tta \
  --download --amp --test-at-end
```

`--pretrained` is off by default, so every custom-model run above trains
from random initialisation. The `custom` model has no pretrained
weights to load anyway, but the flag is left off explicitly so the
saved `run_config.json` clearly shows that none were used.

### ResNet-18 transfer learning

```bash
# Frozen backbone (feature extraction).
python -m pet_classifier.train \
  --experiment-name transfer_resnet18_frozen \
  --model resnet18 --pretrained --freeze-backbone \
  --image-size 224 --batch-size 32 --epochs 15 \
  --augmentation basic \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 \
  --scheduler cosine --warmup-epochs 1 --label-smoothing 0.1 \
  --download --amp --test-at-end

# Full fine-tune with differential LR.
python -m pet_classifier.train \
  --experiment-name transfer_resnet18_finetune \
  --model resnet18 --pretrained \
  --image-size 224 --batch-size 32 --epochs 15 \
  --augmentation basic \
  --optimizer adamw --lr 1e-4 --head-lr-mult 10 \
  --weight-decay 1e-4 --scheduler cosine --warmup-epochs 1 \
  --label-smoothing 0.1 \
  --download --amp --test-at-end

# Fine-tune + EMA + cosine min_lr + TTA (best transfer-learning configuration).
python -m pet_classifier.train \
  --experiment-name transfer_resnet18_finetune_ema \
  --model resnet18 --pretrained \
  --image-size 224 --batch-size 32 --epochs 15 \
  --augmentation basic \
  --optimizer adamw --lr 1e-4 --head-lr-mult 10 \
  --weight-decay 1e-4 --scheduler cosine --warmup-epochs 1 \
  --label-smoothing 0.1 \
  --ema --ema-decay 0.999 --min-lr 1e-5 --tta \
  --download --amp --test-at-end
```

`--lr 1e-4` is the backbone learning rate. `--head-lr-mult 10`
gives the new classifier head an effective `1e-3`, so the fresh
head trains faster than the pretrained backbone.

## 5. Re-score an existing checkpoint

```bash
python -m pet_classifier.evaluate \
  --checkpoint outputs/custom_full_*/best_model.pt \
  --device auto
```

Produces fresh `validation_*` and `test_*` artifacts under an
`evaluation/` sub-folder beside the checkpoint.

## 6. Prediction and Grad-CAM visualisation

```bash
python -m pet_classifier.visualize \
  --checkpoint outputs/transfer_resnet18_finetune_*/best_model.pt \
  --num-samples 16 --filter incorrect --split val
```

Writes `predictions_grid_val_incorrect.png` and
`gradcam_grid_val_incorrect.png` in a `visualisation/` folder next
to the checkpoint. These grids show the model's mistakes: Grad-CAM
on misclassified images reveals whether the model is looking at the
animal or at the background.

Set `--filter correct` or `--filter any` for the other views.

## 7. Inference on new images

```bash
python -m pet_classifier.predict \
  --checkpoint outputs/transfer_resnet18_finetune_*/best_model.pt \
  --input /path/to/image_or_folder \
  --output-csv predictions.csv \
  --top-k 5
```

## 8. Notes on methodology

* Validation accuracy is used to pick the best checkpoint. The
  test split is only touched once at the end of each run (turned
  on by `--test-at-end`), so the test number is an independent
  check rather than something used for model selection.
* The seven custom-CNN runs (A `custom_baseline` through G
  `custom_full_ema`) only change one thing per row, so the
  effect of each technique can be read off from row to row. A→B
  is the augmentation step. C–F add LR scheduling, weight decay,
  label smoothing and Mixup one at a time. G adds EMA + cosine
  `min_lr` + TTA on top of E (skipping Mixup, which made things
  worse). On the transfer side, `transfer_resnet18_frozen` vs
  `transfer_resnet18_finetune` is the feature-extraction vs
  full-fine-tune comparison from Lab 6, and
  `transfer_resnet18_finetune_ema` adds the same EMA + `min_lr` +
  TTA package on top of fine-tuning.
* The default seed is 42. Use `--deterministic` for fully
  reproducible runs, although it costs a bit of throughput.
* `PetResNet` has about 2.7M parameters. The classifier is just
  one `Linear(256, 37)` because the global average pooling already
  reduces the feature map to 256 numbers, so a wider FC block is
  not needed.
