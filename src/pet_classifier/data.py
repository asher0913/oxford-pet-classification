"""Data loading and augmentation for Oxford-IIIT Pet.

The trainval split is divided 80/20 into train/val, while the official
test split is only touched at the very end. Two OxfordIIITPet datasets
are built (same seed) because train and val need different transforms -
then Subset picks the right indices for each.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.error import URLError

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Hardcoded fallback in case the loader version doesn't expose .classes
OXFORD_PET_CLASSES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]

# ImageNet mean/std - matching these is needed for the pretrained ResNet-18
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The long hint is mainly for the Linux server, where DNS/downloads can fail.
OXFORD_PET_DOWNLOAD_HINT = """Oxford-IIIT Pet dataset is not available locally and automatic download failed.

If the machine has no internet/DNS, prepare the dataset manually:

  1. Download on a machine with internet:
       https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
       https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

  2. Copy both archives to:
       {base_folder}

  3. Extract:
       mkdir -p {base_folder}
       tar -xzf images.tar.gz -C {base_folder}
       tar -xzf annotations.tar.gz -C {base_folder}

Expected final structure:
  {base_folder}/images/*.jpg
  {base_folder}/annotations/trainval.txt
  {base_folder}/annotations/test.txt
"""


@dataclass(frozen=True)
class DataBundle:
    """Small container so the train script gets all data bits together."""

    # what build_dataloaders returns
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    train_indices: list[int]
    val_indices: list[int]


def build_transforms(image_size: int, augmentation: str) -> tuple[transforms.Compose, transforms.Compose]:
    """Build train/eval transforms for the chosen augmentation level."""

    # eval transform is always deterministic so val/test numbers are reproducible
    augmentation = augmentation.lower()

    if augmentation == "none":
        # Baseline transform: resize only, no randomness.
        train_steps: list[object] = [transforms.Resize((image_size, image_size))]

    elif augmentation == "basic":
        # Just flip - pets are roughly left-right symmetric
        train_steps = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

    elif augmentation == "strong":
        # RandomResizedCrop + flip + TrivialAugmentWide is the modern default
        # that doesn't need hyperparameter tuning. RandomErasing added at the
        # end below acts like Cutout.
        train_steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.80, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.TrivialAugmentWide(),
        ]

    else:
        raise ValueError("augmentation must be one of: none, basic, strong")

    train_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # RandomErasing has to come AFTER Normalize, otherwise the zero patch
    # ends up at a non-zero value once normalised
    if augmentation == "strong":
        train_steps.append(transforms.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.3)))

    eval_steps = [
        # Validation/test should not depend on random crops or flips.
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def get_class_names(dataset: datasets.OxfordIIITPet) -> list[str]:
    """Read class names in a way that works across torchvision versions."""

    # try the public name, fall back to the private one, then the hardcoded list
    classes = getattr(dataset, "classes", None)
    if classes:
        return list(classes)

    private_classes = getattr(dataset, "_CLASSES", None)
    if private_classes:
        return list(private_classes)

    return OXFORD_PET_CLASSES.copy()


def split_indices(length: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Make the fixed train/val split used by all experiments."""

    # Deterministic split. Using torch's RNG (not numpy) so the seed gives
    # the same split everywhere.
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be between 0.05 and 0.5")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length, generator=generator).tolist()
    val_size = int(round(length * val_fraction))

    # sort within each subset for slightly nicer disk read patterns
    val_indices = sorted(permutation[:val_size])
    train_indices = sorted(permutation[val_size:])
    return train_indices, val_indices


def build_dataloaders(
    data_dir: str | Path,
    image_size: int,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
    augmentation: str,
    seed: int,
    download: bool,
) -> DataBundle:
    """Build the train, validation and test DataLoaders."""

    data_dir = Path(data_dir)
    train_transform, eval_transform = build_transforms(image_size=image_size, augmentation=augmentation)

    # download once with no transform so class names can be read off it
    try:
        # This object is only for metadata and split length.
        base_dataset = datasets.OxfordIIITPet(
            root=str(data_dir),
            split="trainval",
            target_types="category",
            download=download,
            transform=None,
        )
    except URLError as exc:
        base_folder = data_dir / "oxford-iiit-pet"
        raise RuntimeError(OXFORD_PET_DOWNLOAD_HINT.format(base_folder=base_folder)) from exc
    class_names = get_class_names(base_dataset)
    train_indices, val_indices = split_indices(len(base_dataset), val_fraction, seed)

    # train and val need different transforms, so two dataset objects
    train_dataset = datasets.OxfordIIITPet(
        root=str(data_dir), split="trainval", target_types="category",
        download=False, transform=train_transform,
    )
    val_dataset = datasets.OxfordIIITPet(
        root=str(data_dir), split="trainval", target_types="category",
        download=False, transform=eval_transform,
    )
    # The official test split is kept separate from train/val.
    test_dataset = datasets.OxfordIIITPet(
        root=str(data_dir), split="test", target_types="category",
        download=download, transform=eval_transform,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    pin_memory = torch.cuda.is_available()

    # persistent_workers keeps the workers alive between epochs (faster on
    # small datasets); only valid when num_workers > 0
    common_loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # Only the training loader shuffles; the others stay stable for reporting.
    train_loader = DataLoader(train_subset, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        train_indices=train_indices,
        val_indices=val_indices,
    )


def build_image_transform(image_size: int) -> transforms.Compose:
    """Same preprocessing as eval, but for one image at prediction time."""

    # used by predict.py for single-image inference
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize_for_display(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a normalised tensor back to displayable RGB values."""

    # Undo the ImageNet normalisation so the colours look right in matplotlib
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


def label_names_from_indices(class_names: Sequence[str], labels: Sequence[int]) -> list[str]:
    """Map numeric labels back to breed names for plots or prints."""

    return [class_names[int(label)] for label in labels]
