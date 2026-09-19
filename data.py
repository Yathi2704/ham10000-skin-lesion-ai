"""Canonical HAM10000 split + DataLoaders.

Single source of truth for:
  * the fixed label map shared by training and the server (design.md → Data Models)
  * the ONE stratified 70/15/15 split, seed 42 (requirements: model-training 1–3)
  * the preprocessing pipeline: resize 224 → tensor → ImageNet normalize

Everything that consumes data (train.py, evaluate.py) must go through
`make_split` with the default seed — never re-split per model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Fixed label map (design.md): 0=akiec, 1=bcc, 2=bkl, 3=df, 4=mel, 5=nv, 6=vasc
# --------------------------------------------------------------------------- #
CLASS_NAMES: tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_LABELS: dict[str, str] = {
    "akiec": "Actinic keratosis",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
LABEL_MAP: dict[int, str] = dict(enumerate(CLASS_NAMES))
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

DATASET_NAME = "pranay-43/HAM10000"
SEED = 42
SPLIT_FRACTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)  # train, val, test
PARTITIONS: tuple[str, ...] = ("train", "val", "test")

IMAGE_SIZE = 224
BATCH_SIZE = 32
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Split (pure function of the label array → fully reproducible)
# --------------------------------------------------------------------------- #
def make_split(
    labels: Sequence[int] | np.ndarray,
    seed: int = SEED,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> dict[str, np.ndarray]:
    """Stratified train/val/test row indices for `labels`.

    Two-stage sklearn split, both stages seeded with `seed`:
      1. all → train | (val+test)
      2. (val+test) → val | test
    Returns sorted index arrays keyed by partition name.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty 1-D array")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {fractions}")
    train_frac, val_frac, test_frac = fractions

    all_idx = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(
        all_idx,
        test_size=val_frac + test_frac,
        stratify=labels,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_frac / (val_frac + test_frac),
        stratify=labels[rest_idx],
        random_state=seed,
    )
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def class_counts(labels: np.ndarray, indices: np.ndarray | None = None) -> dict[str, int]:
    """Per-class counts (in canonical class order) for the rows in `indices`."""
    labels = np.asarray(labels)
    subset = labels if indices is None else labels[indices]
    counts = np.bincount(subset, minlength=NUM_CLASSES)
    return {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)}


def split_fingerprint(split: dict[str, np.ndarray], image_ids: Sequence[str] | None = None) -> str:
    """Short sha256 over the split — identical runs must print identical fingerprints.

    If `image_ids` are given the hash is over image ids (robust to row order),
    otherwise over row indices.
    """
    h = hashlib.sha256()
    for part in PARTITIONS:
        idx = split[part]
        members = [str(image_ids[i]) for i in idx] if image_ids is not None else [str(i) for i in idx]
        h.update(part.encode())
        h.update("\n".join(members).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def print_split_report(labels: np.ndarray, split: dict[str, np.ndarray], image_ids: Sequence[str] | None = None) -> None:
    """Print per-class counts for each partition (requirements: model-training 3)."""
    labels = np.asarray(labels)
    per_part = {part: class_counts(labels, idx) for part, idx in split.items()}
    total = class_counts(labels)
    header = f"{'class':<8}" + "".join(f"{p:>8}" for p in PARTITIONS) + f"{'total':>8}"
    print(header)
    print("-" * len(header))
    for name in CLASS_NAMES:
        row = f"{name:<8}" + "".join(f"{per_part[p][name]:>8}" for p in PARTITIONS) + f"{total[name]:>8}"
        print(row)
    print("-" * len(header))
    print(f"{'all':<8}" + "".join(f"{len(split[p]):>8}" for p in PARTITIONS) + f"{len(labels):>8}")
    print(f"seed={SEED}  fractions={SPLIT_FRACTIONS}  fingerprint={split_fingerprint(split, image_ids)}")


# --------------------------------------------------------------------------- #
# Loading HAM10000 into one pool with canonical integer labels
# --------------------------------------------------------------------------- #
def _image_id_from_path(path: str | None, fallback: int) -> str:
    if not path:
        return f"row{fallback}"
    return os.path.splitext(os.path.basename(path))[0]


def load_pool(dataset_name: str = DATASET_NAME) -> Any:
    """Load every HF split of `dataset_name` into ONE `datasets.Dataset` pool.

    The pool gets two canonical columns:
      * `label`    — int in the fixed label map above (mapped by class *name*)
      * `image_id` — stable id (ISIC_xxxxxxx) so the split is reproducible by content

    Rows are sorted by `image_id` so the seeded split does not depend on the
    order the files happened to be listed in. The HF repo's own train/val/test
    sharding is deliberately ignored: the canonical split is made here, once.
    """
    from datasets import ClassLabel, Image, concatenate_datasets, load_dataset  # heavy import, keep lazy

    dsd = load_dataset(dataset_name)
    parts = [dsd[k] for k in sorted(dsd.keys())]
    pool = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    # ---- class names → canonical ids ------------------------------------ #
    if "label" in pool.features and isinstance(pool.features["label"], ClassLabel):
        source_names = pool.features["label"].names
        raw_col = "label"
        def to_canonical(v: Any) -> int:
            return CLASS_TO_INDEX[source_names[int(v)]]
    elif "dx" in pool.features:  # HAM10000 metadata-style column
        raw_col = "dx"
        def to_canonical(v: Any) -> int:
            return CLASS_TO_INDEX[str(v)]
    else:
        raise ValueError(
            f"{dataset_name}: cannot find a class column (expected ClassLabel 'label' or string 'dx'); "
            f"features = {list(pool.features)}"
        )

    # ---- stable image ids ------------------------------------------------- #
    undecoded = pool.cast_column("image", Image(decode=False))
    if "image_id" in pool.features:
        image_ids = [str(x) for x in pool["image_id"]]
    else:
        image_ids = [_image_id_from_path(rec.get("path"), i) for i, rec in enumerate(undecoded["image"])]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"{dataset_name}: duplicate image ids in pool — refusing to split ambiguous data")

    canonical = [to_canonical(v) for v in pool[raw_col]]
    stale = [c for c in ("label", "image_id") if c in pool.features]
    if stale:
        pool = pool.remove_columns(stale)
    pool = pool.add_column("label", canonical)
    pool = pool.add_column("image_id", image_ids)
    pool = pool.sort("image_id")

    counts = class_counts(np.asarray(pool["label"]))
    empty = [name for name, n in counts.items() if n == 0]
    if empty:
        raise ValueError(f"{dataset_name}: no rows for classes {empty} — a stratified 7-class split is impossible")
    return pool


# --------------------------------------------------------------------------- #
# Preprocessing + DataLoaders
# --------------------------------------------------------------------------- #
def get_transform() -> transforms.Compose:
    """resize 224 → tensor → ImageNet normalize (identical for train/val/test/server)."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class HamDataset(Dataset):
    """Torch view over a subset (index array) of the HF pool."""

    def __init__(self, pool: Any, indices: np.ndarray, transform: transforms.Compose | None = None):
        self.pool = pool
        self.indices = np.asarray(indices)
        self.transform = transform or get_transform()
        self.labels = np.asarray(pool["label"])[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.pool[int(self.indices[i])]
        image = row["image"].convert("RGB")
        return self.transform(image), int(row["label"])


def build_dataloaders(
    pool: Any,
    split: dict[str, np.ndarray],
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    seed: int = SEED,
) -> dict[str, DataLoader]:
    """train (shuffled, seeded) / val / test loaders over the canonical split."""
    transform = get_transform()
    loaders: dict[str, DataLoader] = {}
    for part in PARTITIONS:
        ds = HamDataset(pool, split[part], transform)
        shuffle = part == "train"
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        loaders[part] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    return loaders


def load_split_data(dataset_name: str = DATASET_NAME, seed: int = SEED) -> tuple[Any, dict[str, np.ndarray]]:
    """Convenience: pool + canonical split, with the per-class report printed."""
    pool = load_pool(dataset_name)
    labels = np.asarray(pool["label"])
    split = make_split(labels, seed=seed)
    print(f"dataset={dataset_name}  rows={len(labels)}")
    print_split_report(labels, split, pool["image_id"])
    return pool, split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and report the canonical HAM10000 split.")
    parser.add_argument("--dataset", default=DATASET_NAME, help="HF dataset id (default: %(default)s)")
    args = parser.parse_args()
    load_split_data(args.dataset)


if __name__ == "__main__":
    main()
