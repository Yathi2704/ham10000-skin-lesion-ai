"""Canonical HAM10000 split + DataLoaders.

Single source of truth for:
  * the fixed label map shared by training and the server (design.md → Data Models)
  * loading the ORIGINAL HAM10000 release (Harvard Dataverse doi:10.7910/DVN/DBW86T:
    `HAM10000_metadata.csv` + `HAM10000_images_part_{1,2}.zip`), de-duplicated by
    image_id and checked against the published counts (10 015 images, 7 classes)
  * the ONE stratified 70/15/15 split, grouped by lesion_id, seed 42
    (requirements: model-training 1–3 as amended)
  * preprocessing: resize 224 → tensor → ImageNet normalize; train-time
    augmentation (random flips + 90° rotations) is applied to the train
    partition only — the eval transform is exactly the server's transform.

Everything that consumes data (train.py, evaluate.py) must go through
`make_split` with the default seed — never re-split per model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from PIL import Image
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

# `dx` spellings seen in the original CSV (the codes) and in re-uploads → canonical code
DX_ALIASES: dict[str, str] = {
    "akiec": "akiec", "actinic_keratoses": "akiec", "actinic_keratosis": "akiec",
    "bcc": "bcc", "basal_cell_carcinoma": "bcc",
    "bkl": "bkl", "benign_keratosis-like_lesions": "bkl", "benign_keratosis": "bkl",
    "df": "df", "dermatofibroma": "df",
    "mel": "mel", "melanoma": "mel",
    "nv": "nv", "melanocytic_nevi": "nv", "melanocytic_nevus": "nv",
    "vasc": "vasc", "vascular_lesions": "vasc", "vascular_lesion": "vasc",
}

# Published facts about the release (Tschandl, Rosendahl & Kittler 2018) — asserted after de-dup
HAM10000_N_IMAGES = 10_015
HAM10000_CLASS_COUNTS: dict[str, int] = {
    "akiec": 327, "bcc": 514, "bkl": 1099, "df": 115, "mel": 1113, "nv": 6705, "vasc": 142,
}

DATA_DIR = Path("data/ham10000")
METADATA_CSV = "HAM10000_metadata.csv"
IMAGE_ZIPS: tuple[str, ...] = ("HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip")

SEED = 42
SPLIT_FRACTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)  # train, val, test
PARTITIONS: tuple[str, ...] = ("train", "val", "test")

IMAGE_SIZE = 224
BATCH_SIZE = 32
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def canonical_dx(value: object) -> int:
    """Map any accepted `dx` spelling to its canonical label id; unknown → loud ValueError."""
    key = str(value).strip().lower().replace(" ", "_")
    if key not in DX_ALIASES:
        raise ValueError(f"unknown dx value {value!r}; accepted: {sorted(DX_ALIASES)}")
    return CLASS_TO_INDEX[DX_ALIASES[key]]


# --------------------------------------------------------------------------- #
# Image stores + Pool
# --------------------------------------------------------------------------- #
class ImageStore(Protocol):
    def open(self, image_id: str) -> Image.Image: ...
    def __contains__(self, image_id: str) -> bool: ...
    def __len__(self) -> int: ...


class FileImageStore:
    """image_id → JPEG, read from the release zips and/or loose *.jpg files under `data_dir`.

    Zip handles are opened lazily per process so the store survives DataLoader workers.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._index: dict[str, tuple[str, Path, str]] = {}  # id → (kind, path, member)
        for zip_name in sorted(p.name for p in self.data_dir.glob("*.zip")):
            zip_path = self.data_dir / zip_name
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    if member.lower().endswith((".jpg", ".jpeg")) and not member.startswith("__MACOSX"):
                        self._index.setdefault(Path(member).stem, ("zip", zip_path, member))
        for path in sorted(self.data_dir.rglob("*.jp*g")):
            self._index.setdefault(path.stem, ("file", path, ""))
        self._handles: dict[tuple[int, Path], zipfile.ZipFile] = {}

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def _zip(self, path: Path) -> zipfile.ZipFile:
        key = (os.getpid(), path)
        if key not in self._handles:
            self._handles[key] = zipfile.ZipFile(path)
        return self._handles[key]

    def open(self, image_id: str) -> Image.Image:
        if image_id not in self._index:
            raise KeyError(f"image {image_id} not found under {self.data_dir}")
        kind, path, member = self._index[image_id]
        if kind == "zip":
            with self._zip(path).open(member) as fh:
                img = Image.open(fh)
                img.load()
        else:
            img = Image.open(path)
            img.load()
        return img.convert("RGB")

    def __getstate__(self):  # DataLoader workers re-open their own handles
        return {"data_dir": self.data_dir, "_index": self._index, "_handles": {}}

    def __setstate__(self, state):
        self.__dict__.update(state)


class NoImageStore:
    """Placeholder when a Pool was loaded metadata-only (load_pool(require_images=False))."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def __contains__(self, image_id: str) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def open(self, image_id: str) -> Image.Image:
        raise RuntimeError(
            f"images were not loaded (metadata-only pool); reload with load_pool({str(self.data_dir)!r}, require_images=True)"
        )


class MemoryImageStore:
    """image_id → in-memory PIL image (tests)."""

    def __init__(self, images: dict[str, Image.Image]):
        self._images = images

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._images

    def __len__(self) -> int:
        return len(self._images)

    def open(self, image_id: str) -> Image.Image:
        return self._images[image_id].convert("RGB")


@dataclass
class Pool:
    """The full de-duplicated dataset, sorted by image_id. Rows are what the split indexes."""

    image_ids: list[str]
    labels: np.ndarray      # canonical ints
    lesion_ids: np.ndarray  # split groups
    images: ImageStore

    def __len__(self) -> int:
        return len(self.image_ids)

    def __post_init__(self) -> None:
        n = len(self.image_ids)
        if not (len(self.labels) == len(self.lesion_ids) == n):
            raise ValueError("Pool columns have different lengths")
        if len(set(self.image_ids)) != n:
            raise ValueError("Pool image_ids are not unique")


# --------------------------------------------------------------------------- #
# Loading the original release
# --------------------------------------------------------------------------- #
def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Read the release CSV → columns image_id, lesion_id, label (canonical int), sorted by image_id.

    Duplicate image_id rows are dropped (first kept) *only if* they agree on lesion_id and dx;
    disagreeing duplicates are a data error and raise.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"metadata CSV not found: {csv_path} — see HANDOVER.md for the download")
    df = pd.read_csv(csv_path, dtype=str)
    missing = [c for c in ("image_id", "lesion_id", "dx") if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}; have {list(df.columns)}")
    df = df[["image_id", "lesion_id", "dx"]].copy()
    df["image_id"] = df["image_id"].str.strip()
    df["lesion_id"] = df["lesion_id"].str.strip()
    df["label"] = [canonical_dx(v) for v in df["dx"]]

    dupes = df[df.duplicated("image_id", keep=False)]
    if len(dupes):
        conflicting = dupes.groupby("image_id")[["lesion_id", "label"]].nunique().max(axis=1) > 1
        if conflicting.any():
            raise ValueError(f"{csv_path}: duplicate image_id rows disagree on lesion_id/dx: {list(conflicting[conflicting].index)[:10]}")
        print(f"note: dropped {len(dupes) - dupes['image_id'].nunique()} exact duplicate image_id rows")
        df = df.drop_duplicates("image_id", keep="first")

    per_lesion = df.groupby("lesion_id")["label"].nunique()
    if (per_lesion > 1).any():
        raise ValueError(f"lesions with more than one dx: {list(per_lesion[per_lesion > 1].index)[:10]}")
    return df.sort_values("image_id", kind="stable").reset_index(drop=True)[["image_id", "lesion_id", "label"]]


def verify_release(df: pd.DataFrame) -> None:
    """Assert the de-duplicated metadata is exactly the published HAM10000 release."""
    counts = class_counts(df["label"].to_numpy())
    problems = []
    if len(df) != HAM10000_N_IMAGES:
        problems.append(f"{len(df)} unique images, expected {HAM10000_N_IMAGES}")
    if counts != HAM10000_CLASS_COUNTS:
        problems.append(f"class counts {counts} != published {HAM10000_CLASS_COUNTS}")
    if problems:
        raise ValueError("this is not the full HAM10000 release: " + "; ".join(problems))


def load_pool(data_dir: str | Path = DATA_DIR, require_images: bool = True, strict: bool = True) -> Pool:
    """Metadata (+ images) from `data_dir` → Pool. Fails loudly on anything short of the full release.

    require_images=False never touches the image zips (split/fingerprint only — e.g. on a
    laptop without them, or while they are still downloading); the Pool then carries a
    NoImageStore that fails loudly if anything tries to read a picture.
    strict=False skips the published-count assertion (tests only).
    """
    data_dir = Path(data_dir)
    df = load_metadata(data_dir / METADATA_CSV)
    if strict:
        verify_release(df)
    images: ImageStore = FileImageStore(data_dir) if require_images else NoImageStore(data_dir)
    if require_images:
        absent = [i for i in df["image_id"] if i not in images]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} of {len(df)} images missing under {data_dir} (e.g. {absent[:3]}); "
                f"expected {list(IMAGE_ZIPS)} or the extracted *.jpg files — see HANDOVER.md"
            )
    return Pool(
        image_ids=df["image_id"].tolist(),
        labels=df["label"].to_numpy(dtype=np.int64),
        lesion_ids=df["lesion_id"].to_numpy(),
        images=images,
    )


# --------------------------------------------------------------------------- #
# Split — stratified by class, grouped by lesion (pure function → reproducible)
# --------------------------------------------------------------------------- #
def make_split(
    labels: Sequence[int] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    seed: int = SEED,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> dict[str, np.ndarray]:
    """Row indices for train/val/test such that every group (lesion) lands in exactly one partition.

    Groups are split 70/15/15 stratified on their (single) class with two seeded
    `train_test_split` calls; images follow their lesion, so image-level fractions are
    approximate. Returns sorted index arrays keyed by partition name.
    """
    labels, groups = np.asarray(labels), np.asarray(groups)
    if labels.ndim != 1 or len(labels) == 0 or labels.shape != groups.shape:
        raise ValueError("labels and groups must be non-empty 1-D arrays of equal length")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {fractions}")
    train_frac, val_frac, test_frac = fractions

    uniq, inverse = np.unique(groups, return_inverse=True)  # sorted → independent of row order
    group_label = np.empty(len(uniq), dtype=labels.dtype)
    group_label[inverse] = labels
    if not np.array_equal(group_label[inverse], labels):
        raise ValueError("a group has more than one label; grouped stratification is undefined")

    gidx = np.arange(len(uniq))
    g_train, g_rest = train_test_split(gidx, test_size=val_frac + test_frac, stratify=group_label, random_state=seed)
    g_val, g_test = train_test_split(
        g_rest, test_size=test_frac / (val_frac + test_frac), stratify=group_label[g_rest], random_state=seed
    )
    code = np.empty(len(uniq), dtype=np.int8)
    code[g_train], code[g_val], code[g_test] = 0, 1, 2
    row_code = code[inverse]
    return {part: np.flatnonzero(row_code == k) for k, part in enumerate(PARTITIONS)}


def class_counts(labels: np.ndarray, indices: np.ndarray | None = None) -> dict[str, int]:
    """Per-class counts (canonical order) for the rows in `indices`."""
    labels = np.asarray(labels)
    subset = labels if indices is None else labels[indices]
    counts = np.bincount(subset, minlength=NUM_CLASSES)
    return {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)}


def split_fingerprint(split: dict[str, np.ndarray], image_ids: Sequence[str]) -> str:
    """Short sha256 over the image ids of each partition — identical splits ⇔ identical fingerprints."""
    h = hashlib.sha256()
    for part in PARTITIONS:
        h.update(part.encode())
        h.update("\n".join(str(image_ids[i]) for i in split[part]).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def print_split_report(pool: Pool, split: dict[str, np.ndarray]) -> None:
    """Per-class image counts (and lesion counts) for each partition (requirements: model-training 3)."""
    per_part = {part: class_counts(pool.labels, idx) for part, idx in split.items()}
    total = class_counts(pool.labels)
    header = f"{'class':<8}" + "".join(f"{p:>8}" for p in PARTITIONS) + f"{'total':>8}"
    print(header)
    print("-" * len(header))
    for name in CLASS_NAMES:
        print(f"{name:<8}" + "".join(f"{per_part[p][name]:>8}" for p in PARTITIONS) + f"{total[name]:>8}")
    print("-" * len(header))
    print(f"{'images':<8}" + "".join(f"{len(split[p]):>8}" for p in PARTITIONS) + f"{len(pool):>8}")
    lesions = {p: len(np.unique(pool.lesion_ids[idx])) for p, idx in split.items()}
    print(f"{'lesions':<8}" + "".join(f"{lesions[p]:>8}" for p in PARTITIONS) + f"{len(np.unique(pool.lesion_ids)):>8}")
    print(f"seed={SEED}  fractions={SPLIT_FRACTIONS}  grouped_by=lesion_id  fingerprint={split_fingerprint(split, pool.image_ids)}")


# --------------------------------------------------------------------------- #
# Preprocessing + DataLoaders
# --------------------------------------------------------------------------- #
class RandomRotate90:
    """Rotate by a random multiple of 90° (exact, no interpolation) — dermoscopy is orientation-free."""

    _OPS = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)

    def __call__(self, img: Image.Image) -> Image.Image:
        op = self._OPS[int(torch.randint(0, 4, (1,)))]
        return img if op is None else img.transpose(op)


def get_transform(train: bool = False) -> transforms.Compose:
    """resize 224 → [train aug] → tensor → ImageNet normalize.

    train=False is THE eval transform: identical for val, test and the server.
    train=True adds random horizontal/vertical flips and random 90° rotations.
    """
    steps: list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), RandomRotate90()]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class HamDataset(Dataset):
    """Torch view over a subset (index array) of the Pool."""

    def __init__(self, pool: Pool, indices: np.ndarray, transform: transforms.Compose):
        self.pool = pool
        self.indices = np.asarray(indices)
        self.transform = transform
        self.labels = pool.labels[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = int(self.indices[i])
        image = self.pool.images.open(self.pool.image_ids[row])
        return self.transform(image), int(self.pool.labels[row])


def _loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_dataloaders(
    pool: Pool,
    split: dict[str, np.ndarray],
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    seed: int = SEED,
) -> dict[str, DataLoader]:
    """train (augmented, shuffled, seeded) / val / test (eval transform) loaders over the canonical split."""
    return {
        part: _loader(
            HamDataset(pool, split[part], get_transform(train=(part == "train"))),
            batch_size, num_workers, shuffle=(part == "train"), seed=seed,
        )
        for part in PARTITIONS
    }


def build_full_loader(pool: Pool, batch_size: int = BATCH_SIZE, num_workers: int = 0, seed: int = SEED) -> DataLoader:
    """All images, augmented + shuffled — for the 100 % retrain (train.py --mode full) only."""
    return _loader(HamDataset(pool, np.arange(len(pool)), get_transform(train=True)), batch_size, num_workers, True, seed)


def load_split_data(data_dir: str | Path = DATA_DIR, seed: int = SEED, require_images: bool = True) -> tuple[Pool, dict[str, np.ndarray]]:
    """Convenience: pool + canonical split, with the per-class report printed."""
    pool = load_pool(data_dir, require_images=require_images)
    split = make_split(pool.labels, pool.lesion_ids, seed=seed)
    print(f"data_dir={data_dir}  images={len(pool)}  lesions={len(np.unique(pool.lesion_ids))}  images_verified={require_images}")
    print_split_report(pool, split)
    return pool, split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and report the canonical HAM10000 split.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="folder with the metadata CSV + image zips (default: %(default)s)")
    parser.add_argument("--check-images", action="store_true", help="also verify every image file is present")
    args = parser.parse_args()
    load_split_data(args.data_dir, require_images=args.check_images)


if __name__ == "__main__":
    main()
