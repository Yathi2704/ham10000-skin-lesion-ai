"""Unit tests for data.py — canonical split, seeding, label map, preprocessing."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

import data
from data import (
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_MAP,
    NUM_CLASSES,
    PARTITIONS,
    SPLIT_FRACTIONS,
    class_counts,
    get_transform,
    make_split,
    split_fingerprint,
)

# Synthetic labels with HAM10000-like imbalance (akiec..vasc), used so the split
# tests never need the real dataset.
HAM_LIKE_COUNTS = {"akiec": 327, "bcc": 514, "bkl": 1099, "df": 115, "mel": 1113, "nv": 6705, "vasc": 142}


@pytest.fixture(scope="module")
def labels() -> np.ndarray:
    rng = np.random.default_rng(0)
    arr = np.concatenate([np.full(n, i) for i, (_, n) in enumerate(HAM_LIKE_COUNTS.items())])
    rng.shuffle(arr)
    return arr


# ---------------------------------------------------------------- label map --
def test_label_map_matches_design_doc():
    assert CLASS_NAMES == ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    assert LABEL_MAP == {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
    assert NUM_CLASSES == 7
    assert data.SEED == 42
    assert SPLIT_FRACTIONS == (0.70, 0.15, 0.15)


# ------------------------------------------------------------------- split --
def test_same_seed_identical_split(labels):
    a = make_split(labels, seed=42)
    b = make_split(labels, seed=42)
    for part in PARTITIONS:
        assert np.array_equal(a[part], b[part]), part
    assert split_fingerprint(a) == split_fingerprint(b)


def test_different_seed_gives_different_split(labels):
    a = make_split(labels, seed=42)
    b = make_split(labels, seed=43)
    assert not np.array_equal(a["test"], b["test"])
    assert split_fingerprint(a) != split_fingerprint(b)


def test_partitions_disjoint_and_complete(labels):
    split = make_split(labels)
    parts = [set(split[p].tolist()) for p in PARTITIONS]
    assert parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2]) and parts[1].isdisjoint(parts[2])
    assert parts[0] | parts[1] | parts[2] == set(range(len(labels)))
    assert sum(len(split[p]) for p in PARTITIONS) == len(labels)


def test_class_counts_sum_correctly(labels):
    split = make_split(labels)
    total = class_counts(labels)
    assert total == HAM_LIKE_COUNTS
    for name in CLASS_NAMES:
        per_part = sum(class_counts(labels, split[p])[name] for p in PARTITIONS)
        assert per_part == total[name], name


def test_split_is_stratified_70_15_15(labels):
    split = make_split(labels)
    n = len(labels)
    for part, frac in zip(PARTITIONS, SPLIT_FRACTIONS):
        assert abs(len(split[part]) - frac * n) <= 1, part
        counts = class_counts(labels, split[part])
        for name, total in HAM_LIKE_COUNTS.items():
            # each class is present in every partition in (roughly) the global proportion
            assert abs(counts[name] - frac * total) <= 2, (part, name)


def test_split_rejects_bad_input():
    with pytest.raises(ValueError):
        make_split([])
    with pytest.raises(ValueError):
        make_split([0, 1, 0, 1], fractions=(0.5, 0.5, 0.5))


def test_fingerprint_uses_image_ids_when_given(labels):
    split = make_split(labels)
    ids_a = [f"ISIC_{i:07d}" for i in range(len(labels))]
    ids_b = [f"ISIC_{i + 1:07d}" for i in range(len(labels))]
    assert split_fingerprint(split, ids_a) == split_fingerprint(split, ids_a)
    assert split_fingerprint(split, ids_a) != split_fingerprint(split, ids_b)


# ----------------------------------------------------------- preprocessing --
def test_transform_output_shape_and_range():
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, size=(450, 600, 3), dtype=np.uint8))  # HAM10000 native 600x450
    x = get_transform()(img)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert x.dtype == torch.float32
    for c in range(3):
        lo = (0.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        hi = (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert x[c].min() >= lo - 1e-4 and x[c].max() <= hi + 1e-4


def test_transform_is_imagenet_normalization():
    # a constant image equal to the ImageNet mean must map to ~0 after normalization
    arr = np.zeros((450, 600, 3), dtype=np.uint8)
    for c in range(3):
        arr[..., c] = round(IMAGENET_MEAN[c] * 255)
    x = get_transform()(Image.fromarray(arr))
    assert torch.allclose(x, torch.zeros_like(x), atol=0.02)


# ------------------------------------------- real dataset (skipped if absent) --
@pytest.fixture(scope="module")
def real_pool():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return data.load_pool()
    except Exception as exc:  # dataset not in the local HF cache → skip, never fail silently
        pytest.skip(f"{data.DATASET_NAME} not available offline: {exc}")


def test_real_pool_split_counts_and_batches(real_pool):
    labels = np.asarray(real_pool["label"])
    split = make_split(labels)
    assert set(np.unique(labels).tolist()) == set(range(NUM_CLASSES))
    total = class_counts(labels)
    for name in CLASS_NAMES:
        assert sum(class_counts(labels, split[p])[name] for p in PARTITIONS) == total[name]
    assert len(set(real_pool["image_id"])) == len(labels)  # ids unique → split reproducible by content

    loaders = data.build_dataloaders(real_pool, split, batch_size=4)
    x, y = next(iter(loaders["test"]))
    assert x.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE) and y.shape == (4,)
    assert y.dtype == torch.int64 and int(y.max()) < NUM_CLASSES
