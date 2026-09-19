"""Unit tests for data.py — release loading, de-dup, grouped split, seeding, label map, preprocessing."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import data
from data import (
    CLASS_NAMES,
    HAM10000_CLASS_COUNTS,
    HAM10000_N_IMAGES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_MAP,
    NUM_CLASSES,
    PARTITIONS,
    SPLIT_FRACTIONS,
    canonical_dx,
    class_counts,
    get_transform,
    make_split,
    split_fingerprint,
)

REAL_METADATA = Path(data.DATA_DIR) / data.METADATA_CSV


@pytest.fixture(scope="module")
def ham_like():
    """Synthetic (labels, lesion groups) with the published HAM10000 class counts and ~1.34 images/lesion."""
    rng = np.random.default_rng(0)
    labels, groups = [], []
    lesion = 0
    for i, n in enumerate(HAM10000_CLASS_COUNTS.values()):
        left = n
        while left > 0:
            size = int(min(left, rng.choice([1, 1, 1, 2, 2, 3])))
            labels += [i] * size
            groups += [f"HAM_{lesion:07d}"] * size
            lesion += 1
            left -= size
    order = rng.permutation(len(labels))
    return np.asarray(labels)[order], np.asarray(groups)[order]


# ---------------------------------------------------------------- label map --
def test_label_map_matches_design_doc():
    assert CLASS_NAMES == ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    assert LABEL_MAP == {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
    assert NUM_CLASSES == 7 and data.SEED == 42 and SPLIT_FRACTIONS == (0.70, 0.15, 0.15)
    assert sum(HAM10000_CLASS_COUNTS.values()) == HAM10000_N_IMAGES == 10015


def test_dx_aliases_map_to_canonical_ids():
    assert canonical_dx("akiec") == 0 and canonical_dx("Actinic_keratoses") == 0
    assert canonical_dx(" nv ") == 5 and canonical_dx("melanocytic_Nevi") == 5
    assert canonical_dx("vascular lesions") == 6 and canonical_dx("df") == 3
    with pytest.raises(ValueError, match="unknown dx"):
        canonical_dx("seborrheic_keratosis")


# ------------------------------------------------------------------- split --
def test_same_seed_identical_split(ham_like):
    labels, groups = ham_like
    a, b = make_split(labels, groups, seed=42), make_split(labels, groups, seed=42)
    for part in PARTITIONS:
        assert np.array_equal(a[part], b[part]), part
    ids = [f"ISIC_{i:07d}" for i in range(len(labels))]
    assert split_fingerprint(a, ids) == split_fingerprint(b, ids)


def test_different_seed_gives_different_split(ham_like):
    labels, groups = ham_like
    a, b = make_split(labels, groups, seed=42), make_split(labels, groups, seed=43)
    assert not np.array_equal(a["test"], b["test"])


def test_split_is_independent_of_row_order(ham_like):
    """Same content in a different row order → same partitions by image id (the fingerprint contract)."""
    labels, groups = ham_like
    ids = np.array([f"ISIC_{i:07d}" for i in range(len(labels))])
    perm = np.random.default_rng(3).permutation(len(labels))
    a = make_split(labels, groups)
    b = make_split(labels[perm], groups[perm])
    for part in PARTITIONS:
        assert set(ids[a[part]]) == set(ids[perm][b[part]]), part


def test_partitions_disjoint_and_complete(ham_like):
    labels, groups = ham_like
    split = make_split(labels, groups)
    parts = [set(split[p].tolist()) for p in PARTITIONS]
    assert parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2]) and parts[1].isdisjoint(parts[2])
    assert parts[0] | parts[1] | parts[2] == set(range(len(labels)))
    for part in PARTITIONS:
        assert np.all(np.diff(split[part]) > 0), f"{part} indices not sorted"


def test_every_lesion_in_exactly_one_partition(ham_like):
    labels, groups = ham_like
    split = make_split(labels, groups)
    seen = {}
    for part in PARTITIONS:
        for g in np.unique(groups[split[part]]):
            assert g not in seen, f"lesion {g} in both {seen.get(g)} and {part}"
            seen[g] = part
    assert len(seen) == len(np.unique(groups))


def test_class_counts_sum_correctly(ham_like):
    labels, groups = ham_like
    split = make_split(labels, groups)
    total = class_counts(labels)
    assert total == HAM10000_CLASS_COUNTS
    for name in CLASS_NAMES:
        assert sum(class_counts(labels, split[p])[name] for p in PARTITIONS) == total[name], name


def test_split_is_stratified_70_15_15_at_lesion_level(ham_like):
    labels, groups = ham_like
    split = make_split(labels, groups)
    n_lesions = len(np.unique(groups))
    for part, frac in zip(PARTITIONS, SPLIT_FRACTIONS):
        lesions_in_part = len(np.unique(groups[split[part]]))
        assert abs(lesions_in_part - frac * n_lesions) <= 2, part            # exact at lesion level
        assert abs(len(split[part]) - frac * len(labels)) <= 0.03 * len(labels), part  # approximate at image level
        counts = class_counts(labels, split[part])
        for name, total in HAM10000_CLASS_COUNTS.items():
            assert abs(counts[name] - frac * total) <= max(6, 0.06 * total), (part, name)


def test_split_rejects_bad_input():
    with pytest.raises(ValueError):
        make_split([], [])
    with pytest.raises(ValueError):
        make_split([0, 1, 0, 1], ["a", "b", "c", "d"], fractions=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="more than one label"):
        make_split(np.tile(np.arange(7), 4), ["same"] * 28)


def test_fingerprint_depends_on_image_ids(ham_like):
    labels, groups = ham_like
    split = make_split(labels, groups)
    ids_a = [f"ISIC_{i:07d}" for i in range(len(labels))]
    ids_b = [f"ISIC_{i + 1:07d}" for i in range(len(labels))]
    assert split_fingerprint(split, ids_a) != split_fingerprint(split, ids_b)


# ----------------------------------------------------------- metadata loading --
def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    pd.DataFrame(rows, columns=["lesion_id", "image_id", "dx"]).assign(dx_type="histo").to_csv(path, index=False)
    return path


def test_load_metadata_dedupes_identical_rows_and_sorts(tmp_path):
    csv = _write_csv(tmp_path / "m.csv", [
        ("HAM_1", "ISIC_2", "nv"), ("HAM_1", "ISIC_1", "nv"), ("HAM_1", "ISIC_2", "nv"),  # exact dup
        ("HAM_2", "ISIC_3", "Melanoma"),
    ])
    df = data.load_metadata(csv)
    assert df["image_id"].tolist() == ["ISIC_1", "ISIC_2", "ISIC_3"]
    assert df["label"].tolist() == [5, 5, 4]
    assert list(df.columns) == ["image_id", "lesion_id", "label"]


def test_load_metadata_rejects_conflicting_duplicates_and_bad_lesions(tmp_path):
    with pytest.raises(ValueError, match="disagree"):
        data.load_metadata(_write_csv(tmp_path / "a.csv", [("HAM_1", "ISIC_1", "nv"), ("HAM_1", "ISIC_1", "mel")]))
    with pytest.raises(ValueError, match="more than one dx"):
        data.load_metadata(_write_csv(tmp_path / "b.csv", [("HAM_1", "ISIC_1", "nv"), ("HAM_1", "ISIC_2", "mel")]))
    with pytest.raises(FileNotFoundError):
        data.load_metadata(tmp_path / "missing.csv")


def test_verify_release_rejects_subsets(tmp_path):
    df = data.load_metadata(_write_csv(tmp_path / "s.csv", [(f"HAM_{i}", f"ISIC_{i}", CLASS_NAMES[i % 7]) for i in range(700)]))
    with pytest.raises(ValueError, match="not the full HAM10000"):
        data.verify_release(df)


# --------------------------------------------------------------- image store --
def _jpeg_bytes(seed: int) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, size=(45, 60, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_file_image_store_reads_zips_and_loose_files(tmp_path):
    with zipfile.ZipFile(tmp_path / "HAM10000_images_part_1.zip", "w") as zf:
        zf.writestr("ISIC_0000001.jpg", _jpeg_bytes(1))
        zf.writestr("__MACOSX/._ISIC_0000001.jpg", b"junk")
    (tmp_path / "loose").mkdir()
    (tmp_path / "loose" / "ISIC_0000002.jpg").write_bytes(_jpeg_bytes(2))
    store = data.FileImageStore(tmp_path)
    assert len(store) == 2 and "ISIC_0000001" in store and "ISIC_0000003" not in store
    for image_id in ("ISIC_0000001", "ISIC_0000002"):
        img = store.open(image_id)
        assert img.mode == "RGB" and img.size == (60, 45)
    with pytest.raises(KeyError):
        store.open("ISIC_0000003")
    import pickle  # DataLoader workers pickle the store; zip handles must not travel with it
    assert pickle.loads(pickle.dumps(store)).open("ISIC_0000001").size == (60, 45)


def test_load_pool_requires_every_image(tmp_path):
    _write_csv(tmp_path / data.METADATA_CSV, [("HAM_1", "ISIC_0000001", "nv"), ("HAM_2", "ISIC_0000002", "mel")])
    with zipfile.ZipFile(tmp_path / "HAM10000_images_part_1.zip", "w") as zf:
        zf.writestr("ISIC_0000001.jpg", _jpeg_bytes(1))
    with pytest.raises(FileNotFoundError, match="1 of 2 images missing"):
        data.load_pool(tmp_path, strict=False)
    pool = data.load_pool(tmp_path, strict=False, require_images=False)
    assert pool.image_ids == ["ISIC_0000001", "ISIC_0000002"] and pool.labels.tolist() == [5, 4]
    with pytest.raises(ValueError, match="not the full HAM10000"):
        data.load_pool(tmp_path, require_images=False)  # strict by default


# ----------------------------------------------------------- preprocessing --
def _random_image(seed: int = 1, size=(600, 450)) -> Image.Image:
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8))


def test_eval_transform_output_shape_and_range():
    x = get_transform(train=False)(_random_image())
    assert isinstance(x, torch.Tensor) and x.shape == (3, IMAGE_SIZE, IMAGE_SIZE) and x.dtype == torch.float32
    for c in range(3):
        lo, hi = (0.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c], (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert x[c].min() >= lo - 1e-4 and x[c].max() <= hi + 1e-4


def test_eval_transform_is_imagenet_normalization_and_deterministic():
    arr = np.zeros((450, 600, 3), dtype=np.uint8)
    for c in range(3):
        arr[..., c] = round(IMAGENET_MEAN[c] * 255)
    x = get_transform()(Image.fromarray(arr))
    assert torch.allclose(x, torch.zeros_like(x), atol=0.02)
    img = _random_image(5)
    assert torch.equal(get_transform()(img), get_transform()(img))


def test_train_transform_augments_but_keeps_pixels():
    """Flips/90° rotations: same shape, same multiset of pixel values, but not always the same tensor."""
    img = _random_image(9)
    eval_x = get_transform(train=False)(img)
    torch.manual_seed(0)
    outs = [get_transform(train=True)(img) for _ in range(8)]
    assert all(o.shape == (3, IMAGE_SIZE, IMAGE_SIZE) for o in outs)
    assert any(not torch.equal(o, eval_x) for o in outs)
    for o in outs:  # a dihedral transform permutes pixels — the sorted values are identical
        assert torch.equal(o.flatten().sort().values, eval_x.flatten().sort().values)


def test_random_rotate90_is_exact():
    img = Image.fromarray(np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3))
    torch.manual_seed(1)
    seen = {np.asarray(data.RandomRotate90()(img)).tobytes() for _ in range(32)}
    assert len(seen) == 4
    assert np.asarray(img.transpose(Image.Transpose.ROTATE_90)).tobytes() in seen


# ------------------------------------------- real release (skipped if absent) --
@pytest.mark.skipif(not REAL_METADATA.is_file(), reason=f"{REAL_METADATA} not downloaded")
def test_real_metadata_is_the_published_release_and_split_is_stable():
    pool = data.load_pool(require_images=False)  # metadata only; images live on the training box
    assert len(pool) == HAM10000_N_IMAGES
    assert class_counts(pool.labels) == HAM10000_CLASS_COUNTS
    assert len(np.unique(pool.lesion_ids)) == 7470
    split = make_split(pool.labels, pool.lesion_ids)
    assert sum(len(split[p]) for p in PARTITIONS) == HAM10000_N_IMAGES
    for part in PARTITIONS:  # no lesion crosses partitions
        others = np.concatenate([split[q] for q in PARTITIONS if q != part])
        assert not set(pool.lesion_ids[split[part]]) & set(pool.lesion_ids[others])
    # the canonical fingerprint: printed by data.py, stored in every split-run checkpoint, cited in HANDOVER.md
    assert split_fingerprint(split, pool.image_ids) == "4b4cc59260945104"


def test_tiny_pool_batches(tiny_pool):
    split = make_split(tiny_pool.labels, tiny_pool.lesion_ids)
    loaders = data.build_dataloaders(tiny_pool, split, batch_size=4)
    x, y = next(iter(loaders["test"]))
    assert x.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE) and y.shape == (4,) and y.dtype == torch.int64
    full = data.build_full_loader(tiny_pool, batch_size=16)
    assert sum(len(b[1]) for b in full) == len(tiny_pool)
