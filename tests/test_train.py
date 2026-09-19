"""Unit tests for train.py — class weights, model head, checkpoint contract, loop smoke test."""

import csv

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models import efficientnet_b0

import data
import train
from data import CLASS_NAMES, LABEL_MAP, NUM_CLASSES


# ---------------------------------------------------------- class weights --
def test_class_weights_inverse_frequency():
    counts = {"akiec": 327, "bcc": 514, "bkl": 1099, "df": 115, "mel": 1113, "nv": 6705, "vasc": 142}
    labels = np.concatenate([np.full(n, i) for i, n in enumerate(counts.values())])
    w = train.compute_class_weights(labels)
    assert w.shape == (NUM_CLASSES,) and w.dtype == torch.float32
    n = labels.size
    for i, c in enumerate(counts.values()):
        assert w[i].item() == pytest.approx(n / (NUM_CLASSES * c), rel=1e-5)
    assert w[CLASS_NAMES.index("df")] > w[CLASS_NAMES.index("akiec")] > w[CLASS_NAMES.index("nv")]
    # Σ w_c·n_c == N: the weighted loss keeps the scale of the unweighted one
    assert float((w * torch.tensor(list(counts.values()), dtype=torch.float32)).sum()) == pytest.approx(n, rel=1e-5)


def test_class_weights_balanced_are_all_one():
    labels = np.repeat(np.arange(NUM_CLASSES), 70)
    assert torch.allclose(train.compute_class_weights(labels), torch.ones(NUM_CLASSES))


def test_class_weights_fail_loudly_on_missing_class():
    with pytest.raises(ValueError, match="vasc"):
        train.compute_class_weights(np.arange(NUM_CLASSES - 1))


# ------------------------------------------------------------------ model --
def test_build_model_has_7_class_head():
    model = train.build_model(pretrained=False).eval()
    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].out_features == NUM_CLASSES
    with torch.no_grad():
        out = model(torch.randn(2, 3, data.IMAGE_SIZE, data.IMAGE_SIZE))
    assert out.shape == (2, NUM_CLASSES)


def test_get_device_override():
    assert train.get_device("cpu").type == "cpu"
    assert train.get_device().type in {"cuda", "mps", "cpu"}


# ------------------------------------------------------------ checkpoint --
def test_checkpoint_loads_without_training_code(tmp_path):
    model = train.build_model(pretrained=False).eval()
    path = tmp_path / "model_best.pth"
    train.save_checkpoint(path, model, epoch=3, val_macro_f1=0.5, seed=42)

    # weights_only=True → only tensors/primitives inside; torchvision alone rebuilds it
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert ckpt["arch"] == "efficientnet_b0"
    assert ckpt["label_map"] == LABEL_MAP == {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
    assert ckpt["class_names"] == list(CLASS_NAMES)
    assert ckpt["class_labels"]["akiec"] == "Actinic keratosis"
    assert ckpt["image_size"] == data.IMAGE_SIZE
    assert ckpt["normalization"] == {"mean": list(data.IMAGENET_MEAN), "std": list(data.IMAGENET_STD)}
    assert ckpt["epoch"] == 3 and ckpt["val_macro_f1"] == 0.5 and ckpt["seed"] == 42

    rebuilt = efficientnet_b0(weights=None)
    rebuilt.classifier[1] = nn.Linear(rebuilt.classifier[1].in_features, ckpt["num_classes"])
    rebuilt.load_state_dict(ckpt["state_dict"])
    rebuilt.eval()
    x = torch.randn(1, 3, data.IMAGE_SIZE, data.IMAGE_SIZE)
    with torch.no_grad():
        assert torch.allclose(model(x), rebuilt(x))
    assert not path.with_suffix(".pth.tmp").exists()  # atomic write cleaned up


# -------------------------------------------------- training loop (smoke) --
def test_train_loop_saves_best_and_logs(tmp_path, tiny_pool):
    split = data.make_split(tiny_pool.labels, tiny_pool.lesion_ids)
    out = tmp_path / "model_best.pth"
    best = train.train(
        tiny_pool, split, out, epochs=2, patience=5, batch_size=8,
        device=torch.device("cpu"), pretrained=False, dataset_name="tiny",
    )
    assert out.exists() and best["epoch"] >= 1
    ckpt = torch.load(out, map_location="cpu", weights_only=True)
    assert ckpt["dataset"] == "tiny" and ckpt["trained_on"] == "train"
    assert ckpt["n_train_images"] == len(split["train"])
    assert ckpt["split_fingerprint"] == data.split_fingerprint(split, tiny_pool.image_ids)
    assert ckpt["class_weights"] == pytest.approx(train.compute_class_weights(tiny_pool.labels[split["train"]]).tolist())
    assert ckpt["val_macro_f1"] == pytest.approx(best["val_macro_f1"])

    with open(tmp_path / "train_log.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert all(set(r) == {"epoch", "train_loss", "val_loss", "val_macro_f1", "best_val_macro_f1", "seconds"} for r in rows)


def test_early_stopping_triggers(tmp_path, tiny_pool):
    split = data.make_split(tiny_pool.labels, tiny_pool.lesion_ids)
    # lr=0 → the model never changes → val macro-F1 never improves after epoch 1 → stop at 1 + patience
    train.train(
        tiny_pool, split, tmp_path / "m.pth", epochs=10, patience=2, lr=0.0, batch_size=8,
        device=torch.device("cpu"), pretrained=False,
    )
    with open(tmp_path / "train_log.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3


def test_predict_fails_loudly_on_empty_loader():
    model = train.build_model(pretrained=False)
    empty = torch.utils.data.DataLoader([], batch_size=1)
    with pytest.raises(RuntimeError):
        train.predict(model, empty, torch.device("cpu"))


# ------------------------------------------------- full-data retrain (--mode full) --
def test_train_full_saves_demo_model_with_provenance(tmp_path, tiny_pool):
    split = data.make_split(tiny_pool.labels, tiny_pool.lesion_ids)
    best_path = tmp_path / "model_best.pth"
    train.train(tiny_pool, split, best_path, epochs=2, patience=5, batch_size=8, device=torch.device("cpu"), pretrained=False)
    epochs, provenance = train.epochs_from_checkpoint(best_path)
    assert epochs in (1, 2) and provenance["split_fingerprint"] == data.split_fingerprint(split, tiny_pool.image_ids)

    final_path = tmp_path / "model_final.pth"
    train.train_full(tiny_pool, final_path, epochs=epochs, derived_from=provenance, batch_size=8,
                     device=torch.device("cpu"), pretrained=False, dataset_name="tiny")
    ckpt = torch.load(final_path, map_location="cpu", weights_only=True)
    assert ckpt["trained_on"] == "all" and ckpt["split_fingerprint"] is None
    assert ckpt["epoch"] == epochs and ckpt["n_train_images"] == len(tiny_pool)
    assert ckpt["derived_from"]["checkpoint"] == str(best_path)
    assert ckpt["class_weights"] == pytest.approx([1.0] * NUM_CLASSES)  # tiny pool is balanced
    with open(tmp_path / "train_log_full.csv") as f:
        assert [r["epoch"] for r in csv.DictReader(f)] == [str(e) for e in range(1, epochs + 1)]

    with pytest.raises(ValueError, match="not a split-run checkpoint"):
        train.epochs_from_checkpoint(final_path)  # a full model cannot seed another full run
    with pytest.raises(ValueError, match="epochs"):
        train.train_full(tiny_pool, final_path, epochs=0, device=torch.device("cpu"), pretrained=False)
