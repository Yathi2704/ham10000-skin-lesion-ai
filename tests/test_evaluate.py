"""Unit tests for evaluate.py — live metrics, akiec sens/spec, artifacts, loud failures."""

import csv

import numpy as np
import pytest
import torch
from sklearn.metrics import classification_report

import data
import evaluate
import train
from data import CLASS_NAMES, CLASS_TO_INDEX, NUM_CLASSES


# ------------------------------------------------------ akiec sens / spec --
def test_akiec_sensitivity_specificity_known_confusion():
    ak, other = CLASS_TO_INDEX["akiec"], CLASS_TO_INDEX["nv"]
    #            TP ×3          FN ×1        TN ×4                    FP ×2
    y_true = [ak, ak, ak,       ak,          other, other, other, other, other, other]
    y_pred = [ak, ak, ak,       other,       other, other, other, other, ak, ak]
    r = evaluate.akiec_sensitivity_specificity(np.array(y_true), np.array(y_pred))
    assert (r["tp"], r["fn"], r["tn"], r["fp"]) == (3, 1, 4, 2)
    assert r["sensitivity"] == pytest.approx(3 / 4)
    assert r["specificity"] == pytest.approx(4 / 6)


def test_akiec_sensitivity_nan_when_no_positives():
    nv = CLASS_TO_INDEX["nv"]
    r = evaluate.akiec_sensitivity_specificity(np.array([nv, nv]), np.array([nv, nv]))
    assert np.isnan(r["sensitivity"]) and r["specificity"] == 1.0


# ----------------------------------------------------------- metric rows --
@pytest.fixture
def synthetic_preds():
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, NUM_CLASSES, size=500)
    y_pred = np.where(rng.random(500) < 0.6, y_true, rng.integers(0, NUM_CLASSES, size=500))
    return y_true, y_pred


def test_compute_metrics_matches_sklearn(synthetic_preds):
    y_true, y_pred = synthetic_preds
    rows = evaluate.compute_metrics(y_true, y_pred)
    by = {(r["metric"], r["class"]): r["value"] for r in rows}
    ref = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    for name in CLASS_NAMES:
        assert by[("precision", name)] == pytest.approx(ref[name]["precision"])
        assert by[("recall", name)] == pytest.approx(ref[name]["recall"])
        assert by[("f1", name)] == pytest.approx(ref[name]["f1-score"])
        assert by[("support", name)] == ref[name]["support"]
    for avg in ("macro", "weighted"):
        assert by[("f1", f"{avg}_avg")] == pytest.approx(ref[f"{avg} avg"]["f1-score"])
        assert by[("precision", f"{avg}_avg")] == pytest.approx(ref[f"{avg} avg"]["precision"])
        assert by[("recall", f"{avg}_avg")] == pytest.approx(ref[f"{avg} avg"]["recall"])
    assert by[("accuracy", "all")] == pytest.approx(ref["accuracy"])
    assert by[("sensitivity", "akiec")] == pytest.approx(ref["akiec"]["recall"])  # sensitivity == recall
    assert 0.0 <= by[("specificity", "akiec")] <= 1.0
    assert by[("n_test", "all")] == 500


def test_metrics_change_with_predictions(synthetic_preds):
    """Guards the 'no hardcoded metrics' rule: different predictions → different numbers."""
    y_true, y_pred = synthetic_preds
    a = {(r["metric"], r["class"]): r["value"] for r in evaluate.compute_metrics(y_true, y_pred)}
    b = {(r["metric"], r["class"]): r["value"] for r in evaluate.compute_metrics(y_true, y_true)}
    assert a[("accuracy", "all")] < 1.0 == b[("accuracy", "all")]
    assert b[("sensitivity", "akiec")] == 1.0 and b[("specificity", "akiec")] == 1.0


# -------------------------------------------------------------- artifacts --
def test_write_metrics_csv_and_confusion_png(tmp_path, synthetic_preds):
    y_true, y_pred = synthetic_preds
    rows = evaluate.compute_metrics(y_true, y_pred)
    csv_path = evaluate.write_metrics_csv(rows, tmp_path / "metrics.csv")
    png_path = evaluate.save_confusion_matrix(y_true, y_pred, tmp_path / "confusion_matrix.png")
    assert png_path.is_file() and png_path.stat().st_size > 1000
    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    with open(csv_path) as f:
        parsed = list(csv.DictReader(f))
    assert set(parsed[0]) == {"metric", "class", "value"}
    keys = {(r["metric"], r["class"]) for r in parsed}
    for name in CLASS_NAMES:
        assert {("precision", name), ("recall", name), ("f1", name), ("support", name)} <= keys
    assert {("f1", "weighted_avg"), ("f1", "macro_avg"), ("accuracy", "all"),
            ("sensitivity", "akiec"), ("specificity", "akiec")} <= keys


# ---------------------------------------------------- checkpoint loading --
def test_load_checkpoint_missing_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        evaluate.load_checkpoint(tmp_path / "nope.pth", torch.device("cpu"))


def test_load_checkpoint_corrupt_fails_loudly(tmp_path):
    bad = tmp_path / "model_best.pth"
    bad.write_bytes(b"definitely not a torch file")
    with pytest.raises(RuntimeError, match="corrupt"):
        evaluate.load_checkpoint(bad, torch.device("cpu"))


def test_load_checkpoint_wrong_label_map_fails_loudly(tmp_path):
    model = train.build_model(pretrained=False)
    path = tmp_path / "model_best.pth"
    train.save_checkpoint(path, model, class_names=["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"])
    with pytest.raises(RuntimeError, match="label map"):
        evaluate.load_checkpoint(path, torch.device("cpu"))


def test_load_checkpoint_roundtrip(tmp_path):
    model = train.build_model(pretrained=False).eval()
    path = tmp_path / "model_best.pth"
    train.save_checkpoint(path, model, epoch=1, val_macro_f1=0.1)
    loaded, meta = evaluate.load_checkpoint(path, torch.device("cpu"))
    assert meta["epoch"] == 1 and "state_dict" not in meta
    x = torch.randn(1, 3, data.IMAGE_SIZE, data.IMAGE_SIZE)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


# ------------------------------------------------------------ end to end --
def test_run_end_to_end_and_split_guard(tmp_path, tiny_pool):
    split = data.make_split(tiny_pool.labels, tiny_pool.lesion_ids)
    ckpt = tmp_path / "model_best.pth"
    train.train(tiny_pool, split, ckpt, epochs=1, batch_size=8, device=torch.device("cpu"), pretrained=False)

    rows = evaluate.run(ckpt, tiny_pool, split, out_dir=tmp_path, device=torch.device("cpu"), batch_size=8)
    assert (tmp_path / "metrics.csv").is_file() and (tmp_path / "confusion_matrix.png").is_file()
    by = {(r["metric"], r["class"]): r["value"] for r in rows}
    assert by[("n_test", "all")] == len(split["test"])
    assert by[("split_fingerprint", "meta")] == data.split_fingerprint(split, tiny_pool.image_ids)
    assert by[("checkpoint_file", "meta")] == "model_best.pth"

    # a different split (seed 43) must be refused — the test set could overlap training data
    other = data.make_split(tiny_pool.labels, tiny_pool.lesion_ids, seed=43)
    with pytest.raises(RuntimeError, match="split mismatch"):
        evaluate.run(ckpt, tiny_pool, other, out_dir=tmp_path, device=torch.device("cpu"), batch_size=8)


def test_run_refuses_the_full_data_demo_model(tmp_path, tiny_pool):
    split = data.make_split(tiny_pool.labels, tiny_pool.lesion_ids)
    final = tmp_path / "model_final.pth"
    train.train_full(tiny_pool, final, epochs=1, batch_size=8, device=torch.device("cpu"), pretrained=False)
    with pytest.raises(RuntimeError, match="refusing to evaluate"):
        evaluate.run(final, tiny_pool, split, out_dir=tmp_path, device=torch.device("cpu"), batch_size=8)
    assert not (tmp_path / "metrics.csv").exists()  # nothing written for a refused model


def test_run_tolerates_checkpoint_without_val_macro_f1(tmp_path, tiny_pool):
    """REVIEW.md Minor 2: a hand-rolled split checkpoint lacking val_macro_f1 must not TypeError in the report."""
    split = data.make_split(tiny_pool.labels, tiny_pool.lesion_ids)
    ckpt = tmp_path / "model_best.pth"
    train.save_checkpoint(ckpt, train.build_model(pretrained=False), trained_on="train", epoch=1,
                          split_fingerprint=data.split_fingerprint(split, tiny_pool.image_ids))
    rows = evaluate.run(ckpt, tiny_pool, split, out_dir=tmp_path, device=torch.device("cpu"), batch_size=8)
    by = {(r["metric"], r["class"]): r["value"] for r in rows}
    assert np.isnan(by[("checkpoint_val_macro_f1", "meta")])
    assert (tmp_path / "metrics.csv").is_file()
