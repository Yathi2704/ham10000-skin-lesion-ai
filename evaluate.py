"""Evaluate `model_best.pth` on the canonical held-out test set.

Every number here is computed live from the model's predictions on the test
partition of the seed-42 lesion-grouped split (requirements: model-training 6–8,
honesty-reproducibility 1–2). Nothing is hardcoded; nothing is cached.

Only a split-run checkpoint (`trained_on == "train"`) is accepted: the 100 %-data
demo model (`model_final.pth`) has seen the test images and is refused.

    python evaluate.py --checkpoint app/model_best.pth

Writes `metrics.csv` (long form: metric,class,value) and `confusion_matrix.png`
next to the checkpoint unless `--out-dir` is given.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless (Vast.ai / ssh)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn

import data
from data import BATCH_SIZE, CLASS_NAMES, CLASS_TO_INDEX, LABEL_MAP, NUM_CLASSES
from train import ARCH, build_model, get_device, predict

METRICS_CSV = "metrics.csv"
CONFUSION_PNG = "confusion_matrix.png"
AKIEC = CLASS_TO_INDEX["akiec"]


# --------------------------------------------------------------------------- #
# Checkpoint loading — fails loudly on anything unexpected
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model from `model_best.pth`; verify it speaks the canonical label map."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {path} — train first (python train.py) or scp it into app/")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"model checkpoint {path} is unreadable/corrupt: {exc}") from exc
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"model checkpoint {path} has no 'state_dict' — not a train.py checkpoint")
    if ckpt.get("arch") != ARCH:
        raise RuntimeError(f"checkpoint arch {ckpt.get('arch')!r} != expected {ARCH!r}")
    if ckpt.get("num_classes") != NUM_CLASSES or list(ckpt.get("class_names", [])) != list(CLASS_NAMES):
        raise RuntimeError(
            f"checkpoint label map {ckpt.get('class_names')} != canonical {list(CLASS_NAMES)} (design.md)"
        )
    if {int(k): v for k, v in ckpt.get("label_map", {}).items()} != LABEL_MAP:
        raise RuntimeError(f"checkpoint label_map {ckpt.get('label_map')} != canonical {LABEL_MAP}")

    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    try:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint {path} does not fit {ARCH}: {exc}") from exc
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta


# --------------------------------------------------------------------------- #
# Metrics (all from live y_true / y_pred)
# --------------------------------------------------------------------------- #
def akiec_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray, positive: int = AKIEC) -> dict[str, float]:
    """One-vs-rest sensitivity (recall) and specificity for the akiec class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pos_true, pos_pred = y_true == positive, y_pred == positive
    tp = int(np.sum(pos_true & pos_pred))
    fn = int(np.sum(pos_true & ~pos_pred))
    tn = int(np.sum(~pos_true & ~pos_pred))
    fp = int(np.sum(~pos_true & pos_pred))
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"sensitivity": sensitivity, "specificity": specificity, "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    """Long-form rows: per-class P/R/F1/support, macro + weighted averages, accuracy, akiec sens/spec."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(NUM_CLASSES))
    rows: list[dict[str, Any]] = []

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        rows += [
            {"metric": "precision", "class": name, "value": float(p[i])},
            {"metric": "recall", "class": name, "value": float(r[i])},
            {"metric": "f1", "class": name, "value": float(f[i])},
            {"metric": "support", "class": name, "value": int(s[i])},
        ]
    for avg in ("macro", "weighted"):
        ap, ar, af, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        rows += [
            {"metric": "precision", "class": f"{avg}_avg", "value": float(ap)},
            {"metric": "recall", "class": f"{avg}_avg", "value": float(ar)},
            {"metric": "f1", "class": f"{avg}_avg", "value": float(af)},
        ]
    rows.append({"metric": "accuracy", "class": "all", "value": float(accuracy_score(y_true, y_pred))})

    ak = akiec_sensitivity_specificity(y_true, y_pred)
    rows += [
        {"metric": "sensitivity", "class": "akiec", "value": ak["sensitivity"]},
        {"metric": "specificity", "class": "akiec", "value": ak["specificity"]},
        {"metric": "tp", "class": "akiec", "value": ak["tp"]},
        {"metric": "fn", "class": "akiec", "value": ak["fn"]},
        {"metric": "tn", "class": "akiec", "value": ak["tn"]},
        {"metric": "fp", "class": "akiec", "value": ak["fp"]},
    ]
    rows.append({"metric": "n_test", "class": "all", "value": int(len(y_true))})
    return rows


def write_metrics_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "class", "value"])
        w.writeheader()
        for row in rows:
            v = row["value"]
            w.writerow({**row, "value": f"{v:.6f}" if isinstance(v, float) else v})
    return path


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path, title: str = "") -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or f"EfficientNet-B0 — HAM10000 test set (n={len(y_true)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def print_report(rows: list[dict[str, Any]]) -> None:
    by = {(r["metric"], r["class"]): r["value"] for r in rows}
    print(f"{'class':<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for name in list(CLASS_NAMES) + ["macro_avg", "weighted_avg"]:
        sup = by.get(("support", name), "")
        print(f"{name:<14}{by[('precision', name)]:>10.4f}{by[('recall', name)]:>10.4f}{by[('f1', name)]:>10.4f}{sup:>10}")
    print(f"accuracy            {by[('accuracy', 'all')]:.4f}   (n_test={by[('n_test', 'all')]})")
    print(f"akiec sensitivity   {by[('sensitivity', 'akiec')]:.4f}   (tp={by[('tp', 'akiec')]}, fn={by[('fn', 'akiec')]})")
    print(f"akiec specificity   {by[('specificity', 'akiec')]:.4f}   (tn={by[('tn', 'akiec')]}, fp={by[('fp', 'akiec')]})")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    checkpoint: str | Path,
    pool: data.Pool,
    split: dict[str, np.ndarray],
    out_dir: str | Path,
    device: torch.device | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    """Held-out test evaluation → metrics.csv + confusion_matrix.png in `out_dir`."""
    device = device or get_device()
    out_dir = Path(out_dir)
    model, meta = load_checkpoint(checkpoint, device)

    if meta.get("trained_on") != "train" or not meta.get("split_fingerprint"):
        raise RuntimeError(
            f"refusing to evaluate {checkpoint}: trained_on={meta.get('trained_on')!r}. "
            "Only the split-run checkpoint (model_best.pth) may be scored — the full-data demo model "
            "(model_final.pth) has seen the test images, so its metrics would be contaminated."
        )
    fingerprint = data.split_fingerprint(split, pool.image_ids)
    if meta.get("split_fingerprint") != fingerprint:
        raise RuntimeError(
            "split mismatch: the checkpoint was trained on split "
            f"{meta.get('split_fingerprint')!r} but this evaluation would use {fingerprint!r} "
            "— the test set may overlap the model's training data. Use the same data and seed 42."
        )

    loader = data.build_dataloaders(pool, split, batch_size=batch_size, num_workers=num_workers)["test"]
    val_f1 = meta.get("val_macro_f1")
    val_f1_text = f"{val_f1:.4f}" if isinstance(val_f1, (int, float)) else "n/a"
    print(f"checkpoint={checkpoint}  epoch={meta.get('epoch')}  val_macro_f1={val_f1_text}  device={device}")
    y_true, y_pred, _, _ = predict(model, loader, device)

    rows = compute_metrics(y_true, y_pred)
    rows += [  # provenance, so the CSV alone says which model and split produced it
        {"metric": "checkpoint_file", "class": "meta", "value": Path(checkpoint).name},
        {"metric": "checkpoint_epoch", "class": "meta", "value": int(meta.get("epoch", 0))},
        {"metric": "checkpoint_val_macro_f1", "class": "meta",
         "value": float(val_f1) if isinstance(val_f1, (int, float)) else float("nan")},
        {"metric": "split_fingerprint", "class": "meta", "value": fingerprint},
        {"metric": "dataset", "class": "meta", "value": str(meta.get("dataset", ""))},
    ]
    out_dir.mkdir(parents=True, exist_ok=True)  # only now: a refused checkpoint leaves no trace
    csv_path = write_metrics_csv(rows, out_dir / METRICS_CSV)
    png_path = save_confusion_matrix(y_true, y_pred, out_dir / CONFUSION_PNG)
    print_report(rows)
    print(f"wrote {csv_path} and {png_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model_best.pth on the canonical HAM10000 test split.")
    parser.add_argument("--checkpoint", default="app/model_best.pth", help="split-run checkpoint (never model_final.pth)")
    parser.add_argument("--data-dir", default=str(data.DATA_DIR), help="metadata CSV + image zips (default: %(default)s)")
    parser.add_argument("--out-dir", default=None, help="where to write metrics.csv + confusion_matrix.png (default: checkpoint dir)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="override: cuda | mps | cpu")
    args = parser.parse_args()

    pool, split = data.load_split_data(args.data_dir)
    run(
        args.checkpoint,
        pool,
        split,
        out_dir=args.out_dir or Path(args.checkpoint).parent,
        device=get_device(args.device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
