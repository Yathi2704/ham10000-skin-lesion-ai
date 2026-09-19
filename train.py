"""Train EfficientNet-B0 on the canonical HAM10000 split.

Runs on a rented CUDA GPU (Vast.ai); also runs on the M2 Air (MPS) or CPU for
smoke tests. Everything data-related comes from data.py — the split is never
recomputed with a different seed here.

    python train.py --epochs 30 --patience 5 --out app/model_best.pth

Output: `model_best.pth` — plain state_dict + label map + preprocessing
metadata, loadable with torchvision alone (no training code, weights_only=True).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

import data
from data import (
    BATCH_SIZE,
    CLASS_LABELS,
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_MAP,
    NUM_CLASSES,
    SEED,
)

ARCH = "efficientnet_b0"
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 30
DEFAULT_PATIENCE = 5


# --------------------------------------------------------------------------- #
# Device / seeding
# --------------------------------------------------------------------------- #
def get_device(prefer: str | None = None) -> torch.device:
    """cuda → mps → cpu, or an explicit override (`--device`)."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """torchvision EfficientNet-B0 (ImageNet weights) with a fresh `num_classes` head."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(labels: np.ndarray | list[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency weights, w_c = N / (C · n_c)  (sklearn's "balanced" formula; Σ w_c·n_c = N).

    Computed from the *training* partition only. On the real HAM10000 (~67% nv)
    this up-weights akiec/df/vasc by roughly an order of magnitude; on a
    balanced subset every weight is exactly 1.0.
    """
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        empty = [CLASS_NAMES[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"training partition has no samples for {empty}; cannot weight the loss")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Checkpoint format (the contract with evaluate.py and the server)
# --------------------------------------------------------------------------- #
def checkpoint_payload(model: nn.Module, **meta: Any) -> dict[str, Any]:
    """Only tensors + JSON-ish primitives, so `torch.load(..., weights_only=True)` works."""
    payload: dict[str, Any] = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arch": ARCH,
        "num_classes": NUM_CLASSES,
        "label_map": dict(LABEL_MAP),
        "class_names": list(CLASS_NAMES),
        "class_labels": dict(CLASS_LABELS),
        "image_size": IMAGE_SIZE,
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "torch_version": str(torch.__version__),
    }
    payload.update(meta)
    return payload


def save_checkpoint(path: Path, model: nn.Module, **meta: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(model, **meta), tmp)
    tmp.replace(path)  # atomic: never leave a half-written model_best.pth


# --------------------------------------------------------------------------- #
# Train / predict loops
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: int | None = None,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for b, (x, y) in enumerate(loader):
        if limit_batches is not None and b >= limit_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    limit_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Live predictions over a loader → (y_true, y_pred, probs[N,C], mean_loss)."""
    model.eval()
    ys, preds, probs = [], [], []
    total_loss, n = 0.0, 0
    for b, (x, y) in enumerate(loader):
        if limit_batches is not None and b >= limit_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            total_loss += float(criterion(logits, y)) * len(y)
        n += len(y)
        p = torch.softmax(logits, dim=1)
        ys.append(y.cpu().numpy())
        preds.append(p.argmax(dim=1).cpu().numpy())
        probs.append(p.cpu().numpy())
    if n == 0:
        raise RuntimeError("predict() received no batches — empty loader?")
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs), total_loss / n


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0))


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #
def train(
    pool: Any,
    split: dict[str, np.ndarray],
    out_path: Path,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    lr: float = DEFAULT_LR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    device: torch.device | None = None,
    pretrained: bool = True,
    limit_batches: int | None = None,
    dataset_name: str = data.DATASET_NAME,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train with early stopping on validation macro-F1; save the best epoch to `out_path`."""
    seed_everything(seed)
    device = device or get_device()
    out_path = Path(out_path)
    loaders = data.build_dataloaders(pool, split, batch_size=batch_size, num_workers=num_workers, seed=seed)

    train_labels = np.asarray(pool["label"])[split["train"]]
    class_weights = compute_class_weights(train_labels)
    print(f"device={device}  epochs≤{epochs}  patience={patience}  lr={lr}  batch={batch_size}")
    print("class weights (train partition, inverse frequency):")
    for name, w in zip(CLASS_NAMES, class_weights.tolist()):
        print(f"  {name:<6} {w:.3f}")

    model = build_model(pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fingerprint = data.split_fingerprint(split, pool["image_id"])
    log_path = out_path.with_name("train_log.csv")
    best = {"val_macro_f1": -1.0, "epoch": 0}
    epochs_without_improvement = 0

    with open(log_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1", "best_val_macro_f1", "seconds"])
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device, limit_batches)
            y_true, y_pred, _, val_loss = predict(model, loaders["val"], device, criterion, limit_batches)
            val_f1 = macro_f1(y_true, y_pred)
            improved = val_f1 > best["val_macro_f1"]
            if improved:
                best = {"val_macro_f1": val_f1, "epoch": epoch}
                epochs_without_improvement = 0
                save_checkpoint(
                    out_path,
                    model,
                    epoch=epoch,
                    val_macro_f1=val_f1,
                    val_loss=val_loss,
                    train_loss=train_loss,
                    seed=seed,
                    dataset=dataset_name,
                    split_fractions=list(data.SPLIT_FRACTIONS),
                    split_fingerprint=fingerprint,
                    class_weights=class_weights.tolist(),
                    lr=lr,
                    batch_size=batch_size,
                    pretrained=pretrained,
                    trained_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                )
            else:
                epochs_without_improvement += 1
            secs = time.time() - t0
            log.writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}", f"{val_f1:.5f}", f"{best['val_macro_f1']:.5f}", f"{secs:.1f}"])
            f.flush()
            print(
                f"epoch {epoch:>3}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
                f"val_macroF1 {val_f1:.4f}  best {best['val_macro_f1']:.4f}@{best['epoch']}"
                f"{'  *saved*' if improved else ''}  {secs:.0f}s"
            )
            if epochs_without_improvement >= patience:
                print(f"early stopping: no val macro-F1 improvement for {patience} epochs")
                break

    print(f"best val macro-F1 {best['val_macro_f1']:.4f} at epoch {best['epoch']} → {out_path}")
    print(f"per-epoch log → {log_path}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on the canonical HAM10000 split.")
    parser.add_argument("--dataset", default=data.DATASET_NAME, help="HF dataset id (default: %(default)s)")
    parser.add_argument("--out", default="app/model_best.pth", help="checkpoint path (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="early-stopping patience on val macro-F1")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (use 4–8 on Vast.ai)")
    parser.add_argument("--seed", type=int, default=SEED, help="split + init seed; leave at 42 for the canonical split")
    parser.add_argument("--device", default=None, help="override: cuda | mps | cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline smoke tests only)")
    parser.add_argument("--limit-batches", type=int, default=None, help="SMOKE TEST ONLY: batches per epoch")
    args = parser.parse_args()

    if args.seed != SEED:
        print(f"WARNING: --seed {args.seed} ≠ canonical seed {SEED}; this is NOT the canonical split")

    pool, split = data.load_split_data(args.dataset, seed=args.seed)
    train(
        pool,
        split,
        out_path=Path(args.out),
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=get_device(args.device),
        pretrained=not args.no_pretrained,
        limit_batches=args.limit_batches,
        dataset_name=args.dataset,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
