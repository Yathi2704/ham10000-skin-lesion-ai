"""Inference server for the EADV skin-lesion demo (design.md → Inference server).

    uvicorn app.main:app --host 0.0.0.0 --port 8000        (see run.sh)

Startup loads the demo artifact `app/model_final.pth` (env `MODEL_PATH` overrides) onto
MPS → CUDA → CPU and refuses to start — loudly — if the file is missing, corrupt, or does
not carry the canonical label map. This module depends on torch/torchvision/pillow/fastapi
only; nothing from the training code.

Routes:  GET /          the single-file mobile frontend (app/static/index.html)
         GET /health    model + device info (no metrics — numbers live in metrics.csv)
         POST /predict  JPEG/PNG upload → top-3 + Grad-CAM overlay (task 6/7)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

log = logging.getLogger("skin-demo")

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DEFAULT_MODEL_PATH = APP_DIR / "model_final.pth"

# Fixed label map (design.md → Data Models). Must equal data.CLASS_NAMES / CLASS_LABELS —
# duplicated here so the server never imports the training code; tests/test_app.py asserts equality.
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
ARCH = "efficientnet_b0"


class ModelLoadError(RuntimeError):
    """Raised at startup for anything that would leave the demo without a trustworthy model."""


def pick_device(override: str | None = None) -> torch.device:
    """MPS (Apple Silicon) → CUDA → CPU, or an explicit override (env DEVICE)."""
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild EfficientNet-B0 from a train.py checkpoint using torchvision alone; validate the contract."""
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(
            f"model file not found: {path}\n"
            "  → scp app/model_final.pth from the training box, or set MODEL_PATH=/path/to/model.pth"
        )
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # corrupt / not a torch file / unsafe pickle
        raise ModelLoadError(f"model file {path} is unreadable or corrupt: {exc}") from exc
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ModelLoadError(f"model file {path} has no 'state_dict' — not a train.py checkpoint")
    if ckpt.get("arch") != ARCH:
        raise ModelLoadError(f"model file {path}: arch {ckpt.get('arch')!r} != {ARCH!r}")
    if tuple(ckpt.get("class_names", ())) != CLASS_NAMES or ckpt.get("num_classes") != len(CLASS_NAMES):
        raise ModelLoadError(
            f"model file {path}: label map {ckpt.get('class_names')} != canonical {list(CLASS_NAMES)}"
        )
    if {int(k): v for k, v in ckpt.get("label_map", {}).items()} != dict(enumerate(CLASS_NAMES)):
        raise ModelLoadError(f"model file {path}: label_map {ckpt.get('label_map')} is not the canonical map")
    for key in ("image_size", "normalization"):
        if key not in ckpt:
            raise ModelLoadError(f"model file {path}: missing preprocessing metadata {key!r}")

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    try:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(f"model file {path} does not fit {ARCH}: {exc}") from exc
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta


def build_transform(meta: dict[str, Any]) -> transforms.Compose:
    """The eval transform, rebuilt from the checkpoint's own metadata (== data.get_transform())."""
    size = int(meta["image_size"])
    mean, std = meta["normalization"]["mean"], meta["normalization"]["std"]
    return transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean, std)]
    )


class Predictor:
    """Everything the routes need, built once at startup."""

    def __init__(self, model_path: str | Path, device: torch.device):
        self.model_path = Path(model_path)
        self.device = device
        self.model, self.meta = load_model(self.model_path, device)
        self.transform = build_transform(self.meta)
        self.lock = threading.Lock()  # one image at a time through the model + Grad-CAM hooks
        size = int(self.meta["image_size"])
        with torch.no_grad():  # warm-up so the first phone request doesn't pay for kernel compilation
            self.model(torch.zeros(1, 3, size, size, device=device))

    def info(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "arch": self.meta.get("arch"),
            "trained_on": self.meta.get("trained_on"),
            "epoch": self.meta.get("epoch"),
            "trained_at": self.meta.get("trained_at"),
            "classes": list(CLASS_NAMES),
        }


def _startup_failure(exc: BaseException) -> None:
    banner = "=" * 72
    print(f"\n{banner}\nSKIN DEMO SERVER REFUSED TO START\n{exc}\n{banner}\n", file=sys.stderr, flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    device = pick_device(os.environ.get("DEVICE"))
    t0 = time.perf_counter()
    try:
        app.state.predictor = Predictor(model_path, device)
    except Exception as exc:  # fail loudly at home, never silently at the venue
        _startup_failure(exc)
        raise
    log.info("model %s loaded on %s in %.1fs", model_path, device, time.perf_counter() - t0)
    print(f"skin-demo: {model_path} on {device} ({time.perf_counter() - t0:.1f}s)", flush=True)
    yield


app = FastAPI(title="EADV skin-lesion demo", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", **app.state.predictor.info()}
