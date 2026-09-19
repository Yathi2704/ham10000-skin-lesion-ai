"""Inference server for the EADV skin-lesion demo (design.md → Inference server).

    uvicorn app.main:app --host 0.0.0.0 --port 8000        (see run.sh)

Startup loads the demo artifact `app/model_final.pth` (env `MODEL_PATH` overrides) onto
MPS → CUDA → CPU and refuses to start — loudly — if the file is missing, corrupt, or does
not carry the canonical label map. This module depends on torch/torchvision/pillow/fastapi
only; nothing from the training code.

Routes:  GET /          the single-file mobile frontend (app/static/index.html)
         GET /health    model + device info (no metrics — numbers live in metrics.csv)
         POST /predict  JPEG/PNG upload (raw body or multipart field "file") → top-3 + Grad-CAM overlay

Uploads never touch disk: the body is read into memory (capped at 10 MB) and multipart is
parsed in memory too — Starlette's own UploadFile would spool anything over 1 MB to a temp file.
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from python_multipart import MultipartParser
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

from .gradcam import HeatmapGenerator, png_to_base64

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

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # design.md: > 10 MB → 413
TOP_K = 3
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MSG_TOO_LARGE = "Image is too large — please send a JPEG or PNG under 10 MB."
MSG_WRONG_TYPE = "Only JPEG or PNG images are accepted."
MSG_UNREADABLE = "That image could not be read — please try another photo."
MSG_EMPTY = "No image received — attach a JPEG or PNG."
MSG_INFERENCE = "Inference failed on the server — please try another image."


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
        self.heatmaps = HeatmapGenerator(self.model)
        self.lock = threading.Lock()  # one image at a time through the model + Grad-CAM hooks
        size = int(self.meta["image_size"])
        warm = torch.zeros(1, 3, size, size, device=device)
        with torch.no_grad():  # warm-up so the first phone request doesn't pay for kernel compilation
            self.model(warm)
        self.heatmaps.heatmap(warm, 0)

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


    def predict(self, image: Image.Image) -> dict[str, Any]:
        """One RGB PIL image → the /predict JSON body (design.md → API contract).

        `inference_ms` covers preprocessing, the forward pass and the Grad-CAM overlay
        (not the upload). The heatmap is for the top-1 class.
        """
        t0 = time.perf_counter()
        x = self.transform(image).unsqueeze(0).to(self.device)
        with self.lock:
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1)[0]
            top = torch.topk(probs, k=TOP_K)
            png = self.heatmaps.overlay_png(x, int(top.indices[0]), image)
        predictions = [
            {
                "class": CLASS_NAMES[int(i)],
                "label": CLASS_LABELS[CLASS_NAMES[int(i)]],
                "probability": round(float(p), 4),
            }
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]
        return {
            "predictions": predictions,
            "heatmap_png_base64": png_to_base64(png),
            "inference_ms": int(round((time.perf_counter() - t0) * 1000)),
        }


# --------------------------------------------------------------------------- #
# Upload handling — everything stays in memory
# --------------------------------------------------------------------------- #
class _MemoryMultipart:
    """Minimal in-memory multipart/form-data reader (python-multipart callbacks → BytesIO)."""

    def __init__(self, boundary: str):
        self.parts: list[dict[str, Any]] = []
        self._field = b""
        self._value = b""
        self._parser = MultipartParser(
            boundary,
            {
                "on_part_begin": self._begin,
                "on_header_field": lambda data, start, end: self._append("_field", data[start:end]),
                "on_header_value": lambda data, start, end: self._append("_value", data[start:end]),
                "on_header_end": self._header_end,
                "on_part_data": lambda data, start, end: self.parts[-1]["data"].write(data[start:end]),
            },
        )

    def _begin(self) -> None:
        self.parts.append({"headers": {}, "data": io.BytesIO()})

    def _append(self, attr: str, chunk: bytes) -> None:
        setattr(self, attr, getattr(self, attr) + chunk)

    def _header_end(self) -> None:
        self.parts[-1]["headers"][self._field.decode("latin-1").lower()] = self._value.decode("latin-1")
        self._field, self._value = b"", b""

    def feed(self, body: bytes) -> list[dict[str, Any]]:
        self._parser.write(body)
        self._parser.finalize()
        return self.parts


def extract_upload(body: bytes, content_type: str) -> bytes:
    """Raw image body, or the `file` part (else the first file part) of a multipart body."""
    ctype = (content_type or "").lower()
    if not ctype.startswith("multipart/form-data"):
        return body
    m = re.search(r'boundary="?([^";]+)"?', content_type, flags=re.IGNORECASE)
    if not m:
        raise HTTPException(status_code=400, detail=MSG_EMPTY)
    try:
        parts = _MemoryMultipart(m.group(1)).feed(body)
    except Exception:
        raise HTTPException(status_code=400, detail=MSG_EMPTY) from None
    files = [p for p in parts if "filename=" in p["headers"].get("content-disposition", "")]
    named = [p for p in files if re.search(r'name="?file"?', p["headers"]["content-disposition"])]
    chosen = (named or files or parts)[:1]
    if not chosen:
        raise HTTPException(status_code=400, detail=MSG_EMPTY)
    return chosen[0]["data"].getvalue()


def decode_image(raw: bytes) -> Image.Image:
    """Bytes → RGB PIL image, or a friendly 4xx. JPEG/PNG only, judged by magic bytes not by name."""
    if not raw:
        raise HTTPException(status_code=400, detail=MSG_EMPTY)
    if not (raw.startswith(JPEG_MAGIC) or raw.startswith(PNG_MAGIC)):
        raise HTTPException(status_code=400, detail=MSG_WRONG_TYPE)
    try:
        probe = Image.open(io.BytesIO(raw))
        probe.verify()  # structural check; invalidates `probe`
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image)  # honour the phone's orientation tag
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail=MSG_UNREADABLE) from None


async def read_body_capped(request: Request, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    """Stream the body into memory; 413 as soon as it exceeds `limit` (Content-Length may lie or be absent)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail=MSG_TOO_LARGE)
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > limit:
            raise HTTPException(status_code=413, detail=MSG_TOO_LARGE)
    return bytes(buf)


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


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", **app.state.predictor.info()}


@app.post("/predict")
async def predict(request: Request) -> dict[str, Any]:
    body = await read_body_capped(request)
    raw = extract_upload(body, request.headers.get("content-type", ""))
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=MSG_TOO_LARGE)
    image = decode_image(raw)
    try:
        return await run_in_threadpool(app.state.predictor.predict, image)
    except Exception:
        log.exception("inference failed (%sx%s %s)", image.width, image.height, image.mode)
        raise HTTPException(status_code=500, detail=MSG_INFERENCE) from None
