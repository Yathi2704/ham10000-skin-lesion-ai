"""Task 8 — integration tests against the served model (design.md → Testing Strategy).

8.1 / 8.3 need real artifacts and are skipped (loudly, with the reason) until they exist:
    app/model_final.pth   the demo model the server loads (also app/model_best.pth if present)
    data/ham10000/        metadata CSV (committed) + image zips (see data/ham10000/README.md)
8.2 (bad uploads → 4xx) is covered exhaustively in tests/test_app.py and re-run here against
the real model when it exists.
"""

import io
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import data
from app import main as server

REPO = Path(__file__).resolve().parents[1]
MODEL_FILES = {"model_final.pth": REPO / "app" / "model_final.pth", "model_best.pth": REPO / "app" / "model_best.pth"}
KNOWN_CLASSES = ("akiec", "mel", "nv")  # the poster's focus class, the dangerous one, the majority one


def _images_missing_reason() -> str | None:
    """None when the full release is under data/ham10000, else why not (skip reason)."""
    if not (data.DATA_DIR / data.METADATA_CSV).is_file():
        return f"{data.DATA_DIR / data.METADATA_CSV} missing"
    try:
        n = len(data.FileImageStore(data.DATA_DIR))
    except zipfile.BadZipFile as exc:  # a partial download is not a smoke-run failure, just not ready
        return f"image zip under {data.DATA_DIR} is incomplete or corrupt ({exc})"
    if n < data.HAM10000_N_IMAGES:
        return f"{n} of {data.HAM10000_N_IMAGES} images under {data.DATA_DIR} — see data/ham10000/README.md"
    return None


def known_test_images(classes=KNOWN_CLASSES) -> list[tuple[str, str, bytes]]:
    """(image_id, true class, JPEG bytes) — the first canonical *test*-partition image of each class."""
    pool = data.load_pool(require_images=True)
    split = data.make_split(pool.labels, pool.lesion_ids)
    out = []
    for name in classes:
        idx = next(i for i in split["test"] if pool.labels[i] == data.CLASS_TO_INDEX[name])
        image_id = pool.image_ids[idx]
        buf = io.BytesIO()
        pool.images.open(image_id).save(buf, format="JPEG", quality=95)
        out.append((image_id, name, buf.getvalue()))
    return out


def phone_photo(size=(4000, 3000)) -> bytes:
    """A 12 MP JPEG the size a real phone produces (~2–4 MB): smooth skin-like gradient + mild noise.

    Built in float32 with broadcasting (peak ≈ 300 MB); the earlier float64 mgrid version peaked
    over 1 GB and got the full suite SIGKILLed on a small CI container (REVIEW.md, Minor 1).
    """
    rng = np.random.default_rng(0)
    w, h = size
    xx = np.arange(w, dtype=np.float32)[None, :]
    yy = np.arange(h, dtype=np.float32)[:, None]
    img = np.empty((h, w, 3), dtype=np.float32)
    img[..., 0] = 200 + 30 * np.sin(xx / 400)
    img[..., 1] = 150 + 30 * np.cos(yy / 300)
    img[..., 2] = 140 + 20 * np.sin((xx + yy) / 500)
    noise = rng.normal(0, 10, size=(h // 2, w // 2, 3)).astype(np.float32)
    img += noise.repeat(2, axis=0).repeat(2, axis=1)
    del noise
    photo = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    del img
    buf = io.BytesIO()
    photo.save(buf, format="JPEG", quality=92)
    payload = buf.getvalue()
    assert 1_000_000 < len(payload) < server.MAX_UPLOAD_BYTES, len(payload)
    return payload


def _client_for(model_path: Path):
    if not model_path.is_file():
        pytest.skip(f"{model_path.relative_to(REPO)} not present — scp it from the training box (HANDOVER.md)")
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ.pop("DEVICE", None)  # let the server pick MPS on the M2
    return TestClient(server.app)


@pytest.mark.parametrize("model_name", list(MODEL_FILES))
def test_8_1_known_test_images_expected_class_in_top3(model_name):
    reason = _images_missing_reason()
    if reason:
        pytest.skip(reason)
    with _client_for(MODEL_FILES[model_name]) as client:
        for image_id, true_class, raw in known_test_images():
            r = client.post("/predict", files={"file": (f"{image_id}.jpg", raw, "image/jpeg")})
            assert r.status_code == 200, r.text
            top3 = [p["class"] for p in r.json()["predictions"]]
            assert true_class in top3, f"{model_name}: {image_id} is {true_class}, server said {top3}"


def test_8_2_bad_uploads_against_real_model():
    with _client_for(MODEL_FILES["model_final.pth"]) as client:
        assert client.post("/predict", files={"file": ("x.gif", b"GIF89a" + bytes(32), "image/gif")}).status_code == 400
        assert client.post("/predict", files={"file": ("x.jpg", b"\xff\xd8\xff" + bytes(99), "image/jpeg")}).status_code == 400
        big = b"\xff\xd8\xff" + bytes(server.MAX_UPLOAD_BYTES)
        assert client.post("/predict", files={"file": ("big.jpg", big, "image/jpeg")}).status_code == 413


@pytest.mark.heavy
def test_8_3_latency_under_3s_on_this_machine():
    """Single image through the real demo model, measured end to end on the M2 (MPS if available)."""
    with _client_for(MODEL_FILES["model_final.pth"]) as client:
        payload = phone_photo()
        client.post("/predict", files={"file": ("warm.jpg", payload, "image/jpeg")})
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = client.post("/predict", files={"file": ("p.jpg", payload, "image/jpeg")})
            times.append(time.perf_counter() - t0)
            assert r.status_code == 200 and r.json()["inference_ms"] < 3000
        device = client.get("/health").json()["device"]
        print(f"\nlatency on {device}: {[f'{t * 1000:.0f} ms' for t in times]} (inference_ms={r.json()['inference_ms']})")
        assert max(times) < 3.0


@pytest.mark.heavy
def test_8_3_latency_budget_holds_for_the_architecture(tmp_path):
    """Same check with a random-weight checkpoint so the budget is verified before the real model exists."""
    import train

    path = tmp_path / "model_final.pth"
    train.save_checkpoint(path, train.build_model(pretrained=False), trained_on="all", epoch=1)
    os.environ["MODEL_PATH"] = str(path)
    os.environ.pop("DEVICE", None)
    with TestClient(server.app) as client:
        payload = phone_photo()
        client.post("/predict", files={"file": ("warm.jpg", payload, "image/jpeg")})
        t0 = time.perf_counter()
        r = client.post("/predict", files={"file": ("p.jpg", payload, "image/jpeg")})
        wall = time.perf_counter() - t0
        device = client.get("/health").json()["device"]
        print(f"\narchitecture latency on {device}: {wall * 1000:.0f} ms wall, inference_ms={r.json()['inference_ms']}")
        assert r.status_code == 200 and wall < 3.0 and r.json()["inference_ms"] < 3000
    os.environ.pop("MODEL_PATH", None)
