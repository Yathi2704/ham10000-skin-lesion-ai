"""Integration tests for the inference server (design.md → Testing Strategy).

A randomly initialised checkpoint in the same format as train.py's proves the contract,
validation, in-memory handling and Grad-CAM path without the trained model. Tests marked
`needs_model` use the real demo artifact and are skipped until it exists.
"""

import io
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

import data
import train
from app import main as server

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------- fixtures --
@pytest.fixture(scope="module")
def fake_checkpoint(tmp_path_factory) -> Path:
    """train.py checkpoint format, random weights, canonical label map."""
    path = tmp_path_factory.mktemp("model") / "model_final.pth"
    train.save_checkpoint(path, train.build_model(pretrained=False), trained_on="all", epoch=1)
    return path


@pytest.fixture(scope="module")
def client(fake_checkpoint):
    os.environ["MODEL_PATH"] = str(fake_checkpoint)
    os.environ["DEVICE"] = "cpu"
    with TestClient(server.app) as c:
        yield c
    os.environ.pop("MODEL_PATH", None)
    os.environ.pop("DEVICE", None)


# ---------------------------------------------------------- contract w/ data.py --
def test_label_map_identical_to_training_code():
    assert server.CLASS_NAMES == data.CLASS_NAMES
    assert server.CLASS_LABELS == data.CLASS_LABELS
    assert server.ARCH == train.ARCH


def test_server_transform_equals_eval_transform(fake_checkpoint):
    _, meta = server.load_model(fake_checkpoint, torch.device("cpu"))
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, size=(450, 600, 3), dtype=np.uint8))
    assert torch.equal(server.build_transform(meta)(img), data.get_transform(train=False)(img))


# -------------------------------------------------------------- startup --
def test_startup_refuses_missing_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.pth"))
    monkeypatch.setenv("DEVICE", "cpu")
    with pytest.raises(server.ModelLoadError, match="not found"):
        with TestClient(server.app):
            pass


def test_startup_refuses_corrupt_model(monkeypatch, tmp_path):
    bad = tmp_path / "model_final.pth"
    bad.write_bytes(b"not a checkpoint")
    monkeypatch.setenv("MODEL_PATH", str(bad))
    monkeypatch.setenv("DEVICE", "cpu")
    with pytest.raises(server.ModelLoadError, match="corrupt"):
        with TestClient(server.app):
            pass


def test_startup_refuses_wrong_label_map(monkeypatch, tmp_path):
    path = tmp_path / "model_final.pth"
    train.save_checkpoint(path, train.build_model(pretrained=False), class_names=list(reversed(data.CLASS_NAMES)))
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("DEVICE", "cpu")
    with pytest.raises(server.ModelLoadError, match="label map"):
        with TestClient(server.app):
            pass


def test_index_and_health(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "not a diagnostic device" in r.text.lower()
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["classes"] == list(data.CLASS_NAMES) and h["device"] == "cpu"


# ------------------------------------------------------------- /predict --
def _jpeg(size=(600, 450), seed=0, quality=90, exif_orientation: int | None = None) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8))
    buf = io.BytesIO()
    kwargs = {}
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif.tobytes()
    img.save(buf, format="JPEG", quality=quality, **kwargs)
    return buf.getvalue()


def _png(size=(300, 200)) -> bytes:
    img = Image.fromarray(np.random.default_rng(1).integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _assert_contract(body: dict):
    assert set(body) >= {"predictions", "inference_ms"}
    preds = body["predictions"]
    assert len(preds) == 3
    for p in preds:
        assert set(p) == {"class", "label", "probability"}
        assert p["class"] in data.CLASS_NAMES and p["label"] == data.CLASS_LABELS[p["class"]]
        assert 0.0 <= p["probability"] <= 1.0
    probs = [p["probability"] for p in preds]
    assert probs == sorted(probs, reverse=True) and sum(probs) <= 1.0 + 1e-3
    assert len({p["class"] for p in preds}) == 3
    assert isinstance(body["inference_ms"], int) and body["inference_ms"] >= 0


def test_predict_multipart_jpeg_matches_contract(client):
    r = client.post("/predict", files={"file": ("lesion.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    _assert_contract(r.json())


def test_predict_raw_png_body(client):
    r = client.post("/predict", content=_png(), headers={"content-type": "image/png"})
    assert r.status_code == 200, r.text
    _assert_contract(r.json())


def test_predict_is_deterministic_and_matches_direct_model(client, fake_checkpoint):
    raw = _jpeg(seed=4)
    a = client.post("/predict", files={"file": ("a.jpg", raw, "image/jpeg")}).json()
    b = client.post("/predict", files={"file": ("a.jpg", raw, "image/jpeg")}).json()
    assert a["predictions"] == b["predictions"]
    model, meta = server.load_model(fake_checkpoint, torch.device("cpu"))
    x = server.build_transform(meta)(Image.open(io.BytesIO(raw)).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, k=3)
    assert [p["class"] for p in a["predictions"]] == [data.CLASS_NAMES[i] for i in top.indices.tolist()]
    assert [p["probability"] for p in a["predictions"]] == pytest.approx([round(v, 4) for v in top.values.tolist()])


def test_predict_honours_exif_orientation(client, fake_checkpoint):
    """A phone photo tagged 'rotate 90°' is scored as the user sees it, not as stored."""
    raw = _jpeg(size=(600, 450), seed=7, exif_orientation=6)
    out = client.post("/predict", files={"file": ("p.jpg", raw, "image/jpeg")}).json()
    model, meta = server.load_model(fake_checkpoint, torch.device("cpu"))
    upright = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(model(server.build_transform(meta)(upright).unsqueeze(0)), dim=1)[0]
    assert out["predictions"][0]["class"] == data.CLASS_NAMES[int(probs.argmax())]


@pytest.mark.parametrize(
    "payload, ctype, status, message",
    [
        (b"GIF89a" + bytes(64), "image/gif", 400, "Only JPEG or PNG"),
        (b"%PDF-1.4 not an image", "application/pdf", 400, "Only JPEG or PNG"),
        (b"", "image/jpeg", 400, "No image"),
        (b"\xff\xd8\xff" + bytes(200), "image/jpeg", 400, "could not be read"),   # JPEG magic, garbage body
        (b"\x89PNG\r\n\x1a\n" + bytes(50), "image/png", 400, "could not be read"),
        (_jpeg()[: len(_jpeg()) // 2], "image/jpeg", 400, "could not be read"),   # truncated JPEG
    ],
)
def test_predict_rejects_bad_uploads_with_friendly_4xx(client, payload, ctype, status, message):
    r = client.post("/predict", files={"file": ("x", payload, ctype)})
    assert r.status_code == status, r.text
    assert message in r.json()["detail"]
    r = client.post("/predict", content=payload, headers={"content-type": ctype})
    assert r.status_code == status, r.text


def test_predict_rejects_oversize_upload_with_413(client):
    big = _jpeg() + bytes(server.MAX_UPLOAD_BYTES)  # valid header, > 10 MB
    r = client.post("/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413 and "10 MB" in r.json()["detail"]
    r = client.post("/predict", content=big, headers={"content-type": "image/jpeg"})
    assert r.status_code == 413
    # a lying Content-Length must not get around the cap either
    r = client.post("/predict", content=big, headers={"content-type": "image/jpeg", "content-length": "100"})
    assert r.status_code == 413


def test_predict_never_touches_disk(client, monkeypatch):
    """A 2–3 MB multipart upload (above Starlette's 1 MB spool threshold) with all temp-file paths booby-trapped."""
    import pathlib
    import tempfile

    def boom(*a, **k):
        raise AssertionError("upload handling tried to create a file on disk")

    for name in ("SpooledTemporaryFile", "NamedTemporaryFile", "TemporaryFile", "mkstemp", "mkdtemp"):
        monkeypatch.setattr(tempfile, name, boom)
    monkeypatch.setattr(pathlib.Path, "write_bytes", boom)
    monkeypatch.setattr(pathlib.Path, "write_text", boom)
    raw = _jpeg(size=(1600, 1600), seed=2, quality=100)
    assert len(raw) > 1024 * 1024
    r = client.post("/predict", files={"file": ("photo.jpg", raw, "image/jpeg")})
    assert r.status_code == 200, r.text
    _assert_contract(r.json())


def test_predict_500_is_generic_when_inference_breaks(client, monkeypatch):
    def broken(image):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(server.app.state.predictor, "predict", broken)
    r = client.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 500
    assert "secret" not in r.text and "try another image" in r.json()["detail"]


def test_read_body_capped_streams_and_stops_without_content_length():
    """The streaming guard itself: no Content-Length, body larger than the cap → 413 mid-stream."""
    import asyncio

    from fastapi import HTTPException

    class FakeRequest:
        headers = {}

        async def stream(self):
            for _ in range(4):
                yield bytes(3 * 1024 * 1024)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.read_body_capped(FakeRequest()))
    assert exc.value.status_code == 413

    class SmallRequest:
        headers = {}

        async def stream(self):
            yield b"abc"
            yield b"def"

    assert asyncio.run(server.read_body_capped(SmallRequest())) == b"abcdef"
