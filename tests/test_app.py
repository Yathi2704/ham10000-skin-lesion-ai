"""Integration tests for the inference server (design.md → Testing Strategy).

A randomly initialised checkpoint in the same format as train.py's proves the contract,
validation, in-memory handling and Grad-CAM path without the trained model. Tests marked
`needs_model` use the real demo artifact and are skipped until it exists.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

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
