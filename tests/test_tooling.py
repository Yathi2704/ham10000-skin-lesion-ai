"""Demo tooling: make_qr.py and run.sh (task 10)."""

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import make_qr

REPO = Path(__file__).resolve().parents[1]


def test_make_qr_writes_a_decodable_png(tmp_path):
    out = make_qr.make_qr("http://192.168.2.1:8000", tmp_path / "qr.png")
    img = Image.open(out)
    assert img.format == "PNG" and img.size[0] == img.size[1] and img.size[0] > 200
    # a QR code is black-and-white: exactly two greyscale values
    assert set(img.convert("L").getdata()) <= {0, 255}


def test_make_qr_rejects_non_urls(tmp_path):
    with pytest.raises(ValueError):
        make_qr.make_qr("192.168.2.1:8000", tmp_path / "qr.png")


def test_make_qr_cli(tmp_path):
    out = tmp_path / "public.png"
    r = subprocess.run([sys.executable, str(REPO / "make_qr.py"), "https://demo.trycloudflare.com", "--out", str(out), "--no-ascii"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0 and out.is_file() and "demo.trycloudflare.com" in r.stdout


def test_run_sh_refuses_without_model(tmp_path):
    r = subprocess.run(["bash", str(REPO / "run.sh")], capture_output=True, text=True, cwd=REPO,
                       env={"PATH": "/usr/bin:/bin", "MODEL_PATH": str(tmp_path / "missing.pth")})
    assert r.returncode == 1 and "not found" in r.stderr and "HANDOVER" in r.stderr


def test_run_sh_rejects_unknown_option():
    r = subprocess.run(["bash", str(REPO / "run.sh"), "--bogus"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 2 and "unknown option" in r.stderr
