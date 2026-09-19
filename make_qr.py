"""URL → QR code PNG (+ ASCII preview in the terminal).

    python make_qr.py https://example.trycloudflare.com            # writes qr.png
    python make_qr.py http://192.168.2.1:8000 --out qr_hotspot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_M


def make_qr(url: str, out: str | Path = "qr.png", box_size: int = 12, border: int = 4) -> Path:
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"not an http(s) URL: {url!r}")
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_M, box_size=box_size, border=border)
    qr.add_data(url)
    qr.make(fit=True)
    out = Path(out)
    qr.make_image(fill_color="black", back_color="white").save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a QR code PNG for the demo URL.")
    parser.add_argument("url", help="the URL to encode (hotspot LAN URL or cloudflared public URL)")
    parser.add_argument("--out", default="qr.png", help="output PNG (default: %(default)s)")
    parser.add_argument("--no-ascii", action="store_true", help="skip the terminal preview")
    args = parser.parse_args()
    try:
        out = make_qr(args.url, args.out)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    if not args.no_ascii:
        qr = qrcode.QRCode(error_correction=ERROR_CORRECT_M, border=1)
        qr.add_data(args.url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    print(f"{args.url}\n→ {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
