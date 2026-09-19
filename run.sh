#!/usr/bin/env bash
# Start the demo server on the M2 Air.
#
#   ./run.sh              Mode A — hotspot / shared Wi-Fi: prints the LAN URL(s) to type or QR-encode
#   ./run.sh --tunnel     Mode B — also opens a cloudflared quick tunnel and prints (+ QR-encodes) the public URL
#
# Env: PORT (default 8000), MODEL_PATH (default app/model_final.pth), DEVICE (mps|cpu override).
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
MODEL="${MODEL_PATH:-app/model_final.pth}"
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
TUNNEL=0
for arg in "$@"; do
  case "$arg" in
    --tunnel) TUNNEL=1 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -f "$MODEL" ]; then
  echo "ERROR: model file '$MODEL' not found — scp app/model_final.pth from the training box (see HANDOVER.md)" >&2
  exit 1
fi

echo "Mode A — phone on the laptop's hotspot / same Wi-Fi, open one of:"
for ip in $(ifconfig 2>/dev/null | awk '/inet / && $2 != "127.0.0.1" {print $2}'); do
  echo "   http://$ip:$PORT"
done
echo "   (macOS Internet Sharing usually gives the laptop 192.168.2.1; QR: $PY make_qr.py http://<ip>:$PORT)"

TUNNEL_PID=""
cleanup() { [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ "$TUNNEL" = 1 ]; then
  command -v cloudflared >/dev/null 2>&1 || { echo "ERROR: cloudflared not installed (brew install cloudflared)" >&2; exit 1; }
  TUNNEL_LOG="$(mktemp -t cloudflared)"
  cloudflared tunnel --url "http://localhost:$PORT" --no-autoupdate >"$TUNNEL_LOG" 2>&1 &
  TUNNEL_PID=$!
  echo "Mode B — starting cloudflared quick tunnel…"
  URL=""
  for _ in $(seq 1 60); do
    URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)"
    [ -n "$URL" ] && break
    sleep 1
  done
  if [ -z "$URL" ]; then
    echo "ERROR: no public URL from cloudflared after 60 s — log: $TUNNEL_LOG" >&2; exit 1
  fi
  echo "Mode B — public URL (lives while this laptop is awake): $URL"
  "$PY" make_qr.py "$URL" --out qr.png
fi

echo "Serving $MODEL on 0.0.0.0:$PORT — Ctrl-C to stop"
"$PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
