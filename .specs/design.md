# Design Document

## Overview

A minimal inference app: phone browser → FastAPI server on an M2 MacBook Air → EfficientNet-B0 (trained on HAM10000) → 7-class probabilities + Grad-CAM overlay. Training is a one-off job on a rented Vast.ai GPU (RTX 4090 spot, ~1–2 h, < £1); the demo itself needs no internet in its fallback mode.

## Architecture

```
┌──────────────┐  hotspot or tunnel  ┌───────────────────────────────┐
│ Phone browser │ ─────────────────▶ │ M2 Air: FastAPI (uvicorn)      │
│ camera/upload │ ◀───────────────── │ ├─ EfficientNet-B0 (.pth)      │
└──────────────┘   JSON + heatmap    │ ├─ Grad-CAM (pytorch-grad-cam) │
                                     │ └─ static frontend (1 file)    │
                                     └───────────────────────────────┘
```

### Two demo modes, one codebase

- **Mode A — bulletproof (primary):** M2 Air creates a hotspot; phone joins it; server reachable at the laptop's LAN IP. Zero internet required — venue Wi-Fi cannot hurt the demo.
- **Mode B — attendee QR (bonus):** `cloudflared` quick tunnel exposes a free public HTTPS URL; QR code printed near the poster lets attendees use their own phones. Lives only while the laptop is awake during the poster session — which is exactly when it's needed.

## Components and Interfaces

### 1. Training pipeline (`train.py` + `data.py` + `evaluate.py` — runs on Vast.ai)

- `data.py`: loads HAM10000 (`pranay-43/HAM10000` via `datasets`), builds the canonical stratified 70/15/15 split (seed 42), prints per-class counts, exposes DataLoaders with ImageNet normalization (resize 224 → tensor → normalize).
- `train.py`: EfficientNet-B0 (torchvision, `weights="IMAGENET1K_V1"`), classifier head replaced for 7 classes; CrossEntropyLoss with per-class weights (inverse frequency); Adam lr=1e-4; early stopping on validation macro-F1; saves best checkpoint as `model_best.pth` (state_dict + label map).
- `evaluate.py`: runs the best checkpoint on the held-out test set; computes per-class precision/recall/F1, weighted averages, akiec sensitivity & specificity — all from live predictions; writes `metrics.csv` and `confusion_matrix.png`.
- Artifacts transferred to the M2 Air with `scp`. Instance destroyed immediately after.

### 2. Inference server (`app/` — runs on M2 Air)

- `main.py`: FastAPI. Loads `model_best.pth` at startup onto MPS (CPU fallback); refuses to start if the file is missing/corrupt. Routes:
  - `GET /` → serves `static/index.html`
  - `POST /predict` → validates upload (JPEG/PNG only, ≤ 10 MB), preprocesses, infers, generates Grad-CAM, returns JSON contract (below). All in-memory; nothing written to disk.
- `gradcam.py`: Grad-CAM on `model.features[-1]` via pytorch-grad-cam; overlays heatmap on the original image; returns PNG bytes → base64.
- Expected latency on M2: ~100–200 ms per image — well inside the 3 s budget.

### 3. Frontend (`app/static/index.html` — single file, no framework)

- Mobile-first layout; big capture button using `<input type="file" accept="image/*" capture="environment">`; local preview; spinner during inference; result card with overlay image, top-3 probability bars (akiec highlighted), inference time, and the disclaimer footer: "Research demo — not a diagnostic device."
- Plain `fetch` POST to `/predict`. No build step, no npm, no external CDN (must work on hotspot mode with no internet).

### 4. Demo tooling

- `make_qr.py`: takes a URL, prints and saves a QR PNG.
- `run.sh`: starts uvicorn on `0.0.0.0:8000`; prints the LAN URL for hotspot mode; optionally launches `cloudflared tunnel --url http://localhost:8000` and prints the public URL for QR mode.

## Data Models

### API contract — POST /predict → 200

```json
{
  "predictions": [
    {"class": "akiec", "label": "Actinic keratosis",       "probability": 0.61},
    {"class": "bcc",   "label": "Basal cell carcinoma",    "probability": 0.22},
    {"class": "bkl",   "label": "Benign keratosis",        "probability": 0.09}
  ],
  "heatmap_png_base64": "...",
  "inference_ms": 143
}
```

Label map (fixed, shared by training and server):
`0=akiec, 1=bcc, 2=bkl, 3=df, 4=mel, 5=nv, 6=vasc`

## Error Handling

| Case                     | Behaviour                                   |
|--------------------------|---------------------------------------------|
| Model file missing/corrupt | Server refuses to start, explicit message |
| Non-JPEG/PNG upload      | 400, friendly message                        |
| File > 10 MB             | 413, friendly message                        |
| Corrupt image bytes      | 400, friendly message                        |
| Inference exception      | 500, generic message (logged server-side)    |

Rule: fail loudly at home, never silently at the venue.

## Testing Strategy

- **Unit:** preprocessing output shape/range; label map identical between `train.py` and `main.py`; class-weight computation.
- **Integration:** `/predict` with 3 known HAM10000 test images → expected class appears in top-3; corrupt-file and oversize uploads → correct 4xx.
- **Manual rehearsal (the real test):** full Mode A and Mode B run-through on the presenter's own phone before travel day, including laptop sleep/wake behaviour and tunnel restart.

## Repository Layout

```
skin-demo/
├── .specs/            requirements.md · design.md · tasks.md
├── CLAUDE.md          builder governance rules
├── data.py            canonical split + loaders
├── train.py           Vast.ai training
├── evaluate.py        metrics + confusion matrix
├── app/
│   ├── main.py        FastAPI server
│   ├── gradcam.py
│   ├── static/index.html
│   └── model_best.pth (scp'd down after training)
├── make_qr.py
├── run.sh
├── requirements.txt
├── HANDOVER.md        builder → reviewer baton
└── REVIEW.md          reviewer findings
```

## Dual-Agent Protocol (governance layer)

- **Claude Code = builder.** Implements only what `tasks.md` specifies, one task at a time, commits after each task (`task N: description`), updates `HANDOVER.md` (what was done, how to run, known issues).
- **Kimi Code = reviewer.** Reviews at three checkpoints (after training pipeline, after server, after frontend/rehearsal). Reviews **git diffs against the spec**, not narration; writes findings to `REVIEW.md` as Critical / Major / Minor; builder fixes; reviewer re-checks.
- The `.specs/` folder is the shared contract. If it's not in the spec or the code, it doesn't exist.

### Reviewer standing checklist

- [ ] Every reported number computed live; zero hardcoded metrics
- [ ] Single canonical split used everywhere
- [ ] API response matches the contract exactly
- [ ] Errors fail loudly, never silently
- [ ] No uploaded image ever touches disk
- [ ] akiec sensitivity/specificity reported in metrics.csv
- [ ] Frontend works with no internet (hotspot mode)
