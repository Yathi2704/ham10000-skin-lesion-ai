# Design Document

## Amendment C (2026-09-19)

Adopted after Checkpoint 1 — see requirements.md. Changes are marked **[C]**: original HAM10000
release instead of the `pranay-43/HAM10000` HF mirror (a 700-image balanced subset); split grouped
by `lesion_id`; train-time augmentation; evaluate-then-retrain (`model_best.pth` for metrics,
`model_final.pth` for the demo).

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

- `data.py` **[C]**: loads the original HAM10000 release from `data/ham10000/` (`HAM10000_metadata.csv` + `HAM10000_images_part_{1,2}.zip`, read straight from the zips), de-duplicates by `image_id`, maps `dx` via an alias table to the fixed label map, asserts 10 015 images with the published class counts, builds the canonical 70/15/15 split (seed 42, stratified by class, **grouped by `lesion_id`**), prints per-class image and lesion counts plus a content fingerprint, exposes DataLoaders. Eval transform: resize 224 → tensor → ImageNet normalize. Train transform adds random horizontal/vertical flips and random 90° rotations (dihedral group — exact, no interpolation) and is used for the train partition only.
- `train.py` **[C]**: EfficientNet-B0 (torchvision, `weights="IMAGENET1K_V1"`), classifier head replaced for 7 classes; CrossEntropyLoss with per-class weights (inverse frequency); Adam lr=1e-4. Two modes:
  - `--mode split` (default): trains on the train partition, early stopping on validation macro-F1, saves the best epoch as `model_best.pth` (state_dict + label map + preprocessing metadata + `split_fingerprint`, `trained_on="train"`).
  - `--mode full --epochs-from app/model_best.pth`: after metrics are frozen, retrains on 100 % of the images for exactly the split run's best epoch count (class weights from all labels), saves `model_final.pth` (`trained_on="all"`, `split_fingerprint=None`, `derived_from` = the split checkpoint's provenance). No validation, no metrics.
- `evaluate.py` **[C]**: runs `model_best.pth` on the held-out test set; refuses any checkpoint whose `trained_on != "train"` or whose `split_fingerprint` differs from the live split; computes per-class precision/recall/F1, macro + weighted averages, accuracy, akiec sensitivity & specificity — all from live predictions; writes `metrics.csv` (long form `metric,class,value` + provenance rows) and `confusion_matrix.png`.
- Order on the GPU box: `data.py --check-images` → `train.py` → `evaluate.py` → `train.py --mode full --epochs-from app/model_best.pth`. Artifacts (`model_best.pth`, `model_final.pth`, `metrics.csv`, `confusion_matrix.png`, `train_log.csv`, `train_log_full.csv`) transferred to the M2 Air with `scp`. Instance destroyed immediately after.

### 2. Inference server (`app/` — runs on M2 Air)

- `main.py` **[C]**: FastAPI. Loads `app/model_final.pth` (env `MODEL_PATH` overrides) at startup onto MPS (CPU fallback); refuses to start if the file is missing/corrupt or its label map differs from the canonical one. Routes:
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

### Checkpoint format **[C]** (`model_best.pth` / `model_final.pth`)

Plain `torch.save` dict of tensors and primitives — loads with `torch.load(..., weights_only=True)` and
torchvision alone: `state_dict`, `arch`, `num_classes`, `label_map`, `class_names`, `class_labels`,
`image_size`, `normalization{mean,std}`, `trained_on` (`"train"` | `"all"`), `epoch`,
`split_fingerprint` (`None` for the full model), `derived_from` (full model only), training
hyper-parameters, `trained_at`.

### Split fingerprint **[C]**

`sha256` over the sorted image ids of train, val and test (first 16 hex chars). Canonical value for
the release + seed 42: **`4b4cc59260945104`** — printed by `python data.py`, stored in every split-run
checkpoint, verified by `evaluate.py` and pinned by `tests/test_data.py`.

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

- **Unit:** preprocessing output shape/range; label map identical between `train.py` and `main.py`; class-weight computation. **[C]** Also: de-dup/alias/release assertions on synthetic CSVs, lesion-grouping invariants, fingerprint stability, augmentation keeps the pixel multiset, full-data mode provenance, evaluator refuses `model_final.pth`.
- **Integration:** `/predict` with 3 known HAM10000 test images → expected class appears in top-3; corrupt-file and oversize uploads → correct 4xx.
- **Manual rehearsal (the real test):** full Mode A and Mode B run-through on the presenter's own phone before travel day, including laptop sleep/wake behaviour and tunnel restart.

## Repository Layout

```
skin-demo/
├── .specs/            requirements.md · design.md · tasks.md
├── CLAUDE.md          builder governance rules
├── data/ham10000/     README.md · HAM10000_metadata.csv (committed) · image zips (git-ignored)  [C]
├── data.py            release loader + canonical split + loaders
├── train.py           Vast.ai training (split mode + full mode)  [C]
├── evaluate.py        metrics + confusion matrix
├── tests/             pytest suite (unit + integration)  [C]
├── app/
│   ├── main.py        FastAPI server
│   ├── gradcam.py
│   ├── static/index.html
│   ├── model_best.pth   split run — the model metrics.csv is computed from  (scp'd, git-ignored)
│   ├── model_final.pth  100 %-data retrain — the model the demo serves      (scp'd, git-ignored)  [C]
│   ├── metrics.csv · confusion_matrix.png · train_log.csv · train_log_full.csv   (scp'd, committed)
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
- [ ] Single canonical split used everywhere, grouped by lesion, fingerprint `4b4cc59260945104`  [C]
- [ ] Metrics come from `model_best.pth` only; the demo serves `model_final.pth`; evaluator refuses the latter  [C]
- [ ] API response matches the contract exactly
- [ ] Errors fail loudly, never silently
- [ ] No uploaded image ever touches disk
- [ ] akiec sensitivity/specificity reported in metrics.csv
- [ ] Frontend works with no internet (hotspot mode)
