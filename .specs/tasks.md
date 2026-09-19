# Implementation Plan

Execution rules for the builder: read ALL files in `.specs/` before starting; ONE task at a time; commit after each task (`task N: description`); update `HANDOVER.md` at each checkpoint; stop at checkpoints for review. Coding tasks only — deployment rehearsal and Vast.ai rental are user actions.

## Phase A: Training pipeline (local code, runs later on Vast.ai)

- [x] 1. Set up project structure
  - Create repo layout per design.md, `requirements.txt` (torch, torchvision, datasets, fastapi, uvicorn, pytorch-grad-cam, pillow, numpy, scikit-learn, matplotlib, seaborn, qrcode, python-multipart)
  - _Requirements: model-training — all_

- [x] 2. Implement `data.py` — canonical split
  - [x] 2.1 Load HAM10000, build stratified 70/15/15 split (seed 42), print per-class counts per partition
  - [x] 2.2 DataLoaders with ImageNet normalization (resize 224, batch 32)
  - [x] 2.3 Unit test: same seed → identical split; class counts sum correctly
  - _Requirements: model-training — 1, 2, 3_

- [x] 3. Implement `train.py`
  - [x] 3.1 EfficientNet-B0 with 7-class head, class-weighted CrossEntropyLoss, Adam lr=1e-4
  - [x] 3.2 Training loop with validation macro-F1 each epoch, early stopping, save best as `model_best.pth` (state_dict + label map)
  - _Requirements: model-training — 4, 5_

- [x] 4. Implement `evaluate.py`
  - [x] 4.1 Load `model_best.pth`, run on test set, compute per-class precision/recall/F1 + weighted averages + akiec sensitivity/specificity from live predictions
  - [x] 4.2 Write `metrics.csv` and `confusion_matrix.png`
  - _Requirements: model-training — 6, 7, 8; honesty-reproducibility — 1, 2_

- [x] 4C. Amendment C — original HAM10000, lesion-grouped split, augmentation, evaluate-then-retrain
  - [x] 4C.1 `data.py`: load `data/ham10000/HAM10000_metadata.csv` + image zips, de-dup by `image_id`, `dx` alias table, assert 10 015 images + published class counts
  - [x] 4C.2 Split grouped by `lesion_id`, stratified, seed 42; report images + lesions per partition; fingerprint regenerated (`4b4cc59260945104`)
  - [x] 4C.3 Train-time augmentation (random flips + 90° rotations) on the train partition only; eval transform unchanged
  - [x] 4C.4 `train.py --mode full --epochs-from app/model_best.pth` → `model_final.pth` (100 % of images, fixed epochs, provenance); `evaluate.py` refuses it
  - [x] 4C.5 Spec amended (requirements/design/tasks), Vast.ai instructions + HANDOVER updated, tests updated
  - _Requirements: model-training — 1, 2, 6, 11; honesty-reproducibility — 3, 4_

**⛔ CHECKPOINT 1 — reviewer verifies: split logic, seeding, class weighting, no hardcoded metrics, artifacts saved. [C] Also: release assertions, lesion grouping, augmentation on train only, full-mode provenance, evaluator refusal. Training spend stays frozen until this review of the amended code passes. User then runs on Vast.ai (RTX 4090 spot): `data.py --check-images` → `train.py` → `evaluate.py` → `train.py --mode full --epochs-from app/model_best.pth`, and scp's `model_best.pth`, `model_final.pth`, `metrics.csv`, `confusion_matrix.png`, `train_log*.csv` down to `app/`.**

## Phase B: Inference server (M2 Air)

- [x] 5. Server scaffold (`app/main.py`)
  - [x] 5.1 FastAPI app; load `app/model_final.pth` (`MODEL_PATH` override) at startup on MPS with CPU fallback; refuse to start with explicit error if missing/corrupt  [C]
  - [x] 5.2 `GET /` serves `static/index.html` (placeholder page for now)
  - _Requirements: inference-server — 2, 3_

- [x] 6. `POST /predict` endpoint
  - [x] 6.1 Upload validation: JPEG/PNG only, ≤ 10 MB, corrupt bytes → friendly 4xx
  - [x] 6.2 Preprocess → infer → return top-3 predictions + inference_ms per the JSON contract; fully in-memory
  - _Requirements: inference-server — 1, 4, 5, 6_

- [ ] 7. Grad-CAM module (`app/gradcam.py`)
  - [ ] 7.1 Grad-CAM on `model.features[-1]`, overlay on original image, return base64 PNG in the response
  - _Requirements: inference-server — 1, 6_

- [ ] 8. Integration tests
  - [ ] 8.1 Three known HAM10000 test images → expected class in top-3
  - [ ] 8.2 Corrupt file, wrong type, oversize → correct 4xx responses
  - [ ] 8.3 Latency check: single image < 3 s on the M2 Air
  - _Requirements: inference-server — 1, 4; honesty-reproducibility — 1_

**⛔ CHECKPOINT 2 — reviewer verifies: API contract exact match, error handling fails loudly, nothing written to disk, tests pass.**

## Phase C: Frontend + demo tooling

- [ ] 9. Mobile frontend (`app/static/index.html`)
  - [ ] 9.1 Single-file page: camera capture + gallery upload, local preview, spinner
  - [ ] 9.2 Result card: overlay image, top-3 probability bars with akiec highlighted, inference time, disclaimer footer
  - [ ] 9.3 No external CDN or npm — must work with zero internet
  - _Requirements: mobile-frontend — 1, 2, 5_

- [ ] 10. Demo tooling
  - [ ] 10.1 `make_qr.py` — URL → QR PNG
  - [ ] 10.2 `run.sh` — uvicorn on 0.0.0.0:8000, prints LAN URL; optional cloudflared quick tunnel, prints public URL
  - _Requirements: mobile-frontend — 3, 4_

**⛔ CHECKPOINT 3 — reviewer + user rehearsal: Mode A (hotspot) and Mode B (tunnel) on the user's phone; laptop sleep/wake behaviour; conference-morning checklist written into HANDOVER.md.**
