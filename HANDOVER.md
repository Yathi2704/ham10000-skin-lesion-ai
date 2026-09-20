# HANDOVER

Builder → reviewer baton. Updated at every ⛔ checkpoint in `.specs/tasks.md`.

---

## The real run — Vast.ai, 2026-09-20 (artifacts committed in `82bacb3`)

RTX 4090 · torch 2.11.0+cu128 · code `0d1a108` · 14 min of GPU time end to end · full log in
`app/vast_run.log`. Everything below is copied from the artifacts, not typed in.

```
data.py --check-images   → images=10015 lesions=7470 images_verified=True  fingerprint=4b4cc59260945104
train.py (split)         → 27 epochs, early stop (patience 5), best val macro-F1 0.7273 @ epoch 22, ~13 s/epoch
evaluate.py              → app/metrics.csv + app/confusion_matrix.png   (1 502 test images, 1 121 unseen lesions)
train.py --mode full     → 22 epochs on 10 015 images → app/model_final.pth (derived_from: epoch 22, 0.7273, 4b4cc5…)
evaluate.py model_final  → refused: trained_on='all'
```

### Held-out test metrics (`app/metrics.csv`, computed by `evaluate.py` from `model_best.pth`)

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| akiec | 0.644 | 0.731 | 0.685 | 52 |
| bcc | 0.607 | 0.761 | 0.675 | 71 |
| bkl | 0.699 | 0.653 | 0.675 | 167 |
| df | 0.389 | 0.700 | 0.500 | 20 |
| mel | 0.565 | 0.575 | 0.570 | 167 |
| nv | 0.912 | 0.881 | 0.897 | 1004 |
| vasc | 0.682 | 0.714 | 0.698 | 21 |
| macro avg | 0.642 | 0.716 | 0.671 | |
| weighted avg | 0.816 | 0.806 | 0.810 | |

**accuracy 0.806** · **akiec sensitivity 0.731** (38 / 52) · **akiec specificity 0.986** (1429 / 1450, 21 false
positives) · confusion matrix: `app/confusion_matrix.png`.

Reading it honestly: the split is lesion-grouped, so these are numbers on *lesions the model never saw*.
The class weighting bought recall on the rare classes (df 0.70, vasc 0.71, akiec 0.73) at the price of
precision on df (20 images — one error = 5 points). The largest confusion is mel ↔ nv (48 + 55), the
classic HAM10000 failure mode; akiec's 14 misses go mostly to bcc/bkl/df, not to nv.

### Files in `app/` now

| file | what | versioned |
|---|---|---|
| `model_best.pth` | split-run model, epoch 22 — the one `metrics.csv` is about | no (`*.pth`), md5 `3ed5da78…` |
| `model_final.pth` | 100 %-data model — what `run.sh` serves | no (`*.pth`), md5 `72a89f4b…` |
| `metrics.csv`, `confusion_matrix.png` | the poster's numbers | yes |
| `train_log.csv`, `train_log_full.csv`, `vast_run.log` | per-epoch curves + the full console log | yes |

Both `.pth` files also exist on the Vast instance until it is destroyed; keep a copy of the two md5s above
with the poster material so any future copy can be checked.

### Test suite with the artifacts in place

`pytest -q` → **79 passed, 0 skipped**. The four previously gated tests are real now: the first test-partition
akiec / mel / nv images each land their true class in the top-3 for *both* models; bad uploads 400/400/413
against the real server; a 12 MP photo through `model_final.pth` on MPS in **368–372 ms wall / 302 ms
`inference_ms`**.

### What's left

1. Destroy the Vast instance (user action; nothing on it is needed any more).
2. Phone rehearsal — Mode A + Mode B — per the conference-morning checklist below (Checkpoint 3).
3. Optional, if the poster wants it: a row-normalised confusion matrix (recall per class) reads better
   than raw counts when nv dominates the colour scale; one flag in `evaluate.py` if asked.

---

## Review closure + real-image smoke run on the M2 — 2026-09-20

**Kimi's verdict: PASS** (`REVIEW.md`, commit `050fa88`). Both Minors closed in `67d4187`; two small
follow-ups the smoke run itself surfaced: `fe3327f` (metadata-only `load_pool` never opens the zips)
and the refused-evaluation `mkdir` fix. Test suite: 77 passed + gated tests, `pytest -m "not heavy"`
for small machines.

### Data on the M2

`data/ham10000/HAM10000_images_part_{1,2}.zip` downloaded from Dataverse (2.7 GB, git-ignored):
md5 `4639bfa7…` / `da43d6cc…` — both match the Dataverse checksums; `zipfile.testzip()` clean;
5 000 + 5 015 JPEGs.

### The chain, on the real release (every step is what the Vast.ai run will do)

```
python data.py --check-images
  data_dir=data/ham10000  images=10015  lesions=7470  images_verified=True
  … fingerprint=4b4cc59260945104                         ← matches the pinned value

python train.py --epochs 2 --patience 1 --limit-batches 60          (MPS, smoke only)
  epoch 1  train_loss 1.8212  val_loss 1.6332  val_macroF1 0.3226  *saved*  73s
  epoch 2  train_loss 1.4108  val_loss 1.3232  val_macroF1 0.3911  *saved*  71s

python evaluate.py --checkpoint …/model_best.pth                     (all 1 502 test images, 22 s)
  accuracy 0.5679 · macro-F1 0.3938 · weighted-F1 0.6089 · akiec sensitivity 0.846 / specificity 0.901
  → metrics.csv (41 rows incl. provenance: checkpoint_file, epoch 2, fingerprint 4b4cc59260945104) + confusion_matrix.png

python train.py --mode full --epochs-from …/model_best.pth --limit-batches 20
  "full mode: 2 epochs taken from model_best.pth" → model_final.pth
  trained_on="all", n_train_images=10015, split_fingerprint=None,
  derived_from={epoch 2, val_macro_f1 0.3911, fingerprint 4b4cc59260945104}

python evaluate.py --checkpoint …/model_final.pth
  RuntimeError: refusing to evaluate …: trained_on='all' … (exit 1, nothing written)
```

**These smoke numbers are not poster numbers** — the model saw 3 840 training images for two
partial epochs. They exist only to prove the zip-reading, split, evaluation and provenance paths
on the real data. The smoke artifacts live outside the repo and were deleted from `app/` after
the tests below so they cannot be mistaken for the trained model.

### Gated integration tests, run once with the smoke checkpoints in `app/`

| Test | Result |
|---|---|
| 8.1 known test images (akiec / mel / nv, read from the zips) vs `model_final.pth` | pass — true class in top-3 for all three |
| 8.1 same vs `model_best.pth` | **fails on the akiec image** (`ISIC_0024511` → mel, bcc, bkl): expected for a 2×60-batch model with val macro-F1 0.39; this is the assertion that becomes real with the Vast.ai model |
| 8.2 bad uploads vs the real server | pass (400 / 400 / 413) |
| 8.3 latency, 12 MP photo, real checkpoint, MPS | **362–389 ms wall, 303 ms `inference_ms`** |
| 8.3 architecture budget | 365 ms |

### Timing, for planning the GPU run

108 batches (60 train + 48 val, zip decode + augmentation on the main thread) took ~72 s on the M2
→ ≈ 0.67 s/batch. A full epoch is 219 train + 48 val batches ≈ 3 min on the M2, so **the whole
30-epoch split run would take ≤ 1.5 h on this laptop** (early stopping will likely end it sooner) and
the full-data retrain another ≤ 1 h. On a 4090 with `--num-workers 8` expect a few minutes per run.
The zips are already on the M2, so a local overnight run is a real alternative to renting.

---

## ⛔ Checkpoints 2 + 3 — Phase B server, Phase C frontend + tooling — 2026-09-19

**Status:** tasks 5–10 done, one commit each. Built ahead of the Checkpoint 1 review at the user's
instruction ("proceed into Phase B/C code; training spend stays frozen until Kimi's review of the
amended code passes"). Nothing here needs the trained model to be *reviewed*; three integration tests
need it to *run* and skip with the exact reason until it exists.

| Commit | Task |
|---|---|
| `task 5: server scaffold` | lifespan model load, refusals, `GET /`, `/health` |
| `task 6: POST /predict` | 10 MB cap, in-memory multipart, magic-byte + PIL validation, top-3 |
| `task 7: Grad-CAM module` | `app/gradcam.py`, exact JSON contract |
| `task 8: integration tests` | known-image top-3, bad uploads, latency (real + architecture) |
| `task 9: mobile frontend` | `app/static/index.html`, verified in a 375×812 browser |
| `task 10: demo tooling` | `make_qr.py`, `run.sh` (Mode A + `--tunnel`) |

### What was built

```
app/__init__.py
app/main.py          pick_device · load_model · build_transform · Predictor(predict) · _MemoryMultipart ·
                     extract_upload · decode_image · read_body_capped · lifespan · GET,HEAD / · GET /health · POST /predict
app/gradcam.py       HeatmapGenerator(heatmap, overlay_png) · png_to_base64
app/static/index.html   single file, no external resources
make_qr.py · run.sh
tests/test_app.py (24) · tests/test_integration.py (5) · tests/test_tooling.py (5)
```

Test suite: **73 passed, 4 skipped** (`pytest -q -rs`); the 4 skips are the real-artifact tests.

### Reviewer standing checklist → where to look

| Check | Where |
|---|---|
| API response matches the contract exactly | `app/main.py:163` `Predictor.predict` returns exactly `{predictions[3]{class,label,probability}, heatmap_png_base64, inference_ms}`; `tests/test_app.py::_assert_contract` asserts the key set is *equal*, order by probability desc, labels from the fixed map, base64 decodes to a PNG, `inference_ms` is an int |
| Model file missing/corrupt → refuse to start, loud | `app/main.py:89` `load_model` raises `ModelLoadError` for missing / unreadable / no state_dict / wrong arch / wrong label map / missing preprocessing metadata; `lifespan` (`app/main.py:284`) prints a banner and re-raises → uvicorn exits; `tests/test_app.py::test_startup_refuses_*` ×3; `run.sh` pre-checks the file too |
| MPS with CPU fallback, no rented GPU | `app/main.py:78` `pick_device` (mps → cuda → cpu, `DEVICE` env override); `/health` reports it |
| Non-JPEG/PNG → 400, > 10 MB → 413, corrupt → 400, inference error → 500 generic | `app/main.py:248` `decode_image` (magic bytes, `PIL.verify`, reload, EXIF transpose); `app/main.py:265` `read_body_capped` (Content-Length *and* mid-stream); `app/main.py:312` `predict` (500 + `log.exception`, message never leaks); `tests/test_app.py::test_predict_rejects_bad_uploads_with_friendly_4xx` (6 cases × 2 encodings), `::test_predict_rejects_oversize_upload_with_413`, `::test_read_body_capped_streams_and_stops_without_content_length`, `::test_predict_500_is_generic_when_inference_breaks` |
| Nothing written to disk | the body is streamed into a `bytearray`; multipart is parsed with python-multipart callbacks into `BytesIO` (`app/main.py:197` `_MemoryMultipart`) — deliberately **not** Starlette's `UploadFile`, which spools > 1 MB to a temp file; Grad-CAM/PNG are `BytesIO`; `tests/test_app.py::test_predict_never_touches_disk` booby-traps `tempfile.*` and `Path.write_*` during a 2 MB upload |
| Label map identical between training code and server | `app/main.py` `CLASS_NAMES`/`CLASS_LABELS` are a deliberate copy (the server imports no training code); `tests/test_app.py::test_label_map_identical_to_training_code`; the checkpoint's map is checked at load |
| Server preprocessing == training eval transform | `app/main.py:126` `build_transform` rebuilds it from the checkpoint's own `image_size`/`normalization`; `tests/test_app.py::test_server_transform_equals_eval_transform` (tensor-equal to `data.get_transform()`) |
| Grad-CAM on `model.features[-1]`, overlay on the original | `app/gradcam.py:28`; overlay downscaled to ≤ 512 px, aspect kept; `tests/test_app.py::test_gradcam_*`, `::test_heatmap_is_for_the_top1_class` (served PNG == direct Grad-CAM of `predictions[0]`) |
| ≤ ~3 s per image on the M2 | measured with a 12 MP JPEG through the real server on MPS: **354 ms wall / 292 ms `inference_ms`** (random-weight checkpoint, same architecture) — `tests/test_integration.py::test_8_3_latency_budget_holds_for_the_architecture`; the real-model variant runs once `app/model_final.pth` exists |
| Frontend works with zero internet | `app/static/index.html`: system fonts, inline SVG, `data:` favicon, plain `fetch('/predict')`; `tests/test_app.py::test_frontend_is_a_single_offline_file` greps for any external reference; `app/static/` contains exactly one file |
| Camera capture + gallery upload, preview, spinner, overlay, top-3 bars (akiec highlighted), inference time, disclaimer | `index.html` — `#camera` (`capture="environment"`), `#gallery`, `#original`, `#spinner`, `#heatmap`, `.row.akiec` (outline + "focus" tag), `#meta`, `<footer>` (always visible, also on the landing view) |
| No metrics in the UI | the page shows live probabilities only; the test asserts none of accuracy/sensitivity/specificity/F1/macro appear in the visible copy |
| One stable URL, QR-encodable; hotspot mode | `run.sh` prints every LAN IPv4 (Mode A) and, with `--tunnel`, the `trycloudflare.com` URL + `qr.png` (Mode B); `make_qr.py` |

### How I verified the frontend (reviewer can repeat)

```bash
MODEL_PATH=/path/to/any/train.py-checkpoint.pth ./run.sh        # e.g. a smoke checkpoint; then open http://localhost:8000
```

Opened in the desktop app's browser at 375×812: picked a canvas-drawn "lesion" JPEG → preview,
spinner, 200 → overlay + three bars + "Inference 294 ms"; picked a `.txt` → red box "Only JPEG or
PNG images are accepted."; stubbed `fetch` with an akiec-first response → outlined akiec row with the
focus tag and 61 / 22 / 9 % bars. Console clean apart from the intentional 400. Server log showed
`GET /` 200 and both `POST /predict` outcomes.

### Known issues / notes (Phase B/C)

- **`inference_ms` = preprocessing + forward + Grad-CAM + PNG**, not upload time. On a hotspot a 3 MB
  photo adds ~0.2–0.5 s of transfer that the phone sees but the number doesn't include.
- **HEIC.** iPhones hand JPEG to `<input type=file>` in almost all cases; if one ever sends HEIC the
  server answers 400 "Only JPEG or PNG" and the page shows it. A client-side canvas re-encode would
  remove that risk (and shrink uploads); not in the spec, so not built — decide at rehearsal.
- **Concurrency.** One image at a time goes through the model + Grad-CAM hooks (`threading.Lock`);
  a second phone waits ~0.3 s. Fine for a poster session.
- **`/health`** exposes model path, device and provenance (epoch, `trained_on`, `trained_at`) —
  no metrics. Useful during rehearsal (`curl http://<ip>:8000/health`).
- **Not in the spec but present:** `GET /health`, `HEAD /`, `tests/test_tooling.py`, the 512 px cap on
  the overlay, EXIF orientation handling. All small; say the word if any should go.

### Conference-morning checklist (rehearsal target for Checkpoint 3)

The night before
- [ ] `git pull`; `ls app/` shows `model_final.pth` (+ `model_best.pth`, `metrics.csv`, `confusion_matrix.png`)
- [ ] `.venv/bin/python -m pytest -q tests/test_app.py tests/test_integration.py` → all green, no skips
- [ ] `./run.sh` → phone on the same Wi-Fi opens `http://<laptop-ip>:8000`, one photo round-trips
- [ ] macOS: System Settings → General → Sharing → Internet Sharing: share from **Ethernet/none** to
      **Wi-Fi**, set network name + WPA3 password; note the laptop IP it gives (usually `192.168.2.1`)
- [ ] `python make_qr.py http://192.168.2.1:8000 --out qr_hotspot.png` → print it (Mode A)
- [ ] `brew install cloudflared`; `./run.sh --tunnel` → `qr.png` with the public URL → print it (Mode B);
      the URL changes every restart — print in the morning if the laptop restarts
- [ ] Energy: Battery → Options → **Prevent automatic sleeping on power adapter**; lid-closed = sleep, so
      keep it open; a display-sleep is fine, the server keeps running
- [ ] Charger, USB-C hub, the three printed QR sheets (Mode A, Mode B, blank spare)

At the poster (10 min)
- [ ] Laptop on power, lid open, Internet Sharing ON (or the venue Wi-Fi if it proved reliable)
- [ ] `./run.sh` (or `./run.sh --tunnel`) in a terminal you keep visible — the log shows every request
- [ ] Own phone: join the hotspot, scan the QR, run one photo → overlay + bars appear
- [ ] `curl http://localhost:8000/health` — device says `mps`
- [ ] If the venue blocks the tunnel: tell attendees to join the hotspot (Mode A works with zero internet)

If it breaks
- server refuses to start → the banner names the exact reason (model path / corrupt / label map)
- phone says "Cannot reach the demo server" → wrong network; re-join the hotspot, re-scan
- 400/413 on the phone → not a JPEG/PNG or > 10 MB; use the camera button instead of the gallery
- laptop slept → reopen, `./run.sh` again; Mode B needs a fresh QR (URL changed)

---

## ⛔ Checkpoint 1 (amended C) — Phase A training pipeline — 2026-09-19

**Status:** tasks 1–4 plus amendment 4C done, one commit each (`git log --oneline`).
**Training spend is frozen until this review passes.**

| Commit | Task |
|---|---|
| `task 1: project structure` | layout, `requirements.txt`, `.gitignore`, governance files |
| `task 2: data.py canonical split + loaders` | label map, seed-42 split, loaders, tests |
| `task 3: train.py` | EfficientNet-B0, weighted CE, early stopping, checkpoint format, tests |
| `task 4: evaluate.py` | live metrics, `metrics.csv`, `confusion_matrix.png`, tests |
| `checkpoint 1: HANDOVER.md` | first handover (flagged the 700-image dataset) |
| `amendment C: …` | original HAM10000, lesion-grouped split, augmentation, evaluate-then-retrain, spec amended |

### The dataset decision (resolved: option C)

The HF mirror in the original design (`pranay-43/HAM10000`) is a 700-image balanced subset. Amendment C
replaces it with the **original release** (Harvard Dataverse doi:10.7910/DVN/DBW86T). The metadata CSV
is committed at `data/ham10000/HAM10000_metadata.csv`; the two image zips (2.7 GB) are git-ignored and
downloaded with the `curl` lines in `data/ham10000/README.md`.

Verified live from the committed CSV (`python data.py`):

```
data_dir=data/ham10000  images=10015  lesions=7470  images_verified=False
class      train     val    test   total
akiec        222      53      52     327
bcc          361      82      71     514
bkl          772     160     167    1099
df            71      24      20     115
mel          773     173     167    1113
nv          4683    1018    1004    6705
vasc          99      22      21     142
images      6981    1532    1502   10015
lesions     5229    1120    1121    7470
seed=42  fractions=(0.7, 0.15, 0.15)  grouped_by=lesion_id  fingerprint=4b4cc59260945104
```

Lesion-level fractions are exactly 70/15/15; image-level are 69.7/15.3/15.0 because images follow
their lesion. **Canonical fingerprint: `4b4cc59260945104`** (was `32db382e09ab40e4` for the subset).

### Evaluate-then-retrain chain (requirements model-training-11, honesty-4)

```
train.py --mode split  ──►  app/model_best.pth   ──►  evaluate.py  ──►  app/metrics.csv + confusion_matrix.png
        (train partition,      trained_on="train"        (test partition,        ← every poster / demo number
         early stop on val)    split_fingerprint=…        refuses anything else)

train.py --mode full --epochs-from app/model_best.pth  ──►  app/model_final.pth   ──►  served by app/main.py
        (100 % of images, fixed epoch count = best epoch     trained_on="all"
         of the split run, no validation, no metrics)        derived_from={model_best epoch, val F1, fingerprint}
```

`model_final.pth` has seen the test images, so it must never be scored: `evaluate.py:183` refuses any
checkpoint with `trained_on != "train"`, and `train.py:360` refuses to seed a full run from anything but
a split-run checkpoint. The poster cites `metrics.csv`; the phone talks to `model_final.pth`.

### What was built

```
data/ham10000/  README.md (download commands) · HAM10000_metadata.csv (committed, 690 KB)
data.py         canonical_dx · FileImageStore (reads JPEGs from the zips) · Pool · load_metadata (de-dup) ·
                verify_release · load_pool · make_split (grouped) · fingerprint · get_transform(train) · loaders
train.py        build_model · compute_class_weights · train (split) · train_full · epochs_from_checkpoint · predict
evaluate.py     load_checkpoint · compute_metrics · akiec sens/spec · metrics.csv · confusion_matrix.png · run
tests/          43 tests (test_data.py, test_train.py, test_evaluate.py, conftest.py)
```

### Reviewer standing checklist → where to look

| Check | Where |
|---|---|
| Full release, de-duplicated, published counts asserted | `data.py:192` `load_metadata` (exact duplicates dropped, conflicting ones raise), `data.py:224` `verify_release` (10 015 + per-class counts), called by `load_pool` (`data.py:244`); `tests/test_data.py::test_verify_release_rejects_subsets`, `::test_real_metadata_is_the_published_release_and_split_is_stable` |
| `dx` aliasing to the canonical map | `data.py:85` `canonical_dx` + `DX_ALIASES`; unknown spellings raise |
| Single canonical split, grouped by lesion, stratified, seed 42 | `data.py:265` `make_split` (two seeded `train_test_split` calls over *lesions*, `data.py:291,293`; images follow their lesion); `train.py` and `evaluate.py` both call `data.load_split_data`; `tests/test_data.py::test_every_lesion_in_exactly_one_partition`, `::test_split_is_stratified_70_15_15_at_lesion_level` |
| Same seed → identical split (fingerprint) | `data.py:309` `split_fingerprint` over image ids (row-order independent, `::test_split_is_independent_of_row_order`); pinned to `4b4cc59260945104` in `tests/test_data.py`; stored in `model_best.pth` (`train.py:235`); `evaluate.py:187–190` refuses a mismatch |
| Per-class counts printed per partition | `data.py:319` `print_split_report` (images and lesions) |
| Eval transform = resize 224 → tensor → ImageNet normalize | `data.py:348` `get_transform(train=False)`; `::test_eval_transform_is_imagenet_normalization_and_deterministic` |
| Augmentation on the train partition only | `data.py:356` (flips + `RandomRotate90`, exact transposes); `build_dataloaders` passes `train=(part == "train")`; `::test_train_transform_augments_but_keeps_pixels` |
| Class-weighted loss (inverse frequency) | `train.py:95` `compute_class_weights` (w = N / (C·n_c)); split mode uses the train partition, full mode all labels; weights printed and stored in the checkpoint |
| Adam lr 1e-4, early stopping on val macro-F1 | `train.py:206` `train`; `tests/test_train.py::test_early_stopping_triggers` |
| Full-data retrain with provenance, never evaluated | `train.py:290` `train_full` (`trained_on="all"`, `split_fingerprint=None`, `derived_from`), `train.py:360` `epochs_from_checkpoint`; `evaluate.py:183` refusal; `tests/test_train.py::test_train_full_saves_demo_model_with_provenance`, `tests/test_evaluate.py::test_run_refuses_the_full_data_demo_model` |
| Checkpoints load without training code | `train.py:113` payload = tensors + primitives; `tests/test_train.py::test_checkpoint_loads_without_training_code` (torchvision + `weights_only=True`) |
| Label map identical everywhere | `data.py` `CLASS_NAMES` is the only definition; stored in every checkpoint; `evaluate.py:45` refuses a mismatch |
| Every number computed live, zero hardcoded metrics | `evaluate.py:91` `compute_metrics` is sklearn over live `y_true/y_pred`; `tests/test_evaluate.py::test_metrics_change_with_predictions`; `grep -nE '\b0\.[0-9]+' data.py train.py evaluate.py` → only ImageNet mean/std, split fractions, lr |
| akiec sensitivity/specificity in `metrics.csv` | `evaluate.py:78`; rows `sensitivity,akiec` / `specificity,akiec` (+ `tp/fn/tn/fp`) |
| `metrics.csv` + `confusion_matrix.png` saved | `evaluate.py:127`, `evaluate.py:138`; provenance rows `checkpoint_file`, `checkpoint_epoch`, `split_fingerprint`, `dataset` |
| Fail loudly | missing CSV / columns / conflicting duplicates / unknown dx / not-the-release / missing images → `data.py`; missing/corrupt/wrong-arch/wrong-label-map checkpoint → `evaluate.py:45–69`; full model → refused; empty loader → `train.predict` |

### How to reproduce my tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # torch 2.14 / py3.14 on the M2
.venv/bin/python -m pytest -q          # 43 passed in ~80 s (CPU)
.venv/bin/python data.py               # the split report above, fingerprint 4b4cc59260945104
```

Not yet exercised on the real images: the M2 has only the metadata CSV. The image path is covered by
`tests/test_data.py::test_file_image_store_reads_zips_and_loose_files` (synthetic zip) and the loop by
the in-memory `tiny_pool`. An earlier end-to-end smoke run (700-image subset, MPS, 3 epochs ≈ 14 s/epoch)
proved train → evaluate → artifacts on this machine; the code path is unchanged apart from the loader.
Downloading the zips to the M2 (2.7 GB) would let me run a real-image smoke pass before the GPU is
rented — say so if you want that.

### How to run the real training (user action — Vast.ai, RTX 4090, after review)

```bash
# on the instance (PyTorch image), from the repo root
pip install -r requirements.txt
(cd data/ham10000 && \
  curl -L -o HAM10000_images_part_1.zip "https://dataverse.harvard.edu/api/access/datafile/3172585" && \
  curl -L -o HAM10000_images_part_2.zip "https://dataverse.harvard.edu/api/access/datafile/3172584")
python data.py --check-images                       # must print 10015 images + fingerprint 4b4cc59260945104
python train.py --epochs 30 --patience 5 --num-workers 8          # → app/model_best.pth, app/train_log.csv
python evaluate.py                                                 # → app/metrics.csv, app/confusion_matrix.png
python train.py --mode full --epochs-from app/model_best.pth --num-workers 8   # → app/model_final.pth, app/train_log_full.csv
```

Budget: ~10 k images × 224 px on a 4090 ≈ 1–2 min/epoch → both runs well under an hour.

```bash
# from the M2 Air
scp -P <port> root@<host>:~/skin/app/{model_best.pth,model_final.pth,metrics.csv,confusion_matrix.png,train_log.csv,train_log_full.csv} app/
git add app/metrics.csv app/confusion_matrix.png app/train_log.csv app/train_log_full.csv && git commit -m "artifacts: Vast.ai run"
```

`*.pth` is git-ignored (16 MB each); the CSV/PNG artifacts are what every poster number traces back to.

### Known issues / notes

- **Image-level fractions are approximate** (69.7/15.3/15.0) — inherent to grouping; the report prints
  the exact counts, and `metrics.csv` carries `support` per class.
- **GPU nondeterminism.** The split is bit-reproducible (fingerprint). Weights were reproducible on MPS;
  on CUDA cuDNN may introduce tiny run-to-run differences. The spec guarantees the split, not the weights.
- **`--limit-batches` / `--no-pretrained`** on `train.py` exist for offline smoke tests only.
- **Python 3.14 + torch 2.14** locally; `requirements.txt` has lower bounds only so the Vast.ai image's own
  torch build is kept. `datasets` was dropped (no longer used); `pandas` added.
