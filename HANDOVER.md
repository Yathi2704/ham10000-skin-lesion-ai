# HANDOVER

Builder → reviewer baton. Updated at every ⛔ checkpoint in `.specs/tasks.md`.

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
