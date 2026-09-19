# HANDOVER

Builder → reviewer baton. Updated at every ⛔ checkpoint in `.specs/tasks.md`.

---

## ⛔ Checkpoint 1 — Phase A (training pipeline) — 2026-09-19

**Status:** tasks 1–4 done, one commit each (`git log --oneline`). Phase B not started.

| Commit | Task |
|---|---|
| `task 1: project structure` | layout, `requirements.txt`, `.gitignore`, governance files |
| `task 2: data.py canonical split + loaders` | label map, seed-42 split, loaders, tests |
| `task 3: train.py` | EfficientNet-B0, weighted CE, early stopping, checkpoint format, tests |
| `task 4: evaluate.py` | live metrics, `metrics.csv`, `confusion_matrix.png`, tests |

### ⚠️ Decision needed before renting the GPU: the spec's dataset is a 700-image subset

`design.md` names `pranay-43/HAM10000`. I inspected it (it was already in `~/.cache/huggingface`):
**700 images, exactly 100 per class** — a balanced toy subset, not HAM10000 (10 015 images, ~67 % `nv`).
Consequences:

- `requirements.md` model-training-4 assumes the real imbalance; with this data every class weight
  is exactly `1.000` (the code computes it live, so nothing is wrong — there's just nothing to weight).
- The held-out test set is 105 images (15 per class). One akiec image = 6.7 percentage points of
  akiec sensitivity. Poster numbers from this would have very wide error bars.
- A 3-epoch smoke run on the M2 reached val macro-F1 0.56; a full run will do better but it is still
  490 training images for a 7-class problem.

Options (my recommendation is **B**):

- **A — stay with the spec.** Fast (~5 min on a 4090), honest, weak. No code change.
- **B — amend `design.md` to the full HAM10000.** `marmal88/skin_cancer` is partially in the cache
  already (train shards complete, validation/test shards missing) but it is *not* clean: 13 354 rows for
  10 015 unique images (duplicates across its own splits) and long `dx` names
  (`actinic_keratoses`, …). The clean route is the original Harvard Dataverse / Kaggle release
  (`HAM10000_metadata.csv` + two image zips). Either way it is a spec change plus ~20 lines in
  `data.py` (dedupe by `image_id`, alias the `dx` names). `data.py` already refuses to load either
  form silently (`data.py:145` `load_pool`), so nothing can go wrong quietly.
- **C — B, and also group the split by `lesion_id`.** HAM10000 has ~1.4 images per lesion; an
  image-level split (what the spec asks for) leaks lesions between train and test and inflates test
  metrics. Only possible with metadata (B). This is the defensible choice for a poster.

I built exactly what the spec says (option A) and did not improvise. Tell me A/B/C and I'll amend
the spec + `data.py` in one commit before you rent the instance.

### What was built

```
data.py       label map · make_split (seed 42) · load_pool · transform · DataLoaders
train.py      build_model · compute_class_weights · train loop · checkpoint format · predict
evaluate.py   load_checkpoint · compute_metrics · akiec sens/spec · metrics.csv · confusion_matrix.png
tests/        30 tests (test_data.py, test_train.py, test_evaluate.py, conftest.py)
pytest.ini    testpaths = tests
```

Deviations from the `design.md` repo layout, all in service of its Testing Strategy: `tests/`,
`tests/conftest.py`, `pytest.ini`. `train.py` also writes `train_log.csv` (per-epoch losses/F1)
next to the checkpoint, and `metrics.csv` carries four provenance rows (`checkpoint_epoch`,
`checkpoint_val_macro_f1`, `split_fingerprint`, `dataset`) so the CSV alone says which model and
split produced it.

### Reviewer standing checklist → where to look

| Check | Where |
|---|---|
| Every number computed live, zero hardcoded metrics | `evaluate.py:88` `compute_metrics` is pure sklearn over `y_true/y_pred` from `train.predict`; `tests/test_evaluate.py::test_metrics_change_with_predictions`; `grep -nE '\b0\.[0-9]+' data.py train.py evaluate.py` → only ImageNet mean/std, split fractions, lr |
| Single canonical split used everywhere | `data.py:57` `make_split` (two seeded `train_test_split` calls, `data.py:81,87`); `train.py` and `evaluate.py` both call `data.load_split_data`; `evaluate.py:180` refuses a checkpoint whose `split_fingerprint` differs |
| Same seed → identical split | `data.py:104` `split_fingerprint`; `tests/test_data.py::test_same_seed_identical_split`; rows sorted by `image_id` (`data.py:198`) so it's content-stable across machines |
| Per-class counts printed per partition | `data.py:120` `print_split_report` — printed by `data.py`, `train.py`, `evaluate.py` |
| ImageNet normalization, resize 224, batch 32 | `data.py:210` `get_transform`; `tests/test_data.py::test_transform_is_imagenet_normalization` |
| Class-weighted loss (inverse frequency) | `train.py:83` `compute_class_weights` (w = N / (C·n_c), train partition only) → `train.py:217`; weights are printed and stored in the checkpoint |
| Adam lr 1e-4, early stopping on val macro-F1 | `train.py:218`, `train.py:233–265`; `tests/test_train.py::test_early_stopping_triggers` |
| `model_best.pth` loadable without training code | `train.py:102` payload = tensors + primitives only; `tests/test_train.py::test_checkpoint_loads_without_training_code` rebuilds it with torchvision + `weights_only=True` |
| Label map identical everywhere | `data.py` `CLASS_NAMES` is the only definition; stored in the checkpoint; `evaluate.py:42` `load_checkpoint` refuses a mismatch |
| akiec sensitivity/specificity in `metrics.csv` | `evaluate.py:75`; rows `sensitivity,akiec` / `specificity,akiec` (+ `tp/fn/tn/fp`) |
| `metrics.csv` + `confusion_matrix.png` saved | `evaluate.py:124`, `evaluate.py:135`; `tests/test_evaluate.py::test_write_metrics_csv_and_confusion_png` |
| Fail loudly | missing/corrupt/wrong-arch/wrong-label-map checkpoint → `evaluate.py:46–66`; empty loader → `train.py` `predict`; missing class → `compute_class_weights`; unknown dataset columns → `load_pool` |

### How to reproduce my tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # torch 2.14 / py3.14 worked on the M2
.venv/bin/python -m pytest -q                                          # 30 passed in ~50 s (CPU)
.venv/bin/python data.py                                               # split report + fingerprint 32db382e09ab40e4
```

Local smoke run I did (M2 Air, MPS, dataset from cache, ~14 s/epoch) — **not poster numbers**:

```bash
.venv/bin/python train.py --epochs 3 --patience 2 --out /tmp/smoke/model_best.pth
.venv/bin/python evaluate.py --checkpoint /tmp/smoke/model_best.pth
```

Output: `model_best.pth` (16 MB), `train_log.csv`, `metrics.csv`, `confusion_matrix.png`. Two runs
gave bit-identical losses on MPS. The `metrics.csv` format is:

```
metric,class,value
precision,akiec,0.611111
...
sensitivity,akiec,0.733333
specificity,akiec,0.922222
split_fingerprint,meta,32db382e09ab40e4
```

### How to run the real training (user action — Vast.ai, RTX 4090)

```bash
# on the instance (PyTorch image), from the repo root
pip install -r requirements.txt
python data.py                                 # must print fingerprint 32db382e09ab40e4 (for pranay-43)
python train.py --epochs 30 --patience 5 --num-workers 8 --out app/model_best.pth
python evaluate.py --checkpoint app/model_best.pth   # writes app/metrics.csv + app/confusion_matrix.png
```

```bash
# from the M2 Air
scp -P <port> root@<host>:~/skin/app/{model_best.pth,metrics.csv,confusion_matrix.png,train_log.csv} app/
```

`*.pth` is git-ignored (16 MB binary); `metrics.csv`, `confusion_matrix.png`, `train_log.csv` should
be committed — they are the artifacts every poster number traces back to.

### Known issues / notes

- **Dataset size** — see the decision block above. Everything else is secondary to it.
- **No augmentation.** `design.md` specifies `resize 224 → tensor → normalize` and nothing else, so
  train and eval share one transform. Random flips/rotations are free accuracy on dermoscopy images
  (they're orientation-invariant); say the word and I'll add them as a spec amendment.
- **GPU nondeterminism.** The *split* is bit-reproducible (fingerprint). Weights are reproducible on
  MPS in my runs; on CUDA cuDNN may introduce tiny run-to-run differences. The spec guarantees the
  split, not the weights.
- **`--limit-batches` / `--no-pretrained`** on `train.py` exist for offline smoke tests only; never use
  them for the real run.
- **Python 3.14 + torch 2.14** locally. `requirements.txt` has lower bounds only so the Vast.ai image's
  own torch build is kept.

### Next

Phase B (`app/main.py`, `gradcam.py`, integration tests) starts after: (1) this review passes,
(2) the dataset decision, (3) `model_best.pth` is in `app/`. B's server code can be written before the
model exists; tasks 8.1/8.3 (real-image top-3, latency) cannot run without it.
