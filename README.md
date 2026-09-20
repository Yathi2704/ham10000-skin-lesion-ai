# Explainable AI for Actinic Keratosis and Malignant Skin Lesion Classification

Code, artifacts and live demo behind the EADV Congress 2026 e-poster *"Explainable AI-assisted
Classification of Actinic Keratosis and Malignant Skin Lesions using a Deep Learning Approach"* —
Fellisha Marwein and Yathish Balachander, NIHR Clinical Research Facility, Liverpool University
Hospitals NHS Foundation Trust.

**Research demo — not a diagnostic device.** Nothing here is fit for clinical use.

This repository holds two things that must not be confused:

| | Where | Data | What it is |
|---|---|---|---|
| **Poster experiment** | [`poster/`](poster/) | 700-image balanced HAM10000 subset, 140 test images | The Colab notebooks and README the e-poster was made from, kept unchanged for provenance (with the corrections recorded there) |
| **Demo pipeline** | repo root | Full HAM10000 release, 10 015 images, lesion-grouped split | EfficientNet-B0 + Grad-CAM retrained after submission, served to attendees' phones at the poster |

The two sets of numbers are **not comparable** — different data, different split protocol. The poster
reports the first; the phone demo runs the second.

## Demo pipeline — results

Held-out test partition: 1 502 images from 1 121 lesions the model never saw
(seed-42, stratified by class, grouped by `lesion_id`; split fingerprint `4b4cc59260945104`).
Every number below is read from [`app/metrics.csv`](app/metrics.csv), which `evaluate.py` computes
live from the model's predictions — nothing in this repository hardcodes a metric.

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| akiec — actinic keratosis | 0.644 | 0.731 | 0.685 | 52 |
| bcc — basal cell carcinoma | 0.607 | 0.761 | 0.675 | 71 |
| bkl — benign keratosis | 0.699 | 0.653 | 0.675 | 167 |
| df — dermatofibroma | 0.389 | 0.700 | 0.500 | 20 |
| mel — melanoma | 0.565 | 0.575 | 0.570 | 167 |
| nv — melanocytic nevus | 0.912 | 0.881 | 0.897 | 1004 |
| vasc — vascular lesion | 0.682 | 0.714 | 0.698 | 21 |
| macro average | 0.642 | 0.716 | 0.671 | |
| weighted average | 0.816 | 0.806 | 0.810 | |

**Accuracy 0.806 · actinic-keratosis sensitivity 0.731 · specificity 0.986.**
Confusion matrix: [`app/confusion_matrix.png`](app/confusion_matrix.png). The largest confusion is
melanoma ↔ nevus (48 + 55 images) — the well-known HAM10000 failure mode.

## How the pipeline works

```
data.py       original HAM10000 release (metadata CSV + image zips) → de-dup by image_id → published
              counts asserted → ONE stratified, lesion-grouped 70/15/15 split (seed 42) → loaders
train.py      --mode split  train partition, class-weighted CE, flips + 90° rotations, Adam 1e-4,
                            early stopping on validation macro-F1  → app/model_best.pth
              --mode full   100 % of the images for the epoch count the split run chose → app/model_final.pth
evaluate.py   model_best.pth on the held-out test partition → metrics.csv + confusion_matrix.png
              (refuses model_final.pth: it has seen the test images)
app/main.py   FastAPI server: loads model_final.pth on Apple-Silicon MPS, POST /predict → top-3 +
              Grad-CAM overlay, fully in memory, exact JSON contract, friendly 4xx on bad uploads
app/static/   one-file mobile page (camera / gallery → overlay + probability bars), no internet needed
```

The honesty chain is enforced by code, not convention: `metrics.csv` names the checkpoint and split it
came from; `model_final.pth` records which split checkpoint chose its epoch count; `evaluate.py` refuses
any checkpoint that was trained on the test images or on a different split.

### Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# images: see data/ham10000/README.md (2.7 GB from Harvard Dataverse; the metadata CSV is committed)
.venv/bin/python data.py --check-images                       # prints the split + fingerprint 4b4cc59260945104
.venv/bin/python train.py --epochs 30 --patience 5            # → app/model_best.pth   (~6 min on an RTX 4090)
.venv/bin/python evaluate.py                                  # → app/metrics.csv, app/confusion_matrix.png
.venv/bin/python train.py --mode full --epochs-from app/model_best.pth   # → app/model_final.pth
.venv/bin/python -m pytest -q                                 # 79 tests (use -m "not heavy" on small machines)
./run.sh                                                      # serve the demo on the LAN; --tunnel for a public URL
```

The trained weights (16 MB each) are not in git; the run that produced them is logged in
[`app/vast_run.log`](app/vast_run.log) and [`app/train_log.csv`](app/train_log.csv).

## Repository layout

```
poster/            notebooks + README behind the e-poster (700-image subset) — unchanged
.specs/            requirements, design, tasks: the contract the pipeline was built and reviewed against
CLAUDE.md          builder rules · HANDOVER.md builder → reviewer notes · REVIEW.md reviewer findings
data/ham10000/     HAM10000_metadata.csv (committed) + download instructions for the image zips
data.py · train.py · evaluate.py · app/ · tests/ · make_qr.py · run.sh · requirements.txt
```

The pipeline was built spec-first with two AI agents — one building against `.specs/`, one reviewing
diffs against it — with a human deciding scope at each checkpoint. `HANDOVER.md` and `REVIEW.md` are
that trail.

## Data and attribution

Tschandl P, Rosendahl C, Kittler H. *The HAM10000 dataset, a large collection of multi-source
dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161 (2018).
Harvard Dataverse doi:10.7910/DVN/DBW86T, licence CC BY-NC 4.0. The poster experiment used the
[`pranay-43/HAM10000`](https://huggingface.co/datasets/pranay-43/HAM10000) 700-image subset.

## Limitations

- Single public dataset; no external validation; central-European dermoscopy, little Fitzpatrick V–VI skin.
- One held-out split, no cross-validation; the small classes (df 20, vasc 21 test images) have wide error bars.
- Grad-CAM is qualitative — no pointing-game or reader-study validation.
- The demo model is trained on 100 % of the data and is therefore never scored; its behaviour is
  inferred from the split-run model that shares its architecture, data and epoch count.

## Licence

No licence file yet — default copyright applies until one is added.
