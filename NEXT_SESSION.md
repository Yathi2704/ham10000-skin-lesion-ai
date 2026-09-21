# Next-session brief — read this before doing anything

For a fresh Claude Code (or human) session continuing this project. Facts only, as of
**2026-09-21**. If something is not here, in `HANDOVER.md`, `REVIEW.md` or the code, it does not exist —
do not assume it.

## 1. Where things are

| Thing | Location / value |
|---|---|
| Repo | `/Users/yathish/skin` — branch `main`, remote `origin` = `git@github.com:Yathi2704/ham10000-skin-lesion-ai.git`, in sync |
| **Open the session here** | `/Users/yathish/skin`. Earlier sessions were opened in empty sibling folders (`/Users/yathish/skin ` with a trailing space, `/Users/yathish/app for confrence`) — nothing lives there |
| Python | `.venv/bin/python` (3.14, torch 2.14, MPS works). Tests: `.venv/bin/python -m pytest -q` |
| Data | `data/ham10000/` — `HAM10000_metadata.csv` (committed) + both image zips (2.7 GB, git-ignored, md5-verified, complete) |
| Models | `app/model_best.pth` (split run, epoch 22 — every metric comes from it) · `app/model_final.pth` (100 % retrain, 22 epochs — the demo serves it). Git-ignored; md5s in `HANDOVER.md` |
| GitHub auth | `~/.ssh/id_ed25519` is registered with GitHub (`Yathi2704`). Push with plain `git push origin main` |
| Poster | `~/Documents/AI_skinLesions_v2.pptx` — EADV 2026 e-poster, **submitted and frozen**, based on the 700-image HF subset (CNN 70.0 %). The demo is the "since submission" update |
| Vast.ai | Instance `root@23.158.136.85 -p 31165` was used for the run on 2026-09-20; should be destroyed by the user — do not assume it exists |

## 2. State

- **Code-complete and reviewed.** `REVIEW.md` holds two Kimi reviews, both PASS (checkpoints 1–3 and the
  artifact audit). All Minor findings closed.
- **Trained.** Vast.ai RTX 4090, 14 min. Artifacts committed in `82bacb3`: `app/metrics.csv`,
  `app/confusion_matrix.png`, `app/train_log.csv`, `app/train_log_full.csv`, `app/vast_run.log`.
- **Tests:** 79 passed, 0 skipped with the artifacts in place (`-m "not heavy"` on small machines).
- **GitHub updated** (`499d1c8`): pipeline at the root, poster-era notebooks + README under `poster/`,
  new top-level `README.md`. The poster's printed QR points at this repo.
- **Rehearsal so far:** Mode A on the same Wi-Fi verified on the user's Android phone (Brave): camera
  capture, Grad-CAM overlay, 342 ms. **Not yet done:** macOS Internet Sharing hotspot test, Mode B
  (`brew install cloudflared` + `./run.sh --tunnel`, cellular-only test), laptop sleep/wake check,
  printing the QR sheets. Checklist: `HANDOVER.md` → "Conference-morning checklist".

## 3. Numbers — cite `app/metrics.csv`, never retype from memory

Held-out test: 1 502 images / 1 121 unseen lesions, split fingerprint `4b4cc59260945104`.
Accuracy 0.806 · macro-F1 0.671 · weighted-F1 0.810 · akiec sensitivity 0.731 (38/52) ·
akiec specificity 0.986 (1429/1450). Per class in the CSV and in `README.md`.
Poster numbers (different experiment, 700-image subset, 140 test images): CNN 0.700, RF 0.493,
XGB 0.471, SVM 0.450 — **not comparable** with the line above; say so whenever both appear.

## 4. How to run

```bash
cd /Users/yathish/skin
./run.sh                    # demo on the LAN (prints the URL); MODEL_PATH / PORT / DEVICE env overrides
./run.sh --tunnel           # + cloudflared quick tunnel → public URL + qr.png (needs brew install cloudflared)
.venv/bin/python make_qr.py http://<ip>:8000 --out qr.png
.venv/bin/python data.py --check-images        # must print fingerprint 4b4cc59260945104
.venv/bin/python evaluate.py                   # re-derives app/metrics.csv from app/model_best.pth
.venv/bin/python -m pytest -q                  # 79 tests
```

## 5. Rules that still apply

- `CLAUDE.md` (builder rules) and `.specs/` (the contract) govern changes; amendments are marked `[C]`.
- No hardcoded metrics anywhere; every displayed number traces to an artifact.
- One canonical split (`data.make_split`, seed 42, grouped by `lesion_id`); `evaluate.py` refuses a
  checkpoint with a different fingerprint or with `trained_on != "train"` — by design, do not "fix".
- Uploads never touch disk: multipart is parsed in memory on purpose (Starlette's `UploadFile` spools
  > 1 MB to a temp file) — do not simplify it away.
- Commit per task with a descriptive message; the `Co-Authored-By: Claude …` trailer is the current
  convention (user has not asked to drop it).
- Never leave smoke/test checkpoints in `app/`; never start paid GPU runs — the user does that.
- Publishing (push, releases, description/licence changes) only when the user asks.

## 6. Gotchas already paid for

- `pranay-43/HAM10000` on Hugging Face is a **700-image balanced subset**, not HAM10000. The pipeline
  uses the original Dataverse release (`data/ham10000/README.md` has the `curl` lines; the API serves
  the zips without a click-through).
- `marmal88/skin_cancer` has duplicate image ids and long `dx` names — rejected.
- On the Vast image, `uv pip install` needed `--native-tls`; `/venv/main` already had torch+cu128.
- macOS blocks reading files inside the WhatsApp container; copy to `~/Documents` first.
- The frontend test greps for external references — keep `app/static/index.html` a single offline file.
- The two 12 MP latency tests are marked `heavy` (≈300 MB peak).

## 7. Open decisions for the user

1. Poster number: the file says **P2071**, the GitHub description/old README say **P2699**.
2. Licence for the public repo (none yet; MIT suggested).
3. Keep or drop the `Co-Authored-By` trailer on future commits.
4. Destroy the Vast instance if not already done.
5. Optional: row-normalised confusion matrix for talks; client-side JPEG re-encode in the page if any
   attendee phone ever sends HEIC (none seen yet).
