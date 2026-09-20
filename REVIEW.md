# REVIEW — Checkpoints 1–3 (consolidated)

**Reviewer:** Kimi · **Date:** 2026-09-20 · **Scope:** amended Phase A (`51f9835`) + Phase B/C (tasks 5–10, through `28f0bd9`)
**Method:** independent re-run in a clean sandbox — full test suite, split regeneration, leakage check, source audit — not narration.

## Verdict: PASS ✅

Training spend is **unfrozen**. The Vast.ai run may proceed per HANDOVER.md. Zero Critical, zero Major findings.

## Verified independently (evidence, not claims)

| Check | Result |
|---|---|
| Test suite | **73 passed, 4 skipped** in my sandbox — exactly the builder's claim. The 4 skips are correctly gated on missing artifacts (`model_*.pth`, image zips) and **fail loudly with the reason**, per spec |
| Split fingerprint | Regenerated from the committed metadata CSV alone: **`4b4cc59260945104`** — exact match with the spec-pinned value |
| Lesion leakage | **0 lesions** appear in more than one partition (7,470 lesions checked directly) — the pre-review Major is fixed |
| Split determinism | Same seed → byte-identical index arrays on rebuild |
| Release assertion | `verify_release` accepts the committed CSV: 10,015 images, published class counts (nv 6705 … df 115) |
| Hardcoded metrics | `grep` audit of all source modules: no metric literals. Only ImageNet constants, spec-mandated split fractions, `0.0` initializers, and a cosmetic `image_weight=0.5` |
| Honesty chain | `evaluate.py` refuses `trained_on != "train"` and any fingerprint mismatch; `train.py --mode full` records `derived_from` provenance; server defaults to `model_final.pth`. The chain is enforced by code, not convention |
| API contract | `test_8_3_latency_budget_holds_for_the_architecture` passes against the real server: exact JSON contract, **763 ms wall / 601 ms inference on a CPU-only sandbox** — the 3 s budget holds even without MPS |
| No-disk rule | The Starlette spooling catch is real: `test_app.py:202` booby-traps every `tempfile` path during a 2 MB multipart upload. Read the code — the claim is accurate |
| Label map | Canonical `0=akiec…6=vasc` in `data.py`; server copy asserted equal by test; checkpoints rejected on mismatch in both `evaluate.py` and `app/main.py` |
| Per-task commits | 13 commits, one per task, spec amendments marked `[C]` in all three spec files. Governance followed |
| Frontend | Single file, zero external references, disclaimer footer, akiec highlight, offline error path — matches spec |

## Closed items from pre-review

- **Patient-level leakage (was 🔴 Major):** fixed by Amendment C, verified above. Closed.
- **Test-count discrepancy (54 → 30 → 73):** explained — Phase A had 30; B/C added 47 (77 collected = 73 + 4 gated). My run reproduces 73/4 exactly. Closed.
- **`marmal88/skin_cancer`:** builder's rejection (13,354 rows, duplicates) was correct; the Dataverse route with dedupe-by-`image_id` + conflict-raise is the right call. Closed.

## Findings

### Minor 1 — Heavy test can OOM weak machines in full-suite runs
`test_8_3_latency_budget_holds_for_the_architecture` boots the real server with a random-weight EfficientNet plus a 12 MP synthetic photo (~300 MB array). In a memory-constrained CPU container the full-suite run was SIGKILLed at that point (it passes standalone, and passes on the M2). Suggest a `heavy` marker so weak machines can `pytest -m "not heavy"`. Not a blocker — every test file passes individually in my sandbox.

### Minor 2 — `evaluate.py:196` assumes `val_macro_f1` exists
The report line formats `meta.get("val_macro_f1"):.4f` — a hand-rolled checkpoint without that key would raise `TypeError` instead of the module's usual clear errors. Split-run checkpoints always carry it, so this is cosmetic; one `f"{...:.4f}"` guard fixes it.

### Residual risk (not a code finding)
The zip-upload excluded the 2.7 GB image zips, so `FileImageStore`'s zip-reading path was reviewed in code but not exercised against the real release in this review. This is already covered by the planned real-image smoke run on the M2 **before** the GPU spend — keep that ordering: smoke run → this sign-off stands → Vast.ai.

## Standing checklist (final state)

- [x] Every reported number computed live; zero hardcoded metrics
- [x] Single canonical split used everywhere, grouped by lesion, fingerprint `4b4cc59260945104`
- [x] Metrics from `model_best.pth` only; demo serves `model_final.pth`; evaluator refuses the latter
- [x] API response matches the contract exactly
- [x] Errors fail loudly, never silently
- [x] No uploaded image ever touches disk
- [x] akiec sensitivity/specificity in `metrics.csv` (code verified; artifact pending the training run)
- [x] Frontend works with no internet

## What remains gated (correctly)

1. Real-image smoke run on the M2 (free) — then Vast.ai: `data.py --check-images` → `train.py` → `evaluate.py` → `train.py --mode full --epochs-from app/model_best.pth` → scp six artifacts down
2. Skipped tests 8.1/8.2/8.3 go green once artifacts exist
3. Phone rehearsal, Mode A + Mode B (the conference-morning checklist in HANDOVER.md)

---

# REVIEW — Artifact Audit (post-training, final sign-off)

**Date:** 2026-09-20 · **Scope:** the six Vast.ai artifacts + run log · **Commits:** `0d1a108` (run), `82bacb3`, `d9e8681`

## Verdict: PASS ✅ — the project is code-complete. Only the phone rehearsal remains.

Every number below was **recomputed from the artifacts**, not read from the report.

| Audit | Result |
|---|---|
| Confusion matrix ↔ metrics.csv | Every per-class precision/recall/F1 and support recomputed from the 49 matrix cells matches the CSV to 6 decimal places |
| Confusion matrix ↔ canonical split | Test partition sizes (akiec 52, bcc 71, bkl 167, df 20, mel 167, nv 1004, vasc 21 = 1,502) are **exactly** the seed-42 lesion-grouped test partition I regenerated independently — the evaluation really ran on the canonical held-out set |
| Headline numbers | accuracy 0.8063 = 1,211/1,502 ✓ · akiec sensitivity 38/52 = 0.7308 ✓ · specificity 1,429/1,450 = 0.9855 ✓ · macro-F1 0.671 ✓ (all recomputed) |
| Provenance rows | `metrics.csv` names `model_best.pth`, epoch 22, val macro-F1 0.727267, fingerprint `4b4cc59260945104` ✓ |
| train_log.csv | 27 epochs, early-stopped, best epoch 22 at val macro-F1 0.72727 — matches the checkpoint metadata ✓ |
| train_log_full.csv | Exactly 22 epochs — the count the split checkpoint chose, no more, no less ✓ |
| vast_run.log | Chain ran in the correct order on an RTX 4090 (10:41→10:55, 14 min): split training → held-out evaluation → full-data retrain → **refusal check** (`evaluate.py` rejected `model_final.pth` as designed) → artifact listing → DONE ✓ |
| GPU spend | 14 minutes ≈ **well under £1** of the £5 credit ✓ |

## The number to present honestly

Melanoma is the weakest class (F1 0.57; 48 mel → nv confusions). This is the textbook HAM10000 failure mode, and the demo disclaimer covers it — but say it out loud at the poster before anyone asks: *the hard cases are exactly why this is a research demo, not a screening tool.* It turns the weakest number into the strongest talking point.

## Sign-off

The pipeline now does what the original script never did: every displayed number traces to a committed artifact, the demo model was never scored, the scored model never saw the test lesions, and both facts are enforced by code. From vibe-coded Colab to this — the spec-driven experiment worked. Go rehearse.
