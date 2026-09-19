# Project Requirements Document

## Introduction

A mobile-demoable skin-lesion classifier to accompany an EADV e-poster on AI skin-lesion classification (actinic keratosis focus). A phone browser captures or uploads a dermoscopic image; a server on an M2 MacBook Air runs an EfficientNet-B0 trained on HAM10000 and returns 7-class probabilities plus a Grad-CAM heatmap overlay. Training happens on a rented Vast.ai GPU; inference and demo happen entirely on the laptop.

This document is the contract. Two AI agents build and review against it (see design.md — Dual-Agent Protocol). Every number shown in the demo or on the poster must trace back to a saved artifact.

## Features

### Feature: model-training

**User Story:** As a presenter, I want a properly trained, saved model file, so the demo runs on real predictions.

#### Acceptance Criteria

1. WHEN training runs THEN the system SHALL load HAM10000 and create one stratified train/val/test split (70/15/15) with fixed seed 42.
2. IF any model or evaluation consumes data THEN the system SHALL use that same canonical split — no per-model re-splitting.
3. WHEN the split is created THEN the system SHALL print per-class counts for each partition.
4. WHEN the CNN trains THEN the system SHALL use ImageNet normalization, class-weighted loss (HAM10000 is ~67% `nv`), and early stopping on validation macro-F1.
5. WHEN training completes THEN the system SHALL export the model as a single weights file (`model_best.pth`) loadable without the training code.
6. WHEN evaluation runs THEN the system SHALL derive all reported metrics from live predictions — no hardcoded values anywhere in the codebase.
7. WHEN evaluation runs THEN the system SHALL report per-class precision/recall/F1, weighted averages, and akiec-specific sensitivity and specificity.
8. WHEN evaluation completes THEN the system SHALL save `metrics.csv` and `confusion_matrix.png` as files.

### Feature: inference-server

**User Story:** As a presenter, I want a local server on the M2 Air, so inference works with no cloud dependency at the venue.

#### Acceptance Criteria

1. WHEN the server receives a valid image via POST /predict THEN it SHALL respond within ~3 seconds with top-3 class probabilities and a Grad-CAM overlay image.
2. WHEN the server starts THEN it SHALL run on Apple Silicon (MPS if available, CPU fallback) with no rented GPU required.
3. WHEN the server starts AND the model file is missing or unloadable THEN the system SHALL refuse to start with a loud, explicit error.
4. IF an uploaded file is unreadable, not JPEG/PNG, or over 10 MB THEN the system SHALL return a friendly 4xx error, never a crash or stack trace.
5. WHEN any image is processed THEN the system SHALL handle it fully in memory — no uploaded photo is ever written to disk.
6. WHEN /predict succeeds THEN the response SHALL match the JSON contract in design.md exactly.

### Feature: mobile-frontend

**User Story:** As a conference attendee, I want to scan a QR code and try the model on my own phone, so the poster is interactive.

#### Acceptance Criteria

1. WHEN the page loads on a phone THEN it SHALL offer camera capture (`capture="environment"`) and gallery upload.
2. WHEN a result returns THEN the page SHALL display the uploaded image, the Grad-CAM overlay, top-3 classes with probability bars (akiec highlighted), and a visible "research demo — not a diagnostic device" disclaimer.
3. WHEN on the conference floor THEN the demo SHALL be reachable via one stable URL encodable as a QR code.
4. IF venue Wi-Fi is unavailable or hostile THEN the system SHALL support a laptop-hotspot mode requiring zero internet (phone joins the M2 Air's hotspot).
5. WHEN the frontend is built THEN it SHALL be a single static HTML file with no build step and no npm dependencies.

### Feature: honesty-reproducibility

**User Story:** As a second author on the poster, I want every claim defensible, so a reviewer asking "show me" always gets an artifact.

#### Acceptance Criteria

1. WHEN anything is displayed — demo UI, poster figure, or metrics table — every number SHALL trace back to a saved artifact (weights, metrics.csv, or figure file).
2. WHEN the repo is inspected THEN there SHALL be no hardcoded accuracy/precision/recall/F1 values anywhere.
3. WHEN training is re-run with the same seed THEN the split SHALL be identical.

## Explicitly Out of Scope

- Skin-vs-not-skin image detection (a second model; corrupt-file handling only)
- User accounts, image history, or any persistence of uploads
- Native mobile app (web only)
- Multi-image batch prediction
- Re-running the classical ML baselines (RF/SVM/XGBoost) — poster figures are already final; the demo showcases the CNN only
