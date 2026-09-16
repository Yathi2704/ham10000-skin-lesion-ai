# Explainable AI-assisted Classification of Actinic Keratosis and Malignant Skin Lesions

**EADV 2026 · ePoster P2699 · 35th European Academy of Dermatology & Venereology Congress, Vienna**

Fellisha Marwein and Yathish Balachander
NIHR Clinical Research Facility, Liverpool University Hospitals NHS Foundation Trust, Liverpool, UK

---

## Overview

Most AI research in skin cancer targets melanoma using opaque "black-box" models, leaving UV-associated premalignant lesions such as actinic keratosis (AK) underexplored — despite ~65% of cutaneous squamous cell carcinomas arising within previously diagnosed AKs.

This repository contains the complete, reproducible pipeline behind our ePoster: three classical machine-learning models (Random Forest, SVM, XGBoost) benchmarked against an **EfficientNet-B0 CNN with Grad-CAM explainability**, for 7-class classification of UV-associated premalignant and malignant skin lesions on a curated subset of HAM10000.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **EfficientNet-B0 (CNN)** | **0.700** | **0.720** | **0.700** | **0.710** |
| Random Forest | 0.493 | 0.493 | 0.493 | 0.489 |
| XGBoost | 0.471 | 0.478 | 0.471 | 0.471 |
| SVM | 0.450 | 0.451 | 0.450 | 0.443 |

*700 images, 7 classes (100 per class), stratified 80:20 train–test split (seed 42), 140-image test set (20 per class). CNN trained 5 epochs, transfer learning from ImageNet.*

**Key findings**

- The CNN outperformed all classical models by ~20 percentage points — learned hierarchical features beat flattened pixel vectors (64×64×3 = 12,288 features).
- The dominant confusion is **AK ↔ basal cell carcinoma**, consistent with their shared dermoscopic morphology.
- **Grad-CAM** attention consistently localised to the lesion rather than background skin — supporting interpretability and clinical auditability.

## Pipeline

1. Load HAM10000 subset from Hugging Face (700 images, 7 classes)
2. Resize 224×224, tensor transform, custom PyTorch `Dataset`, 80:20 split, batches of 16
3. EfficientNet-B0 (ImageNet-pretrained), final layer replaced with 7 outputs; CrossEntropyLoss + Adam (lr 1e-4), 5 epochs
4. Evaluation: accuracy, per-class precision/recall/F1, confusion matrix
5. Classical ML: images resized 64×64, flattened to 12,288 features; Random Forest / SVM (rbf) / XGBoost
6. Grad-CAM on the final feature layer; heatmap overlaid on the original image
7. Model comparison tables, radar/bar/heatmap plots, results saved to CSV

## Running it

Open `Actinic_keratosis.ipynb` in Google Colab (free GPU is sufficient; full run ≈ 15 min):

```bash
pip install datasets transformers timm pytorch-grad-cam xgboost
# if pytorch-grad-cam fails to resolve, use: pip install grad-cam
```

Then run the cells top to bottom. No other setup needed — the dataset downloads automatically from Hugging Face (`pranay-43/HAM10000`).

## Repository contents

| File | Description |
|---|---|
| `Actinic_keratosis.ipynb` | Full pipeline (outputs cleared for size; run top-to-bottom in Colab) |

## Limitations (stated on the poster)

Small single-source dataset (700 images), class-balance of the curated subset, no per-image histopathological confirmation, no clinical metadata, no external validation, 70% accuracy is not clinical grade. This is a proof of concept for **triage/decision support research**, not a diagnostic device.

## Roadmap

- [ ] Retrain on full HAM10000 (10,015 images) with class-weighted loss + augmentation
- [ ] Add ISIC 2019 / BCN20000 / PAD-UFES-20 (smartphone images)
- [ ] External multi-centre, phototype-diverse validation (Fitzpatrick17k, DDI)
- [ ] Vision transformer & foundation model benchmarks
- [ ] Quantitative interpretability (pointing-game / deletion metrics) + reader study
- [ ] Low-cost Raspberry Pi field unit for teledermatology in resource-limited settings

## References

1. Criscione VD, et al. *Cancer.* 2009;115(11):2523-2530.
2. Esteva A, et al. *Nature.* 2017;542(7631):115-118.
3. Selvaraju RR, et al. Grad-CAM. *ICCV* 2017:618-626.
4. Tschandl P, et al. The HAM10000 dataset. *Sci Data.* 2018;5:180161.
5. Tan M, Le QV. EfficientNet. *ICML* 2019:6105-6114.

---

*If you use this pipeline, please cite the ePoster: Marwein F, Balachander Y. Explainable AI-assisted Classification of Actinic Keratosis and Malignant Skin Lesions using a Deep Learning Approach. EADV Congress 2026, Vienna. E-Poster P2699.*
