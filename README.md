# Explainable AI for Actinic Keratosis and Malignant Skin Lesion Classification

Code and results behind **e-poster P2699**, *"Explainable AI-assisted Classification of Actinic Keratosis and Malignant Skin Lesions using a Deep Learning Approach"* — EADV Congress 2026, Vienna.

Fellisha Marwein and Yathish Balachander · NIHR Clinical Research Facility, Liverpool University Hospitals NHS Foundation Trust.

**This is a teaching-scale proof of concept, not a diagnostic tool.** It is trained on 700 images from one public dataset, has no external validation, and is not fit for any clinical use.

---

## Corrections — 17 September 2026

Two claims in the first version of this repository were not supported by the notebook and have been fixed. They are recorded here rather than quietly removed.

| Claim in v1 | What the notebook actually showed | Fix |
|---|---|---|
| "Stratified 80:20 split … 20 test images per class" | Only the classical models used a stratified split. The CNN used the Hugging Face splitter, which is not stratified: its test set had 21/24/20/16/23/19/17 images per class, and shared just 35 of 140 images with the classical models' test set. | `Actinic_keratosis_gradcam_fixed.ipynb` puts all four models on one stratified split. CNN metrics are being re-measured. |
| "Grad-CAM attention consistently localised to the lesion" | The Grad-CAM cell failed with `ModuleNotFoundError: pytorch_grad_cam` — the package is published as `grad-cam` — so no heatmap was ever produced. | The install is fixed and the notebook now saves a Grad-CAM panel. No localisation claim is made here until that panel has been produced and counted. |

---

## Data

- **Source:** [`pranay-43/HAM10000`](https://huggingface.co/datasets/pranay-43/HAM10000) on Hugging Face — a 700-image balanced subset of HAM10000 (Tschandl, Rosendahl & Kittler, *Scientific Data*, 2018).
- **Classes (7):** `akiec` (actinic keratosis / intraepithelial carcinoma), `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` — 100 images each.
- **Split:** stratified 80:20, `random_state=42` → 560 training / 140 test, 20 test images per class, identical for all four models.

**Known weakness:** HAM10000 contains multiple photographs of some lesions (its `lesion_id` column). This subset is split by image, not by lesion, so images of one lesion can fall on both sides of the split. Treat the reported scores as an upper bound. The full-dataset retrain groups by `lesion_id`.

## Method

| Model | Input | Configuration |
|---|---|---|
| Random Forest | 64×64 RGB flattened (12,288 features) | 200 trees |
| SVM | same | RBF kernel |
| XGBoost | same | `multi:softmax`, 7 classes |
| EfficientNet-B0 | 224×224 RGB | ImageNet-pretrained, fine-tuned end to end, 5 epochs, Adam `lr=1e-4`, batch 16 |

Explainability: Grad-CAM over the final convolutional block (`model.features[-1]`), explaining the predicted class.

## Results

Classical baselines, on the stratified 140-image test set:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.493 | 0.493 | 0.493 | 0.489 |
| XGBoost | 0.471 | 0.478 | 0.471 | 0.471 |
| SVM | 0.450 | 0.451 | 0.450 | 0.443 |
| EfficientNet-B0 | _re-running_ | _re-running_ | _re-running_ | _re-running_ |

Precision and recall are weighted averages, taken from the notebook's `classification_report` output.

The first submitted run of EfficientNet-B0 scored 0.700 accuracy (0.72 / 0.70 / 0.71 weighted precision / recall / F1) — but on the unstratified test set described above, so it is not directly comparable with the classical rows. It is being re-measured on the shared split and this table will be updated with that result.

### Explainability

Grad-CAM did not run in the originally published notebook. `Actinic_keratosis_gradcam_fixed.ipynb` fixes the install and produces:

- `gradcam_single.png` — one test image, original and heatmap;
- `gradcam_panel.png` — the first two test images of every class, chosen by position rather than by appearance, so failure cases appear alongside successes;
- `gradcam_panel_predictions.csv` — the prediction, confidence and correctness behind each panel image.

Findings will be added here once the panel has been produced and reviewed.

## Reproducing

1. Open `Actinic_keratosis_gradcam_fixed.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run cells 1 and 2 (about 5–10 minutes). Cell 1 downloads the data, trains all four models and evaluates them; cell 2 produces the Grad-CAM panel.

The notebook sets seeds (42) and saves the trained weights as `efficientnet_b0_ham700_seed42.pt`, so the heatmaps can be traced to a specific model. `Actinic_keratosis.ipynb` is the original submitted notebook, kept unchanged for provenance.

## Limitations

- 700 images from a single public dataset; no external validation.
- No per-image histopathological confirmation for this subset (HAM10000 overall reports more than half of its lesions as histopathology-confirmed).
- Split by image rather than by lesion — see above.
- The balanced test set makes the four-model comparison fair and the accuracy optimistic; it is not clinic prevalence.
- HAM10000 is central European dermoscopy and barely represents Fitzpatrick V–VI skin. Nothing here supports any claim about performance in darker skin.
- Classical baselines use raw flattened pixels deliberately, as a floor rather than a tuned competitor.
- Grad-CAM is qualitative. No pointing-game, deletion/insertion or reader-study validation has been done.

## Next steps

1. Full HAM10000 (10,015 images) with lesion-level grouping, class weighting and augmentation.
2. Balanced accuracy and macro F1 as the headline metrics on the natural class distribution.
3. Evaluation on phototype-labelled data (Fitzpatrick17k, DDI) and smartphone images (PAD-UFES-20).
4. Quantitative validation of the explanations against lesion masks.

## Citation

> Marwein F, Balachander Y. *Explainable AI-assisted Classification of Actinic Keratosis and Malignant Skin Lesions using a Deep Learning Approach.* E-poster P2699, EADV Congress 2026, Vienna.

Dataset: Tschandl P, Rosendahl C, Kittler H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161 (2018).

## Licence

No licence file yet — until one is added, default copyright applies and others cannot reuse the code. MIT is the usual choice for work like this and can be added from GitHub's "Add file → Create new file → LICENSE" template picker.