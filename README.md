# Reproducing minutiae clustering in Fingerprint Spoof Buster Lite

MSc reproduction project, University of the Witwatersrand, 2026.
Paper: T. Chugh and A. K. Jain, *Fingerprint Presentation Attack Detection: Generalization and Efficiency*,
arXiv:1812.11574 (ICB 2019).

## The claim

> Clustering fingerprint minutiae with k-means and classifying only one 96×96 patch per cluster centroid
> cuts computation without a significant loss in presentation-attack detection accuracy.

**Source:** Table 2 of the paper. With k = 10 clusters, TDR = 95.3% at FDR = 0.2%, against 95.7% when every
minutia's patch is classified (35 per image on average). Inference time drops from 1,554 ms to 302 ms on a
Redmi Note 4.

**Our research question:** does k = 10 keep TDR @ FDR = 0.2% within one percentage point of the
all-minutiae baseline, while reducing the number of MobileNet evaluations?

## Route: reimplementation

The authors' code is not public, so the whole pipeline is **reimplemented here**. All code in `src/` is
ours (written with AI assistance and reviewed by the author). We use these libraries:
- `tf.keras.applications.MobileNet` for the MobileNet-v1 architecture and its ImageNet weights,
- scikit-learn `KMeans`, and
- OpenCV and scikit-image for image operations.

| File | What it does |
|---|---|
| `src/data.py` | fixed train/val/test split, committed as `results/split.csv` |
| `src/minutiae.py` | minutiae extractor (orientation field, Gabor enhancement, skeleton, crossing number) |
| `src/patches.py` | all / k-means / random patch selection, aligned 96×96 patch cutting |
| `src/train.py` | trains MobileNet-v1 on all minutia patches of the training images, per seed |
| `src/evaluate.py` | fingerprint-level scores for every setting, saved as `results/scores.csv.gz` |
| `src/timing.py` | single-thread CPU latency per pipeline stage, using a quantised TFLite model |
| `src/report.py` | every table, figure and number in the write-up, from the committed results |
| `src/synthetic.py` | synthetic fingerprints for the smoke test and the illustrative figure |

## Differences from the paper (all are reported in the write-up)

1. **Minutiae extractor.** The paper does not name one; the original Spoof Buster used a commercial SDK.
   We wrote our own, so the minutiae count per image differs from the paper's 35. Ours is reported in
   `results/headline.json`.
2. **Centroid alignment.** A k-means centroid has no minutia direction, and the paper does not say how
   centroid patches were aligned. We rotate each patch to the ridge-orientation field at the centroid.
   Ridge direction is known only modulo 180°.
3. **Framework.** We use Keras MobileNet-v1, not TF-Slim, because TF-Slim needs TensorFlow 1.x. The model
   has the same architecture with a 96×96 input. Global average pooling over the final 3×3 map equals
   the paper's 3×3 pooling change.
4. **Training details the paper omits.** We chose ImageNet initialisation, Adam with lr 1e-4, batch 128,
   6 epochs, a class-balanced loss and 0.2 dropout. To fit in Colab RAM, training uses at most 48 patches per
   image. All settings are in `src/config.py`.
5. **Split.** We follow the paper's 1,000 bona fide test images. The paper gives no spoof split for
   Table 2, so we use a stratified 20% of each material. All 12 materials are seen in training, and the
   split is per image, not per subject.
6. **Hardware.** We use a Colab T4 GPU for training and one CPU thread with TFLite for timing. The paper
   timed a Redmi Note 4 phone, so only *relative* latency is compared.
7. **Repeats.** The paper's "5-fold cross-validation" over k-means initialisation is read as 5 k-means
   seeds on one trained model. We add a second training seed.

**Numbers the paper does not report:**
- random-minutiae ablation,
- unweighted vs size-weighted score averaging,
- second training seed,
- TDR @ FDR = 1%,
- bootstrap error on the test set,
- latency split by stage.

## Data and ethics

The data is the MSU Fingerprint Presentation Attack Dataset (MSU-FPAD) and the Precise Biometrics
Spoof-Kit Dataset (PBSKD): 5,743 bona fide and 4,912 spoof images across 12 materials, captured on a
CrossMatch Guardian 200. Access requires signing an academic data agreement with MSU.

Fingerprints are biometric data about real people. For this reason:
- **No images, minutiae or patches are committed.** The minutiae cache lives in `work/`, which is
  git-ignored, because minutiae coordinates are a biometric template.
- Figures use **synthetic** fingerprints only.
- We commit trained weights, per-image scores (anonymous IDs only) and file names.
- *TODO (author): state the licence and consent terms of the agreement you signed, and confirm that
  committing model weights is allowed under it.*

Expected layout:

```
DATA_DIR/bonafide/**/*.bmp            (any of bmp/png/jpg/tif)
DATA_DIR/spoof/<material>/**/*.bmp    one folder per presentation-attack material
```

## Running it

```bash
conda env create -f environment.yml && conda activate spoofbuster-lite-repro

# 1. Every number and figure in the write-up, from a clean checkout (needs the dataset; ~1.5-2 h on a T4)
DATA_DIR=/path/to/data bash run_all.sh

# 2. Without the dataset: rebuild every table, figure and number from the committed results (~1 min, CPU)
bash run_all.sh --report-only

# 3. Smoke test of the whole pipeline on synthetic fingerprints (~3 min, CPU)
bash run_all.sh --smoke
```

The committed checkpoints in `checkpoints/` are reused, so command 1 skips training. Add `--retrain` to
train from scratch. `colab_run.ipynb` runs command 1 on Colab.

`environment.yml` was generated by `scripts/pin_env.py` in the runtime that produced `results/`.

## Outputs mapped to the write-up

| Write-up item | File |
|---|---|
| Table 2 side by side | `results/table_main.tex` |
| Ablations (random, unweighted) | `results/table_ablation.tex` |
| Training seeds | `results/table_seeds.tex` |
| Latency by stage | `results/table_timing.tex` |
| Per-material TDR | `results/table_material.tex` |
| Numbers quoted in the text | `results/numbers.tex`, `results/headline.json` |
| Figures | `figures/tdr_vs_k.pdf`, `figures/relative_latency.pdf`, `figures/clusters_synthetic.pdf` |
