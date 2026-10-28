# Reproducing minutiae clustering in Fingerprint Spoof Buster Lite

MSc reproduction project, University of the Witwatersrand, 2026.
Paper: T. Chugh and A. K. Jain, *Fingerprint Presentation Attack Detection: Generalization and Efficiency*,
arXiv:1812.11574 (ICB 2019).

> **Important: the results in this repository are on SYNTHETIC fingerprints, not the paper's data.**
> The paper's datasets (MSU-FPAD and PBSKD) are gated behind a data agreement with Michigan State
> University. The first author replied that he no longer has access and cannot share them. A request to
> the MSU lab had no response by the deadline. The pipeline runs unchanged on the real data as soon as it
> is available (see "Running it").

## The claim

> Clustering fingerprint minutiae with k-means and classifying only one 96×96 patch per cluster centroid
> cuts computation without a significant loss in presentation-attack detection accuracy.

**Source:** Table 2 of the paper. With k = 10 clusters, TDR = 95.3% at FDR = 0.2%, against 95.7% when every
minutia's patch is classified (35 per image on average). Inference time drops from 1,554 ms to 302 ms on a
Redmi Note 4.

**Research question:** does k = 10 keep TDR at FDR = 0.2% within one percentage point of the
all-minutiae baseline, while reducing the number of MobileNet evaluations?

## Route: reimplementation

The authors' code is not public, so the whole pipeline is **reimplemented here**. All code in `src/` was
written for this project with AI assistance and reviewed by the author. The only library components used
are Keras's MobileNet-v1 architecture, scikit-learn's `KMeans`, and OpenCV/scikit-image image operations.

| File | What it does |
|---|---|
| `src/synthetic.py` | seeded synthetic fingerprints: bona fide and 6 artefact "materials" |
| `src/data.py` | fixed train/val/test split, committed as `results/split.csv` |
| `src/minutiae.py` | minutiae extractor: orientation field, Gabor enhancement, skeleton, crossing number |
| `src/patches.py` | patch selection (all / k-means / nearest-minutia / random) and aligned 96×96 patch cutting |
| `src/train.py` | trains MobileNet-v1 on minutia-centred patches of the training images, one model per seed |
| `src/evaluate.py` | fingerprint-level scores for every setting, saved as `results/scores.csv.gz` |
| `src/timing.py` | single-thread CPU latency per pipeline stage, using a quantised TFLite model |
| `src/report.py` | every table, figure and number in the write-up, from committed results only |
| `src/config.py` | every setting; `SYNTHETIC` holds the settings used for the committed run |

## Differences from the paper (all reported in the write-up)

1. **Data.** Synthetic fingerprints replace MSU-FPAD/PBSKD: 1,600 bona fide images and 6 materials × 200
   spoofs, 288×288 px. Every image gets random capture variation. Each material adds one artefact (ridge
   width, blur, print quantisation, moisture, ridge breaks or elastic distortion) at a random strength,
   over a random 40–100% of the print. These are loose analogies, not models of real spoof materials.
2. **Minutiae extractor.** The paper does not name one. We wrote our own.
3. **Centroid alignment.** The paper does not say how a centroid patch is rotated. We use the
   ridge-orientation field at the centroid (modulo 180°).
4. **Framework.** We use Keras MobileNet-v1 instead of TF-Slim, which needs TensorFlow 1.x. The
   architecture is the same at a 96×96 input, and global average pooling over the 3×3 map equals the
   paper's 3×3 pool.
5. **Training.** The paper gives no recipe. The committed results use the `SYNTHETIC_T4` preset in
   `src/config.py`: ImageNet initialisation, Adam with lr 1e-4, batch 128, 8 epochs, class-balanced loss,
   dropout 0.2, at most 48 patches per training image, and 4 training seeds, trained on a Colab T4. An
   earlier CPU run (`SYNTHETIC_CPU`, from scratch, git commit `921bd50`) is described in the write-up.
   After training, BatchNorm statistics are recomputed with the final weights (`recalibrate_bn`).
6. **Split.** Half of the bona fide images go to test (800). The paper used 1,000 of 5,743. The paper
   gives no spoof split, so we use a stratified 20% of each material.
7. **Hardware.** Training hardware is recorded in `results/train_env.json`. Timing runs on one CPU thread
   with TFLite 8-bit weights. The paper
   timed a phone, so only *relative* latency is compared.
8. **Repeats.** The paper's "5-fold cross-validation" is read as 5 k-means seeds on one model. We add
   extra training seeds, listed in `SYNTHETIC` in `src/config.py`.

**Numbers the paper does not report:**
- random-minutiae ablation,
- unweighted averaging,
- nearest-minutia patch centres,
- several training seeds,
- fixed-threshold comparison,
- TDR at FDR = 1%,
- bootstrap error,
- latency by stage.

## Data and ethics

MSU-FPAD and PBSKD contain fingerprints of real people and are released only under an academic
agreement with MSU. We did not use unofficial copies. The committed results use synthetic images only,
so no biometric data of any person is in this repository.

The code never commits images, patches or minutiae when run on real data. The minutiae cache stays in
the git-ignored `work/` folder, because minutiae coordinates are a biometric template.

## Running it

```bash
conda env create -f environment.yml && conda activate spoofbuster-lite-repro

# 1. Every number and figure in the write-up, from a clean checkout (synthetic data is regenerated from
#    seeds; committed checkpoints are reused, add --retrain to retrain; ~1 h on 2 CPU cores; ~40 min with --retrain on a Colab T4)
bash run_all.sh --synthetic          # uses committed checkpoints; add --retrain to retrain (T4 recommended)

# 2. Rebuild every table, figure and number from the committed results only (~1 min)
bash run_all.sh --report-only

# 3. On the real data, once available: DATA_DIR with bonafide/ and spoof/<material>/
DATA_DIR=/path/to/data bash run_all.sh        # colab_run.ipynb does this on a Colab GPU

# 4. 3-minute smoke test
bash run_all.sh --smoke
```

`environment.yml` was generated by `scripts/pin_env.py` in the environment that produced `results/`.
Small run-to-run differences in training are possible because of floating-point nondeterminism.

## Outputs mapped to the write-up

| Write-up item | File |
|---|---|
| Table 2 side by side | `results/table_main.tex` |
| Ablations | `results/table_ablation.tex` |
| Training seeds | `results/table_seeds.tex` |
| Latency by stage | `results/table_timing.tex` |
| Per-material TDR | `results/table_material.tex` |
| Numbers quoted in the text | `results/numbers.tex`, `results/headline.json` |
| Figures | `figures/tdr_vs_k.pdf`, `figures/relative_latency.pdf`, `figures/clusters_synthetic.pdf` |
