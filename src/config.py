"""All experiment settings in one place, so every number in the write-up traces back here.

Values marked (paper) come from Chugh & Jain, arXiv:1812.11574. Values marked (ours) are choices the
paper does not specify; each one is listed as a deviation in the README and the write-up.
"""
import argparse
import os

DEFAULTS = dict(
    patch_size=96,                    # (paper) 96x96 patches, Sec. 3
    ks=[5, 10, 15, 20, 25, 30],       # (paper) cluster counts in Table 2
    kmeans_seeds=[0, 1, 2, 3, 4],     # (paper) "5-fold cross-validation" over k-means initialisation; we read it as 5 inits
    model_seeds=[0, 1],               # (ours) second training seed = a number the paper does not report
    epochs=6,                         # (ours) paper gives no training schedule
    batch_size=128,                   # (ours)
    train_patches_per_image=48,       # (ours) RAM cap on Colab; the paper used every minutia
    lr=1e-4,                          # (ours)
    imagenet_init=True,               # (ours) paper does not say whether it fine-tunes from ImageNet
    n_test_bonafide=1000,             # (paper) 1,000 bona fide test images, Sec. 2.3
    spoof_test_frac=0.2,              # (ours) paper does not say how PA images were split for Table 2
    val_frac=0.1,                     # (ours) held out from training images for monitoring only
    split_seed=0,
    fdrs=[0.002, 0.01],               # (paper) 0.2% operating point; 1% is ours (less threshold noise)
    bootstrap=1000,                   # (ours) resamples for the test-set sampling error of TDR
    timing_images=100,                # (ours) images used for latency measurement
)

# Tiny settings for the synthetic smoke test: proves the pipeline runs end to end, not that it is accurate.
SMOKE = dict(
    ks=[5, 10], kmeans_seeds=[0, 1], model_seeds=[0], epochs=1, batch_size=32,
    imagenet_init=False, n_test_bonafide=20, timing_images=4, bootstrap=50,
)


# Synthetic stand-in runs (the real dataset could not be obtained). Two presets; SYNTHETIC picks one.
# CPU: what fits a 2-core CPU overnight, trained from scratch because ImageNet weights could not be
#      downloaded there, 12 training patches per image, 5 epochs.
# T4:  a Colab T4 GPU: ImageNet initialisation (as fine-tuning is the usual MobileNet recipe), up to 48
#      patches per image, 8 epochs, 4 training seeds. Every setting is reported in the write-up.
SYNTHETIC_CPU = dict(
    ks=[5, 10, 15, 20], imagenet_init=False, train_patches_per_image=12, epochs=5,
    lr=1e-3,  # from scratch needs a larger step than fine-tuning
    model_seeds=[0, 1],
)
SYNTHETIC_T4 = dict(
    ks=[5, 10, 15, 20], imagenet_init=True, train_patches_per_image=48, epochs=8,
    lr=1e-4,  # fine-tuning from ImageNet weights
    model_seeds=[0, 1, 2, 3],
)
SYNTHETIC = SYNTHETIC_CPU  # colab_synthetic_t4.ipynb switches this line to SYNTHETIC_T4
SYNTHETIC_DATA = dict(n_bona=1600, n_per_material=200)  # 1,600 bona fide, 6 materials x 200 = 1,200 spoofs


def get_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", default=os.environ.get("DATA_DIR"), help="dataset root with bonafide/ and spoof/<material>/")
    p.add_argument("--work", default=os.environ.get("WORK", "work"), help="scratch dir for caches (not committed)")
    p.add_argument("--results", default="results")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--figures", default="figures")
    p.add_argument("--smoke", action="store_true", default=bool(os.environ.get("SMOKE")))
    p.add_argument("--synthetic", action="store_true", help="settings for the synthetic stand-in run")
    p.add_argument("--retrain", action="store_true", help="train even if a checkpoint exists")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override a setting, e.g. --set epochs=3 (any override must be reported in the write-up)")
    args = p.parse_args()
    cfg = dict(DEFAULTS)
    if args.synthetic:
        cfg.update(SYNTHETIC)
    if args.smoke:
        cfg.update(SMOKE)
    for kv in args.set:
        key, val = kv.split("=", 1)
        default = DEFAULTS[key]
        if isinstance(default, bool):
            cfg[key] = val.lower() in ("1", "true", "yes")
        elif isinstance(default, list):
            cfg[key] = [type(default[0])(v) for v in val.split(",")]
        else:
            cfg[key] = type(default)(val)
    for d in (args.work, args.results, args.checkpoints, args.figures):
        os.makedirs(d, exist_ok=True)
    return args, cfg
