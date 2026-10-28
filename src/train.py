"""Stage 3: train the patch classifier on ALL minutia-centred patches of the training images.

One model per training seed. The same model is then used for every patch-selection setting, as in the paper
(Spoof Buster Lite changes which patches are evaluated at test time, not the network weights).
A checkpoint that already exists is reused unless --retrain is given, so a committed checkpoint skips training.
"""
import json
import os
import pickle

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import get_args
from .model import build_model
from .patches import load_gray, patches_for, select_points


def build_patch_arrays(args, cfg, split, feats):
    """All-minutiae patches for train+val images, cached in WORK (shared by every seed).

    Memory is preallocated from the minutiae counts, so peak RAM is one copy of the patch array.
    Images with more than train_patches_per_image minutiae contribute a random subset of that size."""
    cache = os.path.join(args.work, "trainval_patches.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["x"], z["y"], z["is_val"]
    rows = list(split[split.split != "test"].itertuples())
    cap, size = cfg["train_patches_per_image"], cfg["patch_size"]
    rng = np.random.default_rng(0)
    counts = [min(max(len(feats[r.image_id]["xy"]), 1), cap) for r in rows]
    x = np.empty((sum(counts), size, size), np.uint8)
    y = np.empty(len(x), np.float32)
    v = np.empty(len(x), bool)
    pos = 0
    for r, n in zip(rows, counts):
        img = load_gray(os.path.join(args.data, r.path))
        c, d, _ = select_points(feats[r.image_id], "all")
        if len(c) > n:
            keep = rng.choice(len(c), n, replace=False)
            c, d = c[keep], d[keep]
        x[pos:pos + n] = patches_for(img, c, d, size)
        y[pos:pos + n] = r.label          # every patch inherits its image's label
        v[pos:pos + n] = r.split == "val"
        pos += n
    np.savez(cache, x=x, y=y, is_val=v)
    return x, y, v


def batches(x, y, bs, seed):
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(len(x))
        for i in range(0, len(x) - bs + 1, bs):
            idx = np.sort(perm[i:i + bs])
            yield x[idx, ..., None].astype(np.float32), y[idx]


def bn_layers(layer):
    for sub in getattr(layer, "layers", []):
        if isinstance(sub, keras.layers.BatchNormalization):
            yield sub
        yield from bn_layers(sub)


def recalibrate_bn(model, x, bs=256, n_batches=60, seed=0):
    """Recompute BatchNorm running statistics with the final weights.

    Why: trained from scratch with a large learning rate, the running means/variances (momentum 0.99) lag
    behind the weights, and in inference mode the network then gave one constant output (validation AUC
    stuck at 0.5 while training AUC rose). Forward passes in training mode with momentum set to give an
    exact cumulative average replace the stale statistics; no weights are changed."""
    bns = list(bn_layers(model))
    old = [b.momentum for b in bns]
    rng = np.random.default_rng(seed)
    for i in range(n_batches):
        for b in bns:
            b.momentum = i / (i + 1)  # running = mean of the batch statistics seen so far
        idx = np.sort(rng.choice(len(x), bs, replace=False))
        model(x[idx, ..., None].astype(np.float32), training=True)
    for b, m in zip(bns, old):
        b.momentum = m


def val_auc(model, xva, yva):
    p = model.predict(xva[..., None].astype(np.float32), batch_size=512, verbose=0).ravel()
    return float(keras.metrics.AUC()(yva, p))


def main():
    args, cfg = get_args(__doc__)
    split = pd.read_csv(os.path.join(args.results, "split.csv"))
    with open(os.path.join(args.work, "features.pkl"), "rb") as f:
        feats = pickle.load(f)
    x, y, is_val = build_patch_arrays(args, cfg, split, feats)
    xtr, ytr, xva, yva = x[~is_val], y[~is_val], x[is_val], y[is_val]
    print(f"train patches {len(xtr)} (spoof {ytr.mean():.2%}), val patches {len(xva)}")

    # Balance the loss: bona fide images have more patches than spoofs (paper Table 1: 228k vs 144k)
    pos = ytr.mean()
    class_weight = {0: 0.5 / (1 - pos), 1: 0.5 / pos}

    # Record what the models were trained on, so the write-up can state it (results/train_env.json)
    gpus = tf.config.list_physical_devices("GPU")
    name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU") if gpus else None
    with open(os.path.join(args.results, "train_env.json"), "w") as f:
        json.dump({"gpu": name, "cpu_count": os.cpu_count(), "tensorflow": tf.__version__,
                   "model_seeds": cfg["model_seeds"]}, f, indent=1)

    for seed in cfg["model_seeds"]:
        ckpt = os.path.join(args.checkpoints, f"mobilenet_seed{seed}.keras")
        if os.path.exists(ckpt) and not args.retrain:
            print(f"Reusing {ckpt}")
            continue
        keras.utils.set_random_seed(seed)  # python, numpy and TF seeds; cuDNN can still add small nondeterminism
        model = build_model(cfg["patch_size"], cfg["imagenet_init"])
        model.compile(optimizer=keras.optimizers.Adam(cfg["lr"]), loss="binary_crossentropy",
                      metrics=[keras.metrics.AUC(name="auc")])
        bs = cfg["batch_size"]
        ds = tf.data.Dataset.from_generator(
            lambda: batches(xtr, ytr, bs, seed),
            output_signature=(tf.TensorSpec((bs, cfg["patch_size"], cfg["patch_size"], 1), tf.float32),
                              tf.TensorSpec((bs,), tf.float32))).prefetch(4)
        hist = model.fit(ds, steps_per_epoch=len(xtr) // bs, epochs=cfg["epochs"],
                         validation_data=(xva[..., None].astype(np.float32), yva), validation_batch_size=512,
                         class_weight=class_weight, verbose=2)
        history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        history["val_auc_before_bn_recal"] = val_auc(model, xva, yva)
        recalibrate_bn(model, xtr, seed=seed)
        history["val_auc_after_bn_recal"] = val_auc(model, xva, yva)
        print(f"seed {seed}: val AUC {history['val_auc_before_bn_recal']:.3f} -> "
              f"{history['val_auc_after_bn_recal']:.3f} after BatchNorm recalibration")
        model.compile()  # drop the optimizer state: inference needs only the weights
        model.save(ckpt)
        with open(os.path.join(args.results, f"train_history_seed{seed}.json"), "w") as f:
            json.dump(history, f, indent=1)


if __name__ == "__main__":
    main()
