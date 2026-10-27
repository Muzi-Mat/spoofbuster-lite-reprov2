"""Stage 4: score every test image under every patch-selection setting, write results/scores.csv.gz.

One row per (model seed, image, method, k, k-means/random seed) with the fingerprint-level score, both
weighted by cluster size (the paper's rule) and unweighted (our ablation), plus the number of CNN calls.
scores.csv.gz is committed, so all tables and figures can be rebuilt without the dataset (src/report.py).
"""
import os
import pickle

import keras
import numpy as np
import pandas as pd

from .config import get_args
from .patches import load_gray, patches_for, select_points


def settings(cfg):
    yield "all", 0, 0
    for k in cfg["ks"]:
        for s in cfg["kmeans_seeds"]:
            yield "kmeans", k, s
            yield "nearest", k, s
            yield "random", k, s


def main():
    args, cfg = get_args(__doc__)
    split = pd.read_csv(os.path.join(args.results, "split.csv"))
    test = split[split.split == "test"]
    with open(os.path.join(args.work, "features.pkl"), "rb") as f:
        feats = pickle.load(f)
    rows = []
    for mseed in cfg["model_seeds"]:
        model = keras.models.load_model(os.path.join(args.checkpoints, f"mobilenet_seed{mseed}.keras"))
        recs = list(test.itertuples())
        for start in range(0, len(recs), 16):  # 16 images at a time keeps the patch batch in memory
            chunk, patches, meta = recs[start:start + 16], [], []
            for r in chunk:
                img = load_gray(os.path.join(args.data, r.path))
                for method, k, s in settings(cfg):
                    c, d, w = select_points(feats[r.image_id], method, k, s)
                    patches.append(patches_for(img, c, d, cfg["patch_size"]))
                    meta.append((r, method, k, s, w))
            sizes = [len(p) for p in patches]
            probs = model.predict(np.concatenate(patches)[..., None].astype(np.float32),
                                  batch_size=1024, verbose=0).ravel()
            for (r, method, k, s, w), p in zip(meta, np.split(probs, np.cumsum(sizes)[:-1])):
                rows.append(dict(model_seed=mseed, image_id=r.image_id, label=r.label, material=r.material,
                                 method=method, k=k, seed=s, n_evals=len(p),
                                 score_w=float(np.average(p, weights=w)), score_u=float(p.mean())))
            print(f"model {mseed}: {min(start + 16, len(recs))}/{len(recs)} test images", end="\r")
        print()
    pd.DataFrame(rows).to_csv(os.path.join(args.results, "scores.csv.gz"), index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
