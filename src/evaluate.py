"""Stage 4: score every test image under every patch-selection setting, write results/scores.csv.gz.

One row per (model seed, image, method, k, k-means/random seed) with the fingerprint-level score, both
weighted by cluster size (the paper's rule) and unweighted (our ablation), plus the number of CNN calls.
scores.csv.gz is committed, so all tables and figures can be rebuilt without the dataset (src/report.py).

Efficiency: "random" and "nearest" patches are centred on real minutiae with the minutia's own direction,
so they are exactly patches the all-minutiae baseline already scored; their scores are looked up, not
recomputed. Only k-means centroid patches need new CNN calls. n_evals still counts the CNN calls each
setting would need on its own.
"""
import os
import pickle

import keras
import numpy as np
import pandas as pd

from .config import get_args
from .minutiae import orientation_at
from .patches import kmeans_clusters, load_gray, nearest_minutia, patches_for, random_minutiae, select_points


def main():
    args, cfg = get_args(__doc__)
    split = pd.read_csv(os.path.join(args.results, "split.csv"))
    test = list(split[split.split == "test"].itertuples())
    with open(os.path.join(args.work, "features.pkl"), "rb") as f:
        feats = pickle.load(f)
    rows = []
    for mseed in cfg["model_seeds"]:
        model = keras.models.load_model(os.path.join(args.checkpoints, f"mobilenet_seed{mseed}.keras"))
        for start in range(0, len(test), 32):
            chunk, patches, plans = test[start:start + 32], [], []
            for r in chunk:
                feat = feats[r.image_id]
                img = load_gray(os.path.join(args.data, r.path))
                c, d, _ = select_points(feat, "all")      # includes the no-minutiae fallback patch
                patches.append(patches_for(img, c, d, cfg["patch_size"]))
                clusters = {}
                n = len(feat["xy"])
                for k in cfg["ks"]:
                    for s in cfg["kmeans_seeds"]:
                        if n > k:
                            cent, w = kmeans_clusters(feat["xy"], k, s)
                            dirs = np.array([orientation_at(feat, x, y) for x, y in cent])
                            patches.append(patches_for(img, cent, dirs, cfg["patch_size"]))
                            clusters[k, s] = (cent, w)
                plans.append((r, n, clusters))
            sizes = [len(p) for p in patches]
            probs = model.predict(np.concatenate(patches)[..., None].astype(np.float32),
                                  batch_size=1024, verbose=0).ravel()
            parts = iter(np.split(probs, np.cumsum(sizes)[:-1]))
            for r, n, clusters in plans:
                p_all = next(parts)

                def add(method, k, s, p, w):
                    rows.append(dict(model_seed=mseed, image_id=r.image_id, label=r.label, material=r.material,
                                     method=method, k=k, seed=s, n_evals=len(p),
                                     score_w=float(np.average(p, weights=w)), score_u=float(p.mean())))
                add("all", 0, 0, p_all, np.ones(len(p_all)))
                for k in cfg["ks"]:
                    for s in cfg["kmeans_seeds"]:
                        if n <= k:  # too few minutiae: every setting falls back to all of them
                            for m in ("kmeans", "nearest", "random"):
                                add(m, k, s, p_all, np.ones(len(p_all)))
                            continue
                        cent, w = clusters[k, s]
                        add("kmeans", k, s, next(parts), w)
                        add("nearest", k, s, p_all[nearest_minutia(feats[r.image_id]["xy"], cent)], w)
                        add("random", k, s, p_all[random_minutiae(n, k, s)], np.ones(k))
            print(f"model {mseed}: {min(start + 32, len(test))}/{len(test)} test images", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.results, "scores.csv.gz"), index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
