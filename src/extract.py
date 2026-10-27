"""Stage 2: run minutiae extraction on every image once and cache it in WORK/features.pkl.

The cache is NOT committed: minutiae coordinates are a biometric template of a real finger."""
import os
import pickle
import time
from multiprocessing import Pool

import pandas as pd

from .config import get_args
from .minutiae import extract
from .patches import load_gray


def _one(path):
    return extract(load_gray(path))


def main():
    args, cfg = get_args(__doc__)
    out = os.path.join(args.work, "features.pkl")
    split = pd.read_csv(os.path.join(args.results, "split.csv"))
    if os.path.exists(out):
        with open(out, "rb") as f:
            if set(pickle.load(f)) >= set(split.image_id):
                print(f"Reusing {out}")
                return
    paths = [os.path.join(args.data, p) for p in split.path]
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        feats = pool.map(_one, paths, chunksize=8)
    feats = dict(zip(split.image_id, feats))
    with open(out, "wb") as f:
        pickle.dump(feats, f)
    n = pd.Series({i: len(v["xy"]) for i, v in feats.items()})
    print(f"Extracted {len(feats)} images in {time.time() - t0:.0f}s; minutiae/image mean {n.mean():.1f} "
          f"median {n.median():.0f}; images with 0 minutiae: {(n == 0).sum()}")
    # summary stats only (no coordinates) are safe to commit
    pd.DataFrame({"image_id": n.index, "n_minutiae": n.values}).to_csv(
        os.path.join(args.results, "minutiae_counts.csv"), index=False)


if __name__ == "__main__":
    main()
