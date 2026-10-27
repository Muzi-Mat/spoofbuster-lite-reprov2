"""Stage 1: list the images and make a fixed train/val/test split, saved to results/split.csv.

Expected layout:  DATA_DIR/bonafide/**/*.{bmp,png,...}   DATA_DIR/spoof/<material>/**/*.{...}
The split is committed so the test set is identical on every machine.
"""
import os

import numpy as np
import pandas as pd

from .config import get_args

EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def list_images(data_dir):
    rows = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.lower().endswith(EXTS):
                continue
            rel = os.path.relpath(os.path.join(root, f), data_dir)
            parts = rel.split(os.sep)
            if parts[0] == "bonafide":
                rows.append((rel, 0, "bonafide"))
            elif parts[0] == "spoof" and len(parts) >= 3:
                rows.append((rel, 1, parts[1]))  # material = first folder under spoof/
    if not rows:
        raise SystemExit(f"No images found under {data_dir}/bonafide or {data_dir}/spoof/<material>/")
    # sorted so the split does not depend on filesystem listing order
    return pd.DataFrame(sorted(rows), columns=["path", "label", "material"])


def make_split(df, cfg):
    rng = np.random.default_rng(cfg["split_seed"])
    df = df.copy()
    df["split"] = "train"
    bona = df.index[df.label == 0].to_numpy()
    # paper: 1,000 bona fide test images; a smaller dataset keeps half its bona fide images for training
    n_test_b = min(cfg["n_test_bonafide"], len(bona) // 2)
    df.loc[rng.choice(bona, n_test_b, replace=False), "split"] = "test"
    # spoofs: stratified by material so every material appears in test
    for _, g in df[df.label == 1].groupby("material"):
        n = max(1, int(round(cfg["spoof_test_frac"] * len(g))))
        df.loc[rng.choice(g.index.to_numpy(), n, replace=False), "split"] = "test"
    train = df.index[df.split == "train"].to_numpy()
    df.loc[rng.choice(train, int(round(cfg["val_frac"] * len(train))), replace=False), "split"] = "val"
    df.insert(0, "image_id", np.arange(len(df)))
    return df


def main():
    args, cfg = get_args(__doc__)
    out = os.path.join(args.results, "split.csv")
    df = list_images(args.data)
    if os.path.exists(out):
        old = pd.read_csv(out)
        if sorted(old.path) == sorted(df.path):
            print(f"Reusing committed split {out}")
            return
        raise SystemExit(f"{out} does not match the images in {args.data}; delete it to make a new split")
    split = make_split(df, cfg)
    split.to_csv(out, index=False)
    print(split.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
