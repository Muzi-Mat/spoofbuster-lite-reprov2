"""Synthetic fingerprints for the smoke test and for the illustrative figure (no real biometric data).

Ridges are grown by repeatedly Gabor-filtering random noise along a core/delta orientation field (the idea
behind SFinGe). 'Spoof' materials are crude image-level corruptions so a tiny model has something to learn.
These images only prove the code runs; they say nothing about real presentation-attack detection.
"""
import argparse
import os

import cv2
import numpy as np

from .minutiae import gabor_enhance, normalise


def fingerprint(rng, size=320):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2 + rng.normal(0, 15, 2)
    dx, dy = cx + rng.normal(0, 25), cy + size * 0.38
    theta = np.mod(0.5 * (np.arctan2(yy - cy, xx - cx) - np.arctan2(yy - dy, xx - dx)) + np.pi / 2, np.pi)
    img = rng.normal(size=(size, size)).astype(np.float32)
    for _ in range(6):
        img = np.clip(gabor_enhance(normalise(img), theta.astype(np.float32)), -3, 3)
    ridge = 128 + 100 * np.tanh(1.5 * normalise(img))
    ell = ((xx - size / 2) / (size * 0.4)) ** 2 + ((yy - size / 2) / (size * 0.47)) ** 2
    alpha = np.clip((1.0 - ell) * 8, 0, 1)  # soft edge of the finger
    out = alpha * ridge + (1 - alpha) * 245
    return np.clip(out + rng.normal(0, 4, out.shape), 0, 255).astype(np.uint8)


def corrupt(img, material, rng):
    f = img.astype(np.float32)
    if material == "blurry_lowcontrast":
        f = 245 - 0.6 * (245 - cv2.GaussianBlur(f, (0, 0), 1.5))
    elif material == "speckled":
        f = f + rng.normal(0, 25, f.shape) * (rng.random(f.shape) < 0.3)
    return np.clip(f, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=60, help="images per class")
    a = p.parse_args()
    rng = np.random.default_rng(0)
    plan = [("bonafide", None)] + [(os.path.join("spoof", m), m) for m in ("blurry_lowcontrast", "speckled")]
    for folder, mat in plan:
        os.makedirs(os.path.join(a.out, folder), exist_ok=True)
        n = a.n if mat is None else a.n // 2
        for i in range(n):
            img = fingerprint(rng)
            if mat:
                img = corrupt(img, mat, rng)
            cv2.imwrite(os.path.join(a.out, folder, f"{i:04d}.png"), img)
    print(f"wrote synthetic dataset to {a.out}")


if __name__ == "__main__":
    main()
