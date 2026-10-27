"""Synthetic fingerprints: a clearly labelled stand-in for MSU-FPAD/PBSKD, which I could not obtain.

Ridges are grown by repeatedly Gabor-filtering random noise along a core/delta orientation field (the idea
behind SFinGe). Every image, bona fide or spoof, gets random capture variation (contrast, noise, blur,
dry/wet blotches), so the classes overlap. Each of six synthetic "materials" applies one kind of artefact
(ridge width, blur, print quantisation, moisture, ridge breaks, elastic distortion) with a random strength
and only over a random part of the print, so some spoofs are hard and the patch location matters.

What this can test: whether classifying k centroid patches instead of all minutia patches loses accuracy
for a patch-based detector, and how latency scales. What it cannot test: anything about real spoof
materials. The materials are loose analogies, not models of the real ones.
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np

from .config import SYNTHETIC_DATA
from .minutiae import gabor_enhance, normalise

MATERIALS = ["thick_ridges", "blurred", "printed", "moist", "broken_ridges", "distorted"]


def fingerprint(rng, size=288):
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
    return (alpha * ridge + (1 - alpha) * 245).astype(np.float32), alpha


def smooth_field(rng, shape, sigma):
    f = cv2.GaussianBlur(rng.normal(size=shape).astype(np.float32), (0, 0), sigma)
    return (f - f.mean()) / (f.std() + 1e-6)


def region_mask(rng, shape):
    """Smooth random region covering 40-100% of the image: the artefact appears only there."""
    cover = rng.uniform(0.4, 1.0)
    f = smooth_field(rng, shape, 30)
    m = (f < np.quantile(f, cover)).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 4)


def capture(f, alpha, rng):
    """Variation every live-scan image has, bona fide or not."""
    c = rng.uniform(0.75, 1.1)
    f = 245 - c * (245 - f)
    if rng.random() < 0.5:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 0.9))
    f = f * (1 + 0.08 * rng.uniform(0, 1) * smooth_field(rng, f.shape, 20) * alpha)  # dry or wet patches
    return f + rng.normal(0, rng.uniform(2, 6), f.shape)


def material(f, alpha, name, rng):
    s = rng.uniform(0.3, 1.0)  # artefact strength
    if name == "thick_ridges":
        k = 2 + int(s > 0.65)
        g = cv2.erode(f, np.ones((k, k), np.uint8))           # dark ridges grow
    elif name == "blurred":
        g = cv2.GaussianBlur(f, (0, 0), 0.6 + 1.0 * s)
    elif name == "printed":
        levels = int(round(8 - 4 * s))                          # coarse grey levels, like a printer
        g = np.round(f / 255 * levels) / levels * 255
        g = g + 6 * s * np.sign(np.sin(np.arange(f.shape[1]) * 2.2))[None, :]  # faint banding
    elif name == "moist":
        g = f * (1 - 0.18 * s * np.clip(smooth_field(rng, f.shape, 8), 0, None))
        g = 245 - (1 + 0.25 * s) * (245 - g)                  # higher contrast, blotchy
    elif name == "broken_ridges":
        holes = (rng.random(f.shape) < 0.01 * s).astype(np.float32)
        holes = cv2.dilate(holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        g = f + holes * (240 - f) * 0.8
    elif name == "distorted":
        h, w = f.shape
        dx = smooth_field(rng, f.shape, 12) * 1.8 * s
        dy = smooth_field(rng, f.shape, 12) * 1.8 * s
        mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        g = cv2.GaussianBlur(cv2.remap(f, mx + dx, my + dy, cv2.INTER_LINEAR, borderValue=245), (0, 0), 0.5)
    else:
        raise ValueError(name)
    m = region_mask(rng, f.shape) * alpha
    return m * g + (1 - m) * f


def make(job):
    out, cls, i = job
    rng = np.random.default_rng([2026, MATERIALS.index(cls) + 1 if cls != "bonafide" else 0, i])
    f, alpha = fingerprint(rng)
    if cls != "bonafide":
        f = material(f, alpha, cls, rng)
    f = capture(f, alpha, rng)
    folder = os.path.join(out, "bonafide") if cls == "bonafide" else os.path.join(out, "spoof", cls)
    cv2.imwrite(os.path.join(folder, f"{i:05d}.png"), np.clip(f, 0, 255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n_bona", type=int, default=SYNTHETIC_DATA["n_bona"])
    p.add_argument("--n_per_material", type=int, default=SYNTHETIC_DATA["n_per_material"])
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "bonafide"), exist_ok=True)
    for m in MATERIALS:
        os.makedirs(os.path.join(a.out, "spoof", m), exist_ok=True)
    jobs = [(a.out, "bonafide", i) for i in range(a.n_bona)]
    jobs += [(a.out, m, i) for m in MATERIALS for i in range(a.n_per_material)]
    with Pool(os.cpu_count()) as pool:
        pool.map(make, jobs, chunksize=16)
    print(f"wrote {len(jobs)} synthetic images to {a.out}")


if __name__ == "__main__":
    main()
