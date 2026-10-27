"""Minutiae extraction (our own implementation; the paper's extractor is not named).

Pipeline: foreground mask -> local normalisation -> ridge orientation field (structure tensor)
-> oriented Gabor enhancement -> binarise -> skeletonise -> crossing number -> filter spurious points.
Assumes ~500 ppi images with dark ridges on a light background (CrossMatch Guardian 200 is 500 ppi).
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

RIDGE_PERIOD = 9.0   # pixels between ridges at 500 ppi
OF_DS = 8            # orientation field is stored downsampled by this factor


def foreground_mask(img):
    """Fingerprint area = where local contrast is high; background on an optical reader is flat."""
    m = cv2.blur(img, (16, 16))
    sd = np.sqrt(np.maximum(cv2.blur(img * img, (16, 16)) - m * m, 0))
    mask = (sd > 0.25 * np.percentile(sd, 99)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:  # keep the largest blob only; smudges elsewhere are not the finger
        mask = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return mask


def normalise(img):
    m = cv2.GaussianBlur(img, (0, 0), 8)
    sd = np.sqrt(cv2.GaussianBlur((img - m) ** 2, (0, 0), 8))
    return (img - m) / (sd + 1e-3)


def orientation_field(n):
    """Ridge direction in [0, pi) per pixel, from the smoothed gradient structure tensor.

    Angles are averaged as (cos 2t, sin 2t) because a ridge direction t and t+pi are the same."""
    gx = cv2.Sobel(n, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), 7)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), 7)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), 7)
    c2, s2 = gxx - gyy, 2 * gxy
    coherence = np.sqrt(c2 ** 2 + s2 ** 2) / (gxx + gyy + 1e-6)
    theta_grad = 0.5 * np.arctan2(s2, c2)          # dominant gradient direction
    theta = np.mod(theta_grad + np.pi / 2, np.pi)  # ridges run perpendicular to the gradient
    return theta.astype(np.float32), coherence.astype(np.float32)


def gabor_enhance(n, theta, n_orient=16):
    """Filter with a Gabor tuned to the local ridge direction; joins broken ridges, removes pores/noise."""
    q = np.round(theta / np.pi * n_orient).astype(int) % n_orient
    out = np.zeros_like(n)
    for i in range(n_orient):
        sel = q == i
        if not sel.any():
            continue
        ridge_dir = i * np.pi / n_orient
        # OpenCV's theta is the normal to the stripes, i.e. perpendicular to the ridge direction
        k = cv2.getGaborKernel((25, 25), 4.0, ridge_dir + np.pi / 2, RIDGE_PERIOD, 1.0, 0, ktype=cv2.CV_32F)
        k -= k.mean()  # zero DC so flat regions give zero response
        out[sel] = cv2.filter2D(n, cv2.CV_32F, k)[sel]
    return out


def crossing_number(skel):
    """CN = half the number of 0/1 transitions around the 8-neighbourhood: 1 = ending, 3 = bifurcation."""
    p = np.pad(skel.astype(np.int16), 1)
    nb = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
    return sum(np.abs(nb[i] - nb[(i + 1) % 8]) for i in range(8)) // 2


def extract(img_u8):
    """Returns a dict with minutiae (x, y, ridge direction) plus a compact orientation field.

    The orientation field is kept so a k-means centroid (which is not a minutia) can still be aligned."""
    img = img_u8.astype(np.float32) / 255.0
    mask = foreground_mask(img)
    n = normalise(img)
    theta, coh = orientation_field(n)
    enh = gabor_enhance(n, theta)
    ridges = ((enh < 0) & (mask > 0)).astype(np.uint8)  # ridges are dark, so negative after normalising
    # drop specks: tiny components become fake ridge endings
    cnt, lab, stats, _ = cv2.connectedComponentsWithStats(ridges, connectivity=8)
    ridges = np.isin(lab, 1 + np.where(stats[1:, cv2.CC_STAT_AREA] >= 30)[0]).astype(np.uint8)
    skel = skeletonize(ridges > 0)
    cn = crossing_number(skel)
    cand = skel & ((cn == 1) | (cn == 3))

    # Only trust points away from the mask edge (ridge endings there are just the edge of the print)
    # borderValue=0: OpenCV's default treats outside the image as foreground, which would keep the image edge
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    cand &= (inner > 0) & (coh > 0.2)
    ys, xs = np.nonzero(cand)
    kinds = cn[ys, xs]
    xy = np.stack([xs, ys], 1).astype(np.float32)

    # Two minutiae closer than one ridge period are almost always a break or a spur: drop both
    if len(xy) > 1:
        d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        keep = d.min(1) > RIDGE_PERIOD + 1
        xy, kinds = xy[keep], kinds[keep]

    ix, iy = xy[:, 0].astype(int), xy[:, 1].astype(int)
    my, mx = np.nonzero(mask) if mask.any() else (np.array([img.shape[0] // 2]), np.array([img.shape[1] // 2]))
    return dict(
        xy=xy,
        theta=theta[iy, ix].astype(np.float32),
        kind=kinds.astype(np.int8),
        # downsampled doubled-angle field: enough to look up the direction at any point
        ocos=cv2.resize(np.cos(2 * theta), None, fx=1 / OF_DS, fy=1 / OF_DS, interpolation=cv2.INTER_AREA).astype(np.float16),
        osin=cv2.resize(np.sin(2 * theta), None, fx=1 / OF_DS, fy=1 / OF_DS, interpolation=cv2.INTER_AREA).astype(np.float16),
        mask_centre=np.array([mx.mean(), my.mean()], np.float32),
        shape=img.shape,
    )


def orientation_at(feat, x, y):
    c, s = feat["ocos"], feat["osin"]
    i = int(np.clip(round(y / OF_DS), 0, c.shape[0] - 1))
    j = int(np.clip(round(x / OF_DS), 0, c.shape[1] - 1))
    return float(np.mod(0.5 * np.arctan2(float(s[i, j]), float(c[i, j])), np.pi))
