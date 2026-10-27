"""Choosing patch centres (all minutiae / k-means centroids / random minutiae) and cutting aligned patches."""
import cv2
import numpy as np
from sklearn.cluster import KMeans

from .minutiae import orientation_at


def select_points(feat, method, k=None, seed=0):
    """Returns (centres Nx2, directions N, weights N).

    all    : every minutia, equal weight (Spoof Buster, the baseline row of Table 2)
    kmeans : k centroids of the minutiae coordinates, weight = cluster size (Spoof Buster Lite, Sec. 3)
    random : k minutiae drawn at random, equal weight (our ablation: does clustering matter, or just fewer patches?)
    nearest: k-means clusters, but each patch centred on the minutia closest to its centroid (our fix attempt)
    """
    xy, th = feat["xy"], feat["theta"]
    n = len(xy)
    if n == 0:  # no reliable minutiae: one patch at the centre of the print so the image still gets a score
        c = feat["mask_centre"][None]
        return c, np.array([orientation_at(feat, *c[0])]), np.ones(1)
    if method == "all" or n <= k:  # fewer minutiae than clusters -> clustering would be a no-op
        return xy, th, np.ones(n)
    if method in ("kmeans", "nearest"):
        # n_init=1 so the seed is the only source of variation, matching the paper's repeat-over-initialisation
        km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(xy)
        cent = km.cluster_centers_.astype(np.float32)
        weights = np.bincount(km.labels_, minlength=k).astype(np.float32)
        if method == "nearest":
            # Our fix attempt: the CNN was trained on minutia-centred patches, but a centroid usually lies
            # between minutiae. Snap each centroid to the closest real minutia so test patches look like
            # training patches. Same clusters and weights as "kmeans"; only the patch centre moves.
            idx = [np.argmin(((xy - c) ** 2).sum(1)) for c in cent]
            return xy[idx], th[idx], weights
        # A centroid has no minutia direction of its own; the paper does not say how it aligned these.
        # We use the ridge direction of the orientation field at the centroid (our choice, documented).
        dirs = np.array([orientation_at(feat, x, y) for x, y in cent])
        return cent, dirs, weights
    if method == "random":
        idx = np.random.default_rng(seed).choice(n, k, replace=False)
        return xy[idx], th[idx], np.ones(k)
    raise ValueError(method)


def extract_patch(img_u8, x, y, theta, size=96):
    """Cut a size x size patch centred on (x, y), rotated so the local ridge direction is horizontal.

    Alignment makes patches comparable regardless of how the finger was placed. Ridge direction is only
    known modulo pi (a minutia direction would be modulo 2*pi), so two patches may differ by a 180 deg flip."""
    M = cv2.getRotationMatrix2D((float(x), float(y)), float(np.degrees(theta)), 1.0)
    M[0, 2] += size / 2 - x  # move the rotation centre to the middle of the output patch
    M[1, 2] += size / 2 - y
    # white fill: outside the image an optical reader sees bright background, not black
    return cv2.warpAffine(img_u8, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def patches_for(img_u8, centres, dirs, size=96):
    return np.stack([extract_patch(img_u8, x, y, t, size) for (x, y), t in zip(centres, dirs)])


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img
