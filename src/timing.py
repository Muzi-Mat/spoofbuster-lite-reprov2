"""Stage 5: per-image latency on one CPU thread, split by pipeline stage.

The paper times a quantised TFLite MobileNet on a Redmi Note 4 phone. We cannot run that, so absolute
milliseconds are NOT comparable. What is comparable: how latency scales with the number of patches, and
how much of the total is spent outside the CNN (minutiae extraction, k-means, patch cutting).
We convert the trained model to TFLite with dynamic-range (8-bit weight) quantisation, as close as we can
get to the paper's quantised model, and evaluate patches one at a time like the phone app.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # k-means and BLAS: one thread, like a single phone core
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from .config import get_args  # noqa: E402
from .minutiae import extract  # noqa: E402
from .patches import load_gray, patches_for, select_points  # noqa: E402

cv2.setNumThreads(1)


def tflite_runner(model, path):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantisation: int8 weights
    with open(path, "wb") as f:
        f.write(conv.convert())
    interp = tf.lite.Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    i_in, i_out = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]

    def run(patch):
        interp.set_tensor(i_in, patch[None, ..., None].astype(np.float32))
        interp.invoke()
        return float(interp.get_tensor(i_out)[0, 0])
    return run, os.path.getsize(path)


def main():
    args, cfg = get_args(__doc__)
    split = pd.read_csv(os.path.join(args.results, "split.csv"))
    test = split[split.split == "test"].sample(n=min(cfg["timing_images"], (split.split == "test").sum()),
                                               random_state=0)
    model = keras.models.load_model(os.path.join(args.checkpoints, f"mobilenet_seed{cfg['model_seeds'][0]}.keras"))
    try:
        run, size = tflite_runner(model, os.path.join(args.work, "mobilenet_q8.tflite"))
        backend = "tflite-dynamic-range-int8"
    except Exception as e:  # TFLite is being moved out of TensorFlow; fall back rather than fail
        print(f"TFLite unavailable ({e}); timing the Keras model instead")
        run = lambda p: float(model(p[None, ..., None].astype(np.float32), training=False)[0, 0])  # noqa: E731
        size, backend = None, "keras-cpu"

    settings = [("all", 0)] + [("kmeans", k) for k in cfg["ks"]]
    run(np.full((cfg["patch_size"],) * 2, 255, np.uint8))  # warm-up so the first image is not penalised
    rows = []
    for r in test.itertuples():
        img = load_gray(os.path.join(args.data, r.path))
        t0 = time.perf_counter()
        feat = extract(img)
        t_min = time.perf_counter() - t0
        for method, k in settings:
            t0 = time.perf_counter()
            c, d, w = select_points(feat, method, k, 0)
            t1 = time.perf_counter()
            P = patches_for(img, c, d, cfg["patch_size"])
            t2 = time.perf_counter()
            for p in P:  # one patch at a time, like the phone app
                run(p)
            t3 = time.perf_counter()
            rows.append(dict(image_id=r.image_id, method=method, k=k, n_patches=len(P),
                             minutiae_ms=1e3 * t_min, select_ms=1e3 * (t1 - t0),
                             patch_ms=1e3 * (t2 - t1), cnn_ms=1e3 * (t3 - t2)))
    df = pd.DataFrame(rows)
    df["total_ms"] = df[["minutiae_ms", "select_ms", "patch_ms", "cnn_ms"]].sum(1)
    df.to_csv(os.path.join(args.results, "timing.csv"), index=False, float_format="%.3f")
    with open(os.path.join(args.results, "timing_env.json"), "w") as f:
        json.dump(dict(backend=backend, model_bytes=size, cpu=platform.processor() or platform.machine(),
                       python=platform.python_version(), tensorflow=tf.__version__, threads=1,
                       n_images=len(test)), f, indent=1)
    print(df.groupby(["method", "k"])[["n_patches", "cnn_ms", "total_ms"]].mean().round(1))


if __name__ == "__main__":
    main()
