"""MobileNet-v1 patch classifier.

Deviation from the proposal: we use the Keras implementation of MobileNet-v1 (tf.keras.applications.MobileNet)
instead of TF-Slim. TF-Slim needs TensorFlow 1.x graph mode, which current Colab no longer ships. The
architecture is the same (alpha=1.0, 1024-d bottleneck); with a 96x96 input the final feature map is 3x3,
and global average pooling over it is exactly the paper's change of the last pool from 7x7 to 3x3.
"""
import keras
from keras import layers


def build_model(size=96, imagenet=True):
    inp = keras.Input((size, size, 1), name="patch")             # grayscale uint8 values as float
    x = layers.Rescaling(1 / 127.5, offset=-1.0)(inp)            # MobileNet expects inputs in [-1, 1]
    x = layers.Concatenate()([x, x, x])                          # ImageNet weights expect 3 channels
    base = keras.applications.MobileNet(
        input_shape=(size, size, 3), alpha=1.0, include_top=False,
        weights="imagenet" if imagenet else None, pooling="avg")  # 3x3 global average pool at 96x96
    x = base(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", name="spoofness")(x)  # 0 = bona fide, 1 = spoof
    return keras.Model(inp, out)
