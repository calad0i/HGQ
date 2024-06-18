import numpy as np
import tensorflow as tf


@tf.custom_gradient
def floatlet_quantize(x, m, e, e0=0):
    '''Quantize an array to floatlet (m mantissa bits, e exponent bits) format. Tentative gradient impl.'''
    two = tf.constant(2.0, dtype=tf.float32)
    log2 = tf.math.log(two)
    eps = tf.constant(1e-30, dtype=tf.float32)

    e_req = tf.math.ceil(tf.math.log(tf.abs(x) + eps) / log2)  # type: ignore
    e_low, e_high = -two**e + e0, two**e - 1 + e0
    e_act = tf.clip_by_value(e_req, e_low, e_high)
    scale = two**(e_act - m + 1)
    sig = x / scale
    qsig = tf.floor(sig)
    clip_sig = tf.clip_by_value(qsig, -two**m, two**m - 1)
    qx = clip_sig * scale

    def grad(dy: tf.Tensor):
        dm = scale * (sig - qsig) * dy * log2
        _de = (x - qx) * dy * log2**two
        de = tf.cast(e_req != e_act, tf.float32) * _de
        return dy, dm, de

    return qx, grad


def floatlet_decompose(x, m, e, e0=0):
    eps = 1e-30

    e_req = np.ceil(np.log2(tf.abs(x) + eps))  # type: ignore
    e_low, e_high = -2.**e + e0, 2.**e - 1 + e0
    e_act = np.clip(e_req, e_low, e_high)
    scale = 2.**(e_act - m + 1)
    sig = x / scale
    qsig = np.floor(sig)
    clip_sig = np.clip(qsig, -2.**m, 2.**m - 1)
    return clip_sig, e_act
