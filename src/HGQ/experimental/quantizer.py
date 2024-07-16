import numpy as np
import tensorflow as tf


@tf.custom_gradient
def floatlet_quantize(x, m, e, e0=0):
    '''Quantize an array to floatlet (m mantissa bits, excl. sign bit, e exponent bits) format. Tentative gradient impl.'''
    m = m + 1
    two = tf.constant(2.0, dtype=tf.float32)
    log2 = tf.math.log(two)
    eps = tf.constant(1e-30, dtype=tf.float32)

    e_req = tf.math.floor(tf.math.log(tf.abs(x) + eps) / log2)  # type: ignore
    _e_high = two**(e - 1)
    e_low, e_high = -_e_high + e0, _e_high + e0 - 1
    e_act = tf.clip_by_value(e_req, e_low + 1, e_high)
    scale = two**(e_act - m + 1)
    sig = x / scale
    qsig = tf.floor(sig + 0.5)
    clip_sig = tf.clip_by_value(qsig, -two**m + 1, two**m - 1)
    qx = clip_sig * scale * tf.cast(e_req > e_low, tf.float32)

    def grad(dy: tf.Tensor):
        dm = scale * (sig - qsig) * dy * log2
        _de = (x - qx) * dy * log2**two
        de = tf.cast(e_req != e_act, tf.float32) * _de
        return dy, dm, de

    return qx, grad


def floatlet_decompose(x, m, e, e0=0):
    eps = 1e-30

    e_req = np.floor(np.log2(tf.abs(x) + eps)) + 1  # type: ignore
    e_low, e_high = -2.**e + e0, 2.**e - 1 + e0
    e_act = np.clip(e_req, e_low, e_high)
    scale = 2.**(e_act - m + 2)
    sig = x / scale - 1
    qsig = np.round(sig)
    clip_sig = np.clip(qsig, -2.**m, 2.**m - 1) / 2**(m - 1)
    return clip_sig, e_act
