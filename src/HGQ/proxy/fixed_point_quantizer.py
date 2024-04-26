import abc
from collections.abc import Callable
from warnings import warn

import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow import keras

# Nice figure (Figure. 2 and 3) from https://www.researchgate.net/publication/226964494_Formalization_of_Fixed-Point_Arithmetic_in_HOL to illustrate the rounding and saturation modes.


def TRN(x):
    # Truncate towards negative infinity. Fast. Preferred when possible.
    return tf.floor(x)


def RND(x):
    # Round to nearest, ties to even.
    # Can be reduced to TRN with a bias.
    return tf.floor(x + 0.5)  # type:ignore


def RND_CONV(x):
    # towards nearest integer, ties to even.
    return tf.round(x)


def TRN_ZERO(x):
    # Truncate towards zero.
    sign = K.sign(x)
    return tf.floor(K.abs(x)) * sign


def RND_ZERO(x):
    # Round to nearest, ties to zero.
    sign = K.sign(x)
    return -tf.floor(-K.abs(x) + 0.5) * sign


def RND_MIN_INF(x):
    # Round to nearest, ties to negative infinity.
    return -tf.floor(-x + 0.5)  # type:ignore


def RND_INF(x):
    # Round to nearest, ties away from zero.
    sign = K.sign(x)
    return tf.floor(K.abs(x) + 0.5) * sign


def SAT(x, k, b):
    # Saturate between highest and lowest representable values.
    # Slow, but less trash results when overflows.
    high = 2**(b - k) - 1
    low = -(high + 1) * k
    return tf.clip_by_value(x, low, high)


def SAT_ZERO(x, k, b):
    # Overflow to zero.
    # Slow, and trash results when overflows. Why would anyone use this?
    high = 2**(b - k) - 1
    low = (-high - 1) * k
    mask = tf.cast((x <= high) & (x >= low), 'float32')
    return x * mask


def SAT_SYM(x, k, b):
    # Saturate between highest and lowest representable values when unsigned; between highest and -highest when signed.
    # Slow, but less trash results when overflows. Smaller representable range than SAT.
    high = 2**(b - k) - 1
    low = -high * k
    return tf.clip_by_value(x, low, high)


def WRAP(x, k, b):
    # Wrap around.
    # Faaaast, but trash results when overflows. YOLO.
    high = 2**(b - k) - 1
    low = -(high + 1) * k
    return tf.math.floormod(x - low, high - low + 1) + low


def WRAP_SM(x, k, b):
    # High and low bounds are reflective. When overflows, can be less trash than WARP but still more trash than SAT.
    dtype = x.dtype
    high = 2**(b - k) - 1
    low = -(high + 1) * k
    interval = (high - low + 1) * 2
    mapped = K.cast(tf.math.floormod(x - high - 1, interval), 'float32')
    return K.cast(K.abs(mapped - interval / 2 + 0.5) - 0.5 + low, dtype)


RND_MAP = {
    'RND': RND,
    'RND_ZERO': RND_ZERO,
    'RND_MIN_INF': RND_MIN_INF,
    'RND_INF': RND_INF,
    'RND_CONV': RND_CONV,
    'TRN_ZERO': TRN_ZERO,
    'TRN': TRN,
}

SAT_MAP = {
    'SAT': SAT,
    'SAT_ZERO': SAT_ZERO,
    'SAT_SYM': SAT_SYM,
    'WRAP': WRAP,
    'WRAP_SM': WRAP_SM,
}


@tf.function(autograph=False, jit_compile=True)
def gfixed_quantizer(x, keep_negative, bits, integer_bits, RND='TRN', SAT='WRAP'):
    '''Generalized fixed point quantizer, should have the same behavior to ap_fixed/ap_ufixed. Support high granularity quantization and broadcasting of bitwidths. RND and SAT mode must be strings.'''

    keep_negative = tf.cast(keep_negative, 'float32')
    bits = tf.cast(bits, 'float32')
    integer_bits = tf.cast(integer_bits, dtype='float32')

    two = tf.constant(2, dtype='float32')
    float_bits = bits - integer_bits  # type:ignore
    scale = tf.pow(two, float_bits)

    scaled_input = x * scale
    rnd, sat = RND_MAP[RND], SAT_MAP[SAT]
    quantized = sat(rnd(scaled_input), keep_negative, bits)
    return quantized / scale * tf.cast(bits != 0, 'float32')


def gfixed(keep_negative, bits, integer_bits, RND='TRN', SAT='WRAP') -> Callable:
    def compute(x):
        return gfixed_quantizer(x, keep_negative, bits, integer_bits, RND, SAT)  # type:ignore
    return compute


def ufixed(bits, integer_bits, RND='TRN', SAT='WRAP') -> Callable:
    return gfixed(0, bits, integer_bits, RND, SAT)


def fixed(bits, integer_bits, RND='TRN', SAT='WRAP') -> Callable:
    return gfixed(1, bits, integer_bits, RND, SAT)


@keras.utils.register_keras_serializable(package='HGQ')
class FixedPointQuantizer(keras.layers.Layer, metaclass=abc.ABCMeta):

    def __init__(self, keep_negative, bits, integers, RND: str = 'TRN', SAT: str = 'WRAP', overrides: dict | None = None, **kwargs):

        zeros = bits == 0

        if not hasattr(keep_negative, 'shape'):
            keep_negative = np.int8(keep_negative)
            keep_negative = tf.constant([keep_negative], dtype='int8')
        if not hasattr(bits, 'shape'):
            bits = np.int8(bits)
            bits = tf.constant([bits], dtype='int8')
        if not hasattr(integers, 'shape'):
            integers = np.int8(integers)
            integers = tf.constant([integers], dtype='int8')

        keep_negative = tf.where(zeros, tf.zeros_like(keep_negative), keep_negative)
        integers = tf.where(zeros, tf.zeros_like(integers), integers)
        self.keep_negative = tf.Variable(keep_negative, dtype='int8', name='keep_negative', trainable=False)
        self.bits = tf.Variable(bits, dtype='int8', name='bits', trainable=False)
        self.integers = tf.Variable(integers, dtype='int8', name='integers', trainable=False)
        assert self.keep_negative.shape == self.bits.shape == self.integers.shape, f'Shapes mismatch: keep_negative, bits, and integers must have the same shape. Got {self.keep_negative.shape}, {self.bits.shape}, {self.integers.shape}.'

        self.RND = RND
        self.SAT = SAT
        self.overrides = overrides or {'layers': {}}
        kwargs.pop('trainable', None)
        self._quantizer_created = False

        super().__init__(trainable=False, **kwargs)

    def call(self, x, training=None):  # type:ignore
        assert not training, "Proxy model shall can not be trained!"
        if not self.built:
            self.build(x.shape)
        return gfixed_quantizer(x, self.keep_negative, self.bits, self.integers, self.RND, self.SAT)  # type:ignore

    @property
    def result_t_kif(self):
        k, i, f = self.keep_negative, self.integers - self.keep_negative, self.bits - self.integers  # type:ignore
        k, i, f = np.max(k), np.max(i), np.max(f)  # type:ignore
        return k, i, f

    @property
    def fusible(self):
        """Delete this quantizer if no heterogeneity is detected."""
        assert len(self._inbound_nodes) == 1, 'FixedPointQuantizer must not be reused. Create proxy model only via proviced functions.'
        last_layer = self._inbound_nodes[0].inbound_layers
        assert not isinstance(last_layer, list), f'FixedPointQuantizer has exactly one inbound layer. Got a list of {len(last_layer)} layers.'
        if len(last_layer._outbound_nodes) != 1:
            return False
        return not self.heterogeneous

    @property
    def heterogeneous(self):
        k0, b0, i0 = tf.reduce_max(self.keep_negative), tf.reduce_max(self.bits), tf.reduce_max(self.integers)
        if not tf.reduce_all(self.keep_negative == k0):
            return True
        if not tf.reduce_all(self.bits == b0):
            return True
        if not tf.reduce_all(self.integers == i0):
            return True
        return False

    def get_config(self):
        assert tf.reduce_all((self.keep_negative == 0) | (self.keep_negative == 1)), 'Illegal bitwidth config: keep_negative must be 0 or 1.'
        if not tf.reduce_all(self.bits >= 0):  # type:ignore
            warn('Illegal bitwidth config: bits must be non-negative.')
            self.bits.assign(tf.maximum(self.bits, 0))
        conf = super().get_config()
        conf['RND'] = self.RND
        conf['SAT'] = self.SAT
        conf['shape'] = tuple(self.bits.shape)
        overrides = self.overrides

        conf['overrides'] = overrides
        conf['fusible'] = self.fusible
        return conf

    @classmethod
    def from_config(cls, config: dict):
        dummy_v = np.full(config.pop('shape'), -128, dtype='int8')
        keep_negative = K.variable(dummy_v, dtype='int8', name='keep_negative')
        bits = K.variable(dummy_v, dtype='int8', name='bits')
        integers = K.variable(dummy_v, dtype='int8', name='integers')
        config.pop('fusible', None)
        return cls(keep_negative, bits, integers, **config)
