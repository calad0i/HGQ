import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from ..utils import apf_to_tuple, tuple_to_apf, warn

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


def WRAP_SYM(x, k, b):
    # High and low bounds are reflective. When overflows, can be less trash than WARP but still more trash than SAT.
    dtype = x.dtype
    high = 2**(b - k) - 1
    low = -(high + 1) * k
    interval = (high - low + 1) * 2
    mapped = K.cast(tf.math.floormod(x - high - 1, interval), 'float32')
    return K.cast(K.abs(mapped - interval / 2 + 0.5) - 0.5 + low, dtype)


from typing import Callable

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
    'WRAP_SYM': WRAP_SYM,
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


class FixedPointQuantizer(keras.layers.Layer):

    def __init__(self, keep_negative, bits, integers, RND: str = 'TRN', SAT: str = 'WRAP', hls_config: dict | None = None, aggressive=True, **kwargs):
        self.keep_negative = tf.Variable(keep_negative, dtype='int8', name='keep_negative', trainable=False)
        self.bits = tf.Variable(bits, dtype='int8', name='bits', trainable=False)
        self.integers = tf.Variable(integers, dtype='int8', name='integers', trainable=False)
        assert self.keep_negative.shape == self.bits.shape == self.integers.shape, f'Shapes mismatch: keep_negative, bits, and integers must have the same shape. Got {self.keep_negative.shape}, {self.bits.shape}, {self.integers.shape}.'

        self.RND = RND
        self.SAT = SAT
        self.aggressive = aggressive
        if not aggressive and SAT == 'WRAP':
            warn('aggressive=False and SAT=WRAP seems to be odd.')
        self.hls_config = hls_config
        kwargs.pop('trainable', None)
        self._quantizer_created = False

        super().__init__(trainable=False, **kwargs)

    def call(self, x):
        if not self.built:
            self.build(x.shape)
        return gfixed_quantizer(x, self.keep_negative, self.bits, self.integers, self.RND, self.SAT)  # type:ignore

    @property
    def result_t_kif(self):
        k, i, f = self.keep_negative, self.integers - self.keep_negative, self.bits - self.integers  # type:ignore
        k, i, f = np.max(k), np.max(i), np.max(f)  # type:ignore
        return k, i, f

    @property
    def result_t(self):
        """result_t to be used for the last layer and this layer's outputs."""
        k, i, f = self.result_t_kif
        u = '' if k != 0 else 'u'
        if self.delete_me:
            return f'{u}fixed<{i+i+f},{k+i},{self.RND},{self.SAT}>'
        else:
            return f'{u}fixed<{i+i+f},{k+i},TRN,WRAP>'

    @staticmethod
    def last_layer(layer: keras.layers.Layer):
        assert len(layer._inbound_nodes) == 1, f'input_container is only available for layers used only once. {layer.name} is used {len(layer._inbound_nodes)} times.'
        assert not isinstance(layer._inbound_nodes[0].inbound_layers, list), f'input_container is only available for layers with a single input. {layer.name} has {len(layer._inbound_nodes[0].inbound_layers)} inputs.'
        return layer._inbound_nodes[0].inbound_layers

    @property
    def last_quantizer_layer(self):
        """The last quantizer layer in the model. Return None if this is the last quantizer layer."""
        layer = self.last_layer(self)
        while not isinstance(layer, FixedPointQuantizer):
            layer = self.last_layer(layer)
        return layer

    @property
    def bit_accurate_accum_t(self):
        """bit accurate accum_t to be used for the intermediate results. If the current layer does not have a weight to determine the quantizer, None is returned."""
        if self.hls_config is None:
            return None
        if 'last_layer' not in self.hls_config:
            return None
        if 'weight_t' not in self.hls_config['last_layer']:
            return None

        assert 'last_layer' in self.hls_config, 'last_layer must be specified in hls_config when FixedPointQuantizer is used following a layer requiring accum_t.'

        k1, i1, f1 = apf_to_tuple(self.hls_config['last_layer']['weight_t'])
        k2, i2, f2 = self.last_quantizer_layer.result_t_kif
        k, i, f = self.result_t_kif
        if self.aggressive:
            accum_t = tuple_to_apf((k, i, f1 + f2))
        else:
            _accum_multiplicity = self.hls_config['last_layer'].get('_accum_multiplicity', 1024)
            bg_int_bias = int(np.ceil(np.log2(_accum_multiplicity)))
            accum_t = tuple_to_apf((k1 or k2 or k, i1 + i2 + bg_int_bias, f1 + f2))
        return accum_t

    @property
    def bit_accurate_table_t_and_table_size(self):
        """Return bit-accurate table_t and table_size to be used for the last Activation layer. If the last layer is not Activation, None is returned. If softmax is used, fixed<18,8,RND,SAT> is returned for table_t (not used) and table_size is computed as usual."""
        if not isinstance(last_layer := self.last_layer(self), keras.layers.Activation):
            return None
        activation = last_layer.activation
        if activation is keras.activations.softmax:
            table_t = 'fixed<18,8,RND,SAT>'
        else:
            table_t = self.result_t
        k, i, f = self.result_t_kif
        if activation is tf.keras.activations.sigmoid:
            table_size = int(16 / 2.**-f)  # LUT Range hardcoded to -8 ~ 8, match #fractional bits
        elif activation is tf.keras.activations.tanh:
            table_size = int(8 / 2.**-f)  # LUT Range hardcoded to -4 ~ 4, match #fractional bits
        else:
            table_size = 2**(k + i + f)
        return table_t, table_size

    @property
    def delete_me(self):
        """Delete this quantizer if no heterogeneity is detected."""
        if np.prod(self.bits.shape) == 1:
            return True
        k0, b0, i0 = tf.reduce_max(self.keep_negative), tf.reduce_max(self.bits), tf.reduce_max(self.integers)
        if tf.reduce_all(self.keep_negative == k0) and tf.reduce_all(self.bits == b0) and tf.reduce_all(self.integers == i0):
            return True
        return False

    def get_config(self):
        assert tf.reduce_all((self.keep_negative == 0) | (self.keep_negative == 1)), 'Illegal bitwidth config: keep_negative must be 0 or 1.'
        assert tf.reduce_all(self.bits >= 0), 'Illegal bitwidth config: bits must be non-negative.'  # type:ignore
        conf = super().get_config()
        conf['RND'] = self.RND
        conf['SAT'] = self.SAT
        conf['shape'] = tuple(self.bits.shape)
        hls_config = self.hls_config or {'last_layer': {}}
        hls_config['last_layer']['result_t'] = self.result_t
        if accum_t := self.bit_accurate_accum_t:
            hls_config['last_layer']['accum_t'] = accum_t
            hls_config['last_layer']['bias_t'] = accum_t
        if table_t_and_size := self.bit_accurate_table_t_and_table_size:
            hls_config['last_layer']['table_t'], hls_config['last_layer']['table_size'] = table_t_and_size
        hls_config['last_layer']['name'] = self.name.replace('_quantizer', '')
        conf['hls_config'] = hls_config
        return conf

    @classmethod
    def from_config(cls, config: dict):
        dummy_v = np.full(config.pop('shape'), -128, dtype='int8')
        keep_negative = K.variable(dummy_v, dtype='int8', name='keep_negative')
        bits = K.variable(dummy_v, dtype='int8', name='bits')
        integers = K.variable(dummy_v, dtype='int8', name='integers')
        return cls(keep_negative, bits, integers, **config)
