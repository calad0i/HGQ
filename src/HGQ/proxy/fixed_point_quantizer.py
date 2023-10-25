import re
from typing import Optional, Generator

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

    def __init__(self, keep_negative, bits, integers, RND: str = 'TRN', SAT: str = 'WRAP', overrides: dict | None = None, aggressive=True, accum_bits_bias=None, **kwargs):
        self.keep_negative = tf.Variable(keep_negative, dtype='int8', name='keep_negative', trainable=False)
        self.bits = tf.Variable(bits, dtype='int8', name='bits', trainable=False)
        self.integers = tf.Variable(integers, dtype='int8', name='integers', trainable=False)
        assert self.keep_negative.shape == self.bits.shape == self.integers.shape, f'Shapes mismatch: keep_negative, bits, and integers must have the same shape. Got {self.keep_negative.shape}, {self.bits.shape}, {self.integers.shape}.'

        self.accum_bits_bias = accum_bits_bias
        self.RND = RND
        self.SAT = SAT
        self.aggressive = aggressive
        if not aggressive and SAT == 'WRAP':
            warn('It\'s odd that aggressive=False and SAT=WRAP are used together. Though probably no one should never do this, you are the boss.')
        self.overrides = overrides
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
        return tuple_to_apf(self.result_t_kif, 'TRN', 'WRAP')

    @property
    def result_t_last(self):
        """result_t to be used for the last layer and this layer's outputs."""
        k, i, f = self.result_t_kif
        if self.removable:
            result_t_last = tuple_to_apf((k, i, f), self.RND, self.SAT)
        else:
            if self.SAT != 'WRAP':
                result_t_last = self.accum_t  # If the last layer is kernel-operation type (has accum_t)
                if result_t_last is not None:
                    k, i, _ = apf_to_tuple(result_t_last)
                    result_t_last = tuple_to_apf((k, i, f + 1), 'TRN', self.SAT)
                else:
                    result_t_last = tuple_to_apf((k, i, f + 1), 'TRN', self.SAT)  # If not, last layer is of activation type
            elif self.RND != 'TRN':
                k, i, f = self.result_t_kif
                result_t_last = tuple_to_apf((k, i, f + 1), 'TRN', 'WRAP')  # Take care of the rounding mode
            else:
                result_t_last = tuple_to_apf((k, i, f), 'TRN', 'WRAP')  # Nothing to take care of.
        return result_t_last

    @staticmethod
    def get_last_layer(layer: keras.layers.Layer) -> keras.layers.Layer:
        assert len(layer._inbound_nodes) == 1, f'input_container is only available for layers used only once. {layer.name} is used {len(layer._inbound_nodes)} times.'
        assert not isinstance(layer._inbound_nodes[0].inbound_layers, list), f'input_container is only available for layers with a single input. {layer.name} has {len(layer._inbound_nodes[0].inbound_layers)} inputs.'
        return layer._inbound_nodes[0].inbound_layers

    @staticmethod
    def get_next_layer(layer: keras.layers.Layer) -> Optional[keras.layers.Layer]:
        if len(layer._outbound_nodes) == 0:
            return None
        assert len(layer._outbound_nodes) == 1, f'output_container is only available for layers used only once. {layer.name} is used {len(layer._outbound_nodes)} times.'
        assert not isinstance(layer._outbound_nodes[0].outbound_layer, list), f'output_container is only available for layers with a single output. {layer.name} has {len(layer._outbound_nodes[0].outbound_layer)} outputs.'
        return layer._outbound_nodes[0].outbound_layer

    @property
    def last_quantizer_layer(self) -> 'FixedPointQuantizer':
        """The last quantizer layer in the model. Return None if this is the last quantizer layer."""
        layer = self.get_last_layer(self)
        while not isinstance(layer, FixedPointQuantizer):
            layer = self.get_last_layer(layer)
        return layer

    @property
    def covered_layers(self) -> list[keras.layers.Layer]:
        """Cover the last layer immediately before this quantizer, all all layers after it until second last (inclusive) layer to the next quantizer, or end of model."""
        covered_layers = [self.get_last_layer(self)]
        layer = self.get_next_layer(self)
        if layer is None:
            return covered_layers
        while True:
            next_layer = self.get_next_layer(layer)
            if isinstance(next_layer, FixedPointQuantizer):
                break
            covered_layers.append(layer)
            if next_layer is None:
                break
            layer = next_layer
        return covered_layers

    @property
    def bit_accurate_accum_t(self):
        """bit accurate accum_t to be used for the intermediate results. If the current layer does not have a weight to determine the quantizer, None is returned."""
        last_layer_name = self.get_last_layer(self).name

        if self.overrides is None:
            return None
        if 'layers' not in self.overrides:
            return None
        if last_layer_name not in self.overrides['layers']:
            return None
        if 'weight_t' not in self.overrides['layers'][last_layer_name]:
            return None

        last_layer_conf: dict = self.overrides['layers'][last_layer_name]

        assert 'layers' in self.overrides, 'layers must be specified in hls_config when FixedPointQuantizer is used following a layer requiring accum_t.'

        k1, i1, f1 = apf_to_tuple(last_layer_conf['weight_t'])
        k2, i2, f2 = self.last_quantizer_layer.result_t_kif
        k, i, f = self.result_t_kif
        if self.aggressive:
            accum_t = tuple_to_apf((k, i, f1 + f2))
        else:
            _accum_multiplicity = last_layer_conf.get('_accum_multiplicity', 1024)
            bg_int_bias = int(np.ceil(np.log2(_accum_multiplicity)))
            accum_t = tuple_to_apf((k1 or k2 or k, i1 + i2 + bg_int_bias, f1 + f2))
        return accum_t

    @property
    def accum_t(self):
        if self.accum_bits_bias is None:
            return self.bit_accurate_accum_t
        bit_accurate_acccum_t = self.bit_accurate_accum_t
        if bit_accurate_acccum_t is None:
            return None
        k, i, _ = apf_to_tuple(bit_accurate_acccum_t)
        _, _, f = self.result_t_kif
        return tuple_to_apf((k, i, f + self.accum_bits_bias))

    @property
    def bit_accurate_table_t_and_table_size(self):
        """Return bit-accurate table_t and table_size to be used for the last Activation layer. If the last layer is not Activation, None is returned. If softmax is used, as table_t is not used, None is returned."""
        if not isinstance(layers := self.get_last_layer(self), keras.layers.Activation):
            return None
        activation = layers.activation
        if activation is keras.activations.softmax:
            return None
        else:
            k, i, f = apf_to_tuple(self.result_t_last)
            if self.removable:
                table_t = tuple_to_apf((k, i, f), self.RND, self.SAT)
            else:
                if self.RND == 'TRN':
                    table_t = tuple_to_apf((k, i, f), 'TRN', self.SAT)
                else:
                    table_t = tuple_to_apf((k, i, f + 1), 'TRN', self.SAT)
        inp_k, inp_i, inp_f = self.last_quantizer_layer.result_t_kif
        if activation is tf.keras.activations.sigmoid:
            table_size = int(16 / 2.**-inp_f)  # LUT Range hardcoded to -8 ~ 8, match #fractional bits
        elif activation is tf.keras.activations.tanh:
            table_size = int(8 / 2.**-inp_f)  # LUT Range hardcoded to -4 ~ 4, match #fractional bits
        else:
            table_size = 2**(inp_k + inp_i + inp_f)
        return table_t, int(table_size)

    @property
    def removable(self):
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
        overrides = self.overrides or {'layers': {}}

        # Set result_t. For last layer, special consideration is made based on if this layer will be removed before synth, RND, WRAP, etc...
        result_t = self.result_t
        result_t_last = self.result_t_last
        covered_layers = self.covered_layers
        overrides['layers'].setdefault(covered_layers[0].name, {})['result_t'] = result_t_last
        for layer in self.covered_layers[1:]:
            overrides['layers'].setdefault(layer.name, {})['result_t'] = result_t
        overrides['layers'][self.name] = {'result_t': result_t}

        # Set accum_t, table_t, and table_size when applicable. accum_t is only assume for last_layer with weight_t specified.
        last_layer_name = self.get_last_layer(self).name
        last_layer_config = overrides['layers'][last_layer_name]
        if accum_t := self.accum_t:
            last_layer_config['accum_t'] = accum_t
            last_layer_config['bias_t'] = accum_t
        if table_t_and_size := self.bit_accurate_table_t_and_table_size:
            last_layer_config['table_t'], last_layer_config['table_size'] = table_t_and_size
        conf['overrides'] = overrides
        conf['removable'] = self.removable
        return conf

    @classmethod
    def from_config(cls, config: dict):
        dummy_v = np.full(config.pop('shape'), -128, dtype='int8')
        keep_negative = K.variable(dummy_v, dtype='int8', name='keep_negative')
        bits = K.variable(dummy_v, dtype='int8', name='bits')
        integers = K.variable(dummy_v, dtype='int8', name='integers')
        config.pop('removable', None)
        return cls(keep_negative, bits, integers, **config)
