import os
from collections.abc import Callable

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Layer

from HGQ.proxy.fixed_point_quantizer import gfixed_quantizer
from HGQ.proxy.precision_derivation import get_input_kifs, get_produced_kif, get_result_kifRS
from HGQ.utils import apf_to_tuple, tuple_to_apf

LUT_SIZE_LIMITATION = int(os.environ.get('LUT_SIZE_LIMITATION', 2**12))


@keras.utils.register_keras_serializable(package='HGQ')
class UnaryLUT(Layer):
    proxy_ready = True

    def __init__(self, kif_in: tuple[int, int, int], kif_out: tuple[int, int, int], RND='TRN', SAT='WRAP', **kwargs):
        assert sum(kif_in) > 0, 'Input to activation is constantly zero'
        assert sum(kif_out) > 0, 'Output of activation is constantly zero'
        if LUT_SIZE_LIMITATION > 0:
            assert 2**sum(kif_in) < LUT_SIZE_LIMITATION, f'Input to activation is too large ({2**sum(kif_in)} > {LUT_SIZE_LIMITATION}). If you want to raise this limit, set the LUT_SIZE_LIMITATION environment variable.'
        self.kif_in = kif_in
        self.kif_out = kif_out
        k, i, f = kif_in
        self.scale = 2. ** f
        self.table = None
        if (table := kwargs.pop('table', None)) is not None:
            k, i, f, = kif_out
            k, b, i = k, k + i + f, k + i
            table = gfixed_quantizer(table, k, b, i, RND, SAT)  # type:ignore
            self.table = tf.Variable(table, dtype='float32', trainable=False, name='table')
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):  # type:ignore
        if not self.built:
            self.build(inputs.shape)
        inputs = tf.round(inputs * self.scale)
        inputs = inputs % self.table.shape[0]  # type:ignore
        return tf.gather(self.table, tf.cast(inputs, 'int32'))

    def build(self, input_shape):
        super().build(input_shape)
        if self.table is not None:
            return
        N = 2**sum(self.kif_in)
        self.table = tf.Variable(tf.zeros(N), dtype='float32', trainable=False, name='table')

    @classmethod
    def from_activation(cls, activation: Layer | Callable, kif_in=None, kifRS_out=None):

        if kif_in is None:
            kifs_in = get_input_kifs(activation)
            assert len(kifs_in) == 1, f'Activation function {activation} has more than one input. Please specify the input dtype.'
            kif_in = kifs_in[0]

        kifRS_out = kifRS_out or get_result_kifRS(activation)
        kif_out = kifRS_out[:3]
        R, S = kifRS_out[-2:]

        k, i, f = kif_in
        kif_in = k, i, f
        assert k + i + f > 0, 'Activation function is applied to an zero array. Something is wrong.'
        N = 2**(k + i + f)
        assert N < int(os.environ.get('HLS_MAX_ACTIVATION_LUT_SIZE', 2**16)), f'Input to activation function is too large ({N} > {os.environ.get("HLS_MAX_ACTIVATION_LUT_SIZE", 2**16)}). If you want to raise this limit, set the HLS_MAX_ACTIVATION_LUT_SIZE environment variable.'
        if k:
            inp_table = np.empty(N, dtype=np.float64)
            inp_table[:N // 2] = np.linspace(0, 2.**i - 2.**-f, N // 2, dtype=np.float64)
            inp_table[N // 2:] = inp_table[:N // 2] - 2.**i
        else:
            inp_table = np.linspace(-2.**i * k, 2.**i - 2.**-f, N, dtype=np.float64)
        table: np.ndarray = np.array(activation(inp_table), dtype=np.float32)
        return cls(kif_in, kif_out, table=table, RND=R, SAT=S)

    def get_config(self):
        config = super().get_config()
        config.update({
            'kif_in': self.kif_in,
            'kif_out': self.kif_out,
        })
        return config


def xfr_to_unary_lut(layer: keras.layers.Layer, max_table_size=1024):
    if not isinstance(layer, keras.layers.Activation):
        return layer
    if layer.activation is keras.activations.softmax:
        return layer  # simply doesn't work
    if layer.activation in (keras.activations.relu, keras.activations.linear):
        return layer  # not necessary
    kifs_in = get_input_kifs(layer)
    if len(kifs_in) > 1:
        return layer
    if 2**sum(*kifs_in) > max_table_size:
        return layer
    kif_in = kifs_in[0]

    return UnaryLUT.from_activation(layer, kif_in=kif_in)


get_produced_kif.register(UnaryLUT, lambda x: x.kif_out)
