import numpy as np
import tensorflow as tf

from keras.layers.pooling.base_pooling2d import Pooling2D
from keras.layers.pooling.base_pooling1d import Pooling1D
from keras.utils import conv_utils

from ..utils import apf_to_tuple, tuple_to_apf


class PLayerBase(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_last_layer = False

    @property
    def last_layer(self):
        if not self._has_last_layer:
            if len(self._inbound_nodes) != 1:
                raise ValueError('input_container is only available for layers with a single input.')
            self._has_last_layer = True
        return self._inbound_nodes[0].inbound_layers

    @property
    def act_bw(self):
        """Returns the bitwidth of the pre-activation values. Differentiable."""
        shape = (1,) + self.output_shape[1:]
        input_bw = tf.reshape(self.input_bw, shape)
        return tf.ensure_shape(input_bw, shape)

    @property
    def input_bw(self):
        return self.last_layer.act_bw

    @property
    def act_container(self) -> str:
        return self.last_layer.act_container

    @property
    def result_container(self) -> str:
        return self.act_container

    @property
    def pre_activation_quantizer(self):
        return self.last_layer.pre_activation_quantizer


class Signature(PLayerBase):

    def __init__(self, keep_negative, bits, int_bits, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits
        self.int_bits = int_bits
        self.keep_negative = keep_negative

    def build(self, input_shape):
        self.built = True
        self.bits = tf.broadcast_to(self.bits, (1,) + input_shape[1:])
        self.int_bits = tf.broadcast_to(self.int_bits, (1,) + input_shape[1:])
        self.keep_negative = tf.broadcast_to(self.keep_negative, (1,) + input_shape[1:])

    @tf.function(jit_compile=True)
    def call(self, x, training=None, record_minmax=None):
        return x

    @property
    @tf.function(jit_compile=True)
    def act_bw(self):
        return tf.keras.backend.cast_to_floatx(self.bits)

    @property
    def act_container(self) -> str:
        k = self.keep_negative.numpy().max().astype(bool).item()  # type: ignore
        i = self.int_bits.numpy().max().astype(int).item()  # type: ignore
        b = self.bits.numpy().max().astype(int).item()  # type: ignore
        f = b - i - k
        return tuple_to_apf((k, i, f))

    @property
    def input_bw(self):
        raise ValueError('Signature layer does not have input_bw')

    def get_config(self):
        return {
            'name': self.name,
            'cat': self.keep_negative.numpy().tolist(),  # type: ignore
            'bits': self.bits.numpy().tolist(),  # type: ignore
            'int_bits': self.int_bits.numpy().tolist(),  # type: ignore
        }


class PReshape(tf.keras.layers.Reshape, PLayerBase):
    pass


class PFlatten(tf.keras.layers.Flatten, PLayerBase):
    pass


class PConcatenate(tf.keras.layers.Concatenate, PLayerBase):

    @property
    @tf.function(jit_compile=True)
    def act_bw(self):
        """Returns the bitwidth of the pre-activation values. Differentiable."""
        shape = (1,) + self.output_shape[1:]
        return tf.concat(self.input_bw, axis=self.axis)

    @property
    def input_bw(self):
        if len(self._inbound_nodes) <= 1:
            raise ValueError("Concatenate layer should have at least two inputs")
        input_bws = [l.act_bw for l in self._inbound_nodes[0].inbound_layers]
        return tf.concat(input_bws, axis=self.axis)

    @property
    def act_container(self) -> str:
        if not self._inbound_nodes:
            raise ValueError(f"Layer {self.name} does not have inbound nodes")

        input_containers = np.array([apf_to_tuple(l.act_container) for l in self._inbound_nodes[0].inbound_layers])

        container = tuple_to_apf(tuple(np.max(input_containers, axis=0)))
        return container


class PPool2D(PLayerBase, Pooling2D):

    def build(self, input_shape):
        super().build(input_shape)
        self._tf_data_format = conv_utils.convert_data_format(self.data_format, 4)

    @property
    def act_bw(self):
        act_bw = super().input_bw
        return tf.nn.max_pool2d(act_bw, ksize=self.pool_size, strides=self.strides, padding=self.padding.upper(), data_format=self._tf_data_format)


class PPool1D(PLayerBase, Pooling1D):

    def build(self, input_shape):
        super().build(input_shape)
        self._tf_data_format = conv_utils.convert_data_format(self.data_format, 3)

    @property
    def act_bw(self):
        act_bw = super().input_bw
        return tf.nn.max_pool1d(act_bw, ksize=self.pool_size, strides=self.strides, padding=self.padding.upper(), data_format=self._tf_data_format)


class PMaxPool2D(PPool2D, tf.keras.layers.MaxPool2D):
    pass


class PAvgPool2D(PPool2D, tf.keras.layers.MaxPool2D):
    pass


class PMaxPool1D(PPool1D, tf.keras.layers.MaxPool1D):
    pass


class PAvgPool1D(PPool1D, tf.keras.layers.AvgPool1D):
    pass
