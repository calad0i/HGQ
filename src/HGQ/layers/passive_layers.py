import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable
from keras.src.layers.pooling.base_pooling1d import Pooling1D
from keras.src.layers.pooling.base_pooling2d import Pooling2D
from keras.src.utils import conv_utils

from ..utils import apf_to_tuple, tuple_to_apf
from .base import ABSBaseLayer


@register_keras_serializable(package="HGQ")
class PLayerBase(ABSBaseLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_last_layer = False

    @property
    def act_bw(self):
        """Returns the bitwidth of the pre-activation values. Differentiable."""
        shape = (1,) + self.output_shape[1:]
        input_bw = tf.reshape(self.input_bw, shape)
        return tf.ensure_shape(input_bw, shape)

    @property
    def act_container(self) -> str:
        return self.last_layer.act_container

    @property
    def result_container(self) -> str:
        return self.act_container


@register_keras_serializable(package="HGQ")
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


@register_keras_serializable(package="HGQ")
class PReshape(tf.keras.layers.Reshape, PLayerBase):
    pass


@register_keras_serializable(package="HGQ")
class PFlatten(tf.keras.layers.Flatten, PLayerBase):
    pass


@register_keras_serializable(package="HGQ")
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


@register_keras_serializable(package="HGQ")
class PPool2D(PLayerBase, Pooling2D):

    def build(self, input_shape):
        super().build(input_shape)
        self._tf_data_format = conv_utils.convert_data_format(self.data_format, 4)

    @property
    def act_bw(self):
        act_bw = super().input_bw
        return tf.nn.max_pool2d(act_bw, ksize=self.pool_size, strides=self.strides, padding=self.padding.upper(), data_format=self._tf_data_format)


@register_keras_serializable(package="HGQ")
class PPool1D(PLayerBase, Pooling1D):

    def build(self, input_shape):
        super().build(input_shape)
        self._tf_data_format = conv_utils.convert_data_format(self.data_format, 3)

    @property
    def act_bw(self):
        act_bw = super().input_bw
        return tf.nn.max_pool1d(act_bw, ksize=self.pool_size, strides=self.strides, padding=self.padding.upper(), data_format=self._tf_data_format)


@register_keras_serializable(package="HGQ")
class PMaxPooling2D(PPool2D, tf.keras.layers.MaxPool2D):
    pass


@register_keras_serializable(package="HGQ")
class PAveragePooling2D(PPool2D, tf.keras.layers.AvgPool2D):
    pass


@register_keras_serializable(package="HGQ")
class PMaxPooling1D(PPool1D, tf.keras.layers.MaxPool1D):
    pass


@register_keras_serializable(package="HGQ")
class PAveragePooling1D(PPool1D, tf.keras.layers.AvgPool1D):
    pass


PMaxPool2D = PMaxPooling2D
PAvgPool2D = PAveragePooling2D
PMaxPool1D = PMaxPooling1D
PAvgPool1D = PAveragePooling1D


@register_keras_serializable(package="HGQ")
class PDropout(PLayerBase, tf.keras.layers.Dropout):
    pass
