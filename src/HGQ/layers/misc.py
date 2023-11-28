from collections.abc import Callable

import tensorflow as tf
from keras import activations
from keras.saving import register_keras_serializable
from keras.src.layers.merging.base_merge import _Merge
from tensorflow.python.ops.nn_ops import leaky_relu, relu6

from ..layers.base import HLayerBase
from ..utils import apf_to_tuple, tuple_to_apf


@register_keras_serializable(package="HGQ")
class HQuantize(HLayerBase):
    def __init__(self, paq_conf=None, beta=0., **kwargs):
        super().__init__(
            paq_conf=paq_conf,
            beta=beta,
            **kwargs
        )

    @tf.function(jit_compile=True)
    def forward(self, x, training=None, record_minmax=None):
        return self.paq(x, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape


@register_keras_serializable(package="HGQ")
class HActivation(HLayerBase, tf.keras.layers.Activation):
    def __init__(self, activation, beta=0., paq_conf=None, **kwargs):
        super().__init__(activation=activation, beta=beta, paq_conf=paq_conf, **kwargs)

    def post_build(self, input_shape):
        if self.paq_config['rnd_strategy'] == 'auto':
            self.paq_config['rnd_strategy'] = 'standard_round'

        if self.paq_config['rnd_strategy'] not in ('floor', 3):
            # Jit when using standard round. Very unlikely to have errors.
            self.forward: Callable = tf.function(jit_compile=True)(self.__forward)  # type: ignore

        elif self.activation in (activations.relu, activations.linear, relu6, leaky_relu):
            # tf 213 scatters everthing around. Terrible design choice.
            # Jit when activation is more or less linear.
            self.forward: Callable = tf.function(jit_compile=True)(self.__forward)  # type: ignore

        else:
            self.forward: Callable = self.__forward

        super().post_build(input_shape)

    def __forward(self, x, training=None, record_minmax=None):
        x = self.activation(x)  # type: ignore
        return self.paq(x, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape


@register_keras_serializable(package="HGQ")
class HAdd(HLayerBase, _Merge):

    @tf.function(jit_compile=True)
    def forward(self, inputs, training=None, record_minmax=None):
        output = tf.reduce_sum(inputs, axis=0)
        return self.paq(output, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# class HMultply(HLayerBase, _Merge):

#     @tf.function(jit_compile=True)
#     def forward(self, inputs, training=None, record_minmax=None):
#         output = inputs[0]
#         for i in range(1, len(inputs)):
#             output *= inputs[i]
#         return self.paq(output, training=training, record_minmax=record_minmax)  # type: ignore

#     def compute_output_shape(self, input_shape):
#         return input_shape[0]
