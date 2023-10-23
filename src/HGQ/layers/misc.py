from typing import Callable
import tensorflow as tf

from ..layers.base import HLayerBase
from ..utils import apf_to_tuple, tuple_to_apf

from keras.layers.merging.base_merge import _Merge
from keras import activations


class HQuantize(HLayerBase):
    def __init__(self, pre_activation_quantizer_config=None, beta=0., **kwargs):
        super().__init__(
            pre_activation_quantizer_config=pre_activation_quantizer_config,
            beta=beta,
            **kwargs
        )

    @tf.function(jit_compile=True)
    def forward(self, x, training=None, record_minmax=None):
        return self.pre_activation_quantizer(x, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape


class HActivation(HLayerBase, tf.keras.layers.Activation):
    def __init__(self, activation, beta=0., pre_activation_quantizer_config=None, **kwargs):
        super().__init__(activation=activation, beta=beta, pre_activation_quantizer_config=pre_activation_quantizer_config, **kwargs)

    def post_build(self, input_shape):
        if self.pre_activation_quantizer_config['rnd_strategy'] == 'auto':
            self.pre_activation_quantizer_config['rnd_strategy'] = 'standard_round'

        if self.pre_activation_quantizer_config['rnd_strategy'] not in ('floor', 3):
            # Jit when using standard round. Very unlikely to have errors.
            self.forward: Callable = tf.function(jit_compile=True)(self.__forward)  # type: ignore
            
        elif self.activation in (activations.relu, activations.linear, activations.relu6, activations.leaky_relu):
            # Jit when activation is more or less linear.
            self.forward: Callable = tf.function(jit_compile=True)(self.__forward)  # type: ignore
    
        else:
            self.forward: Callable = self.__forward
    
        super().post_build(input_shape)

    # @tf.function(jit_compile=True)
    def __forward(self, x, training=None, record_minmax=None):
        x = self.activation(x)  # type: ignore
        return self.pre_activation_quantizer(x, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def table_container(self) -> str:
        if self.activation is not tf.keras.activations.softmax:
            return self.result_container
        return 'ap_fixed<18,8,AP_RND>'  # No bit-match for softmax anyway, just maxout it for now.


class HAdd(HLayerBase, _Merge):

    @tf.function(jit_compile=True)
    def forward(self, inputs, training=None, record_minmax=None):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return self.pre_activation_quantizer(output, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# class HMultply(HLayerBase, _Merge):

#     @tf.function(jit_compile=True)
#     def forward(self, inputs, training=None, record_minmax=None):
#         output = inputs[0]
#         for i in range(1, len(inputs)):
#             output *= inputs[i]
#         return self.pre_activation_quantizer(output, training=training, record_minmax=record_minmax)  # type: ignore

#     def compute_output_shape(self, input_shape):
#         return input_shape[0]
