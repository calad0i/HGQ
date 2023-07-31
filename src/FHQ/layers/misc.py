import tensorflow as tf

from ..layers.base import HLayerBase

from keras.layers.merging.base_merge import _Merge
from keras import activations


class HQuantize(HLayerBase):
    def __init__(self, pre_activation_quantizer_config=None, bops_reg_factor=0., **kwargs):
        super().__init__(
            pre_activation_quantizer_config=pre_activation_quantizer_config,
            bops_reg_factor=bops_reg_factor,
            **kwargs
        )

    @tf.function(jit_compile=True)
    def forward(self, x, training=None, record_minmax=None):
        return self.pre_activation_quantizer(x, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape


class HActivation(HLayerBase, tf.keras.layers.Activation):
    def __init__(self, activation, bops_reg_factor=0., pre_activation_quantizer_config=None, **kwargs):
        super().__init__(activation=activation, bops_reg_factor=bops_reg_factor, pre_activation_quantizer_config=pre_activation_quantizer_config, **kwargs)

    def post_build(self, input_shape):
        if self.pre_activation_quantizer_config['rnd_strategy'] == 'auto':
            self.pre_activation_quantizer_config['rnd_strategy'] = 'standard_round'
        super().post_build(input_shape)

    @tf.function(jit_compile=True)
    def forward(self, x, training=None, record_minmax=None):
        x = self.activation(x)  # type: ignore
        return self.pre_activation_quantizer(x, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape


class HAdd(HLayerBase, _Merge):

    @tf.function(jit_compile=True)
    def forward(self, inputs, training=None, record_minmax=None):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return self.pre_activation_quantizer(output, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class HMultply(HLayerBase, _Merge):

    @tf.function(jit_compile=True)
    def forward(self, inputs, training=None, record_minmax=None):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output *= inputs[i]
        return self.pre_activation_quantizer(output, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape[0]
