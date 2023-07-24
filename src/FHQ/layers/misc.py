import tensorflow as tf

from ..layers.base import HLayerBase


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
