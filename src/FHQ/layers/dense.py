import numpy as np
import tensorflow as tf

from .base import HLayerBase, scale_grad


class HDense(HLayerBase, tf.keras.layers.Dense):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        kernel_quantizer_config=None,
        pre_activation_quantizer_config=None,
        bops_reg_factor=0.,
        **kwargs,
    ):
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            kernel_quantizer_config=kernel_quantizer_config,
            pre_activation_quantizer_config=pre_activation_quantizer_config,
            bops_reg_factor=bops_reg_factor,
            **kwargs
        )

    def forward(self, x, training=None, record_minmax=None):

        a, kq = self.jit_forward(x, training=training, record_minmax=record_minmax)  # type: ignore

        input_bw = self.input_bw
        if input_bw is not None:
            kernel_bw = self._kernel_bw(kq)  # type: ignore
            bops = tf.reduce_sum(tf.matmul(input_bw, kernel_bw))
            self.bops.assign(bops)
            bops = tf.cast(bops, tf.float32) * self.bops_reg_factor  # type: ignore
            if self.bops_reg_factor > 0:
                pass
                self.add_loss(tf.convert_to_tensor(bops))
        return a

    @tf.function(jit_compile=True)
    def jit_forward(self, x, training=None, record_minmax=None):

        kq = self.kernel_quantizer(self.kernel, training, False)  # type: ignore
        z = tf.matmul(x, kq)
        if self.use_bias:
            b = self.pre_activation_quantizer.bias_forward(self.bias, training, self.channel_loc)  # type: ignore
            z = tf.nn.bias_add(z, b)
        z = self.pre_activation_quantizer(z, training, record_minmax)  # type: ignore
        a = self.activation(z)  # type: ignore

        return a, kq

    @property
    def compute_exact_bops(self):
        kernel_bw = self.kernel_bw_exact
        input_bw = self.input_bw.numpy()  # type: ignore
        bops = np.sum(np.matmul(input_bw, kernel_bw))
        self.bops.assign(tf.constant(bops, dtype=tf.float32))
        return bops
