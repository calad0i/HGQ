import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable

from .base import HLayerBase
from .batchnorm_base import HBatchNormBase


@register_keras_serializable(package="HGQ")
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
        kq_conf=None,
        paq_conf=None,
        beta=0.,
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
            kq_conf=kq_conf,
            paq_conf=paq_conf,
            beta=beta,
            **kwargs
        )

    def forward(self, x, training=None, record_minmax=None):

        a, kq = self.jit_forward(x, training=training, record_minmax=record_minmax)  # type: ignore

        input_bw = self.input_bw
        if input_bw is not None:
            kernel_bw = self._kernel_bw(kq)  # type: ignore
            bops = tf.reduce_sum(tf.matmul(input_bw, kernel_bw))
            self.bops.assign(bops)
            bops = tf.cast(bops, tf.float32) * self.beta  # type: ignore
            self.add_loss(tf.convert_to_tensor(bops))
        return a

    @tf.function(jit_compile=True)
    def jit_forward(self, x, training=None, record_minmax=None):

        kq = self.kq(self.fused_kernel, training, False)  # type: ignore
        z = tf.matmul(x, kq)
        if self.use_bias:
            b = self.paq.bias_forward(self.fused_bias, training, self.channel_loc)  # type: ignore
            z = tf.nn.bias_add(z, b)
        z = self.paq(z, training, record_minmax)  # type: ignore
        a = self.activation(z)  # type: ignore

        return a, kq

    @property
    def compute_exact_bops(self):
        kernel_bw = self.kernel_bw_exact
        input_bw = self.input_bw_exact
        bops = np.sum(np.matmul(input_bw, kernel_bw))  # type: ignore
        self.bops.assign(tf.constant(bops, dtype=tf.float32))
        return bops


@register_keras_serializable(package="HGQ")
class HDenseBatchNorm(HDense, HBatchNormBase):

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
        kq_conf=None,
        paq_conf=None,
        beta=0.,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            kq_conf=kq_conf,
            paq_conf=paq_conf,
            beta=beta,
            axis=-1,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            **kwargs,
        )
        self._delayed_kernel_bits_adaption = True

    def forward(self, x, training=None, record_minmax=None):

        if training and self._do_adapt_kernel_bits and self._delayed_kernel_bits_adaption:
            self.adapt_fused_bn_kernel_bw_bits(x)
            self._do_adapt_kernel_bits = False

        if training:
            a, kq = self.bn_train_jit_forward(x, True, record_minmax=record_minmax)  # type: ignore
        else:
            a, kq = self.jit_forward(x, False, record_minmax=record_minmax)  # type: ignore

        input_bw = self.input_bw
        if input_bw is not None:
            kernel_bw = self._kernel_bw(kq)  # type: ignore
            bops = tf.reduce_sum(tf.matmul(input_bw, kernel_bw))
            self.bops.assign(bops)
            bops = tf.cast(bops, tf.float32) * self.beta  # type: ignore
            self.add_loss(tf.convert_to_tensor(bops))
        return a

    @tf.function(jit_compile=True)
    def bn_train_jit_forward(self, x, training, record_minmax=None):

        if self.scale:
            kq = self.kernel
        else:
            kq = self.kq(self.kernel, training, False)  # type: ignore

        z = tf.matmul(x, kq)

        if self.center:
            mean = tf.math.reduce_mean(z, axis=self._reduction_axis) + self.bias
            self.moving_mean.assign(self.momentum * self.moving_mean + (1 - self.momentum) * mean)

        if self.scale:
            var = tf.math.reduce_variance(z, axis=self._reduction_axis)
            self.moving_variance.assign(self.momentum * self.moving_variance + (1 - self.momentum) * var)
            scale = self.bn_gamma * tf.math.rsqrt(var + self.epsilon)
            train_fused_kernel = self.kernel * scale
            kq = self.kq(train_fused_kernel, training, False)  # type: ignore
            z = tf.matmul(x, kq)
        else:
            scale = tf.constant(1.0, dtype=self.dtype)

        if self.center:
            train_fused_bias = self.bias - mean * scale  # type: ignore
            bq = self.paq.bias_forward(train_fused_bias, training, self.channel_loc)  # type: ignore
            z = tf.nn.bias_add(z, bq)  # type: ignore

        z = self.paq(z, training, record_minmax)  # type: ignore
        a = self.activation(z)  # type: ignore

        return a, kq
