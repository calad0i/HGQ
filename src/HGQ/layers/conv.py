from functools import partialmethod

import tensorflow as tf
from keras.saving import register_keras_serializable
from keras.src.layers.convolutional.base_conv import Conv

from ..utils import warn
from .base import HLayerBase
from .batchnorm_base import HBatchNormBase


class HConv(HLayerBase, Conv):
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        conv_op=None,
        kq_conf=None,
        paq_conf=None,
        beta=0.,
        parallel_factor: int = 1,
        **kwargs,
    ):
        """parallel_factor: number of parallel kernel operation to be used. Only used in bops estimation."""
        super().__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            conv_op=conv_op,
            kq_conf=kq_conf,
            paq_conf=paq_conf,
            beta=beta,
            **kwargs,
        )
        if conv_op is not None:
            warn(r'Warning: conv_op is defined. This will override the default convolution operation.')
            self.convolution_op = conv_op
        self.parallel_factor = tf.constant(parallel_factor, dtype=tf.float32)
        self.channel_loc = 1 if self.data_format == "channels_first" else -1

    def post_build(self, input_shape):
        r = super().post_build(input_shape)
        self.strides = list(self.strides)
        self.dilation_rate = list(self.dilation_rate)
        output_shape = self.compute_output_shape(input_shape)
        self.total_channels = tf.cast(tf.reduce_prod(output_shape[1:-1]), dtype=tf.float32)
        self.kq.degeneracy *= float(self.parallel_factor)
        return r

    @tf.function(jit_compile=True)
    def convolution_op(self, inputs, kernel):
        if self.padding == "causal":
            tf_padding = "VALID"  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding

        return tf.nn.convolution(
            inputs,
            kernel,
            strides=self.strides,
            padding=tf_padding,
            dilations=self.dilation_rate,
            data_format=self._tf_data_format,
            name=self.__class__.__name__,
        )

    def forward(self, x, training=None, record_minmax=None):

        a, kq = self.jit_forward(x, training=training, record_minmax=record_minmax)  # type: ignore

        input_bw = self.input_bw
        if input_bw is not None:
            kernel_bw = self._kernel_bw(kq)  # type: ignore
            bops = tf.reduce_sum(self.convolution_op(self.input_bw, kernel_bw))  # type: ignore
            bops = bops * self.parallel_factor / self.total_channels
            self.bops.assign(bops)
            bops = tf.cast(bops, tf.float32) * self.beta
            self.add_loss(tf.convert_to_tensor(bops))
        return a

    @tf.function(jit_compile=True)
    def jit_forward(self, x, training=None, record_minmax=None):

        kq = self.kq(self.fused_kernel, training, False)  # type: ignore
        z = self.convolution_op(x, kq)  # type: ignore
        if self.use_bias:
            b = self.paq.bias_forward(self.fused_bias, training)  # type: ignore
            z = tf.nn.bias_add(z, b, data_format=self._tf_data_format)
        z = self.paq(z, training, record_minmax)  # type: ignore
        a = self.activation(z)  # type: ignore

        return a, kq

    @property
    def compute_exact_bops(self):
        kernel_bw = tf.constant(self.kernel_bw_exact, dtype=tf.float32)
        input_bw = self.input_bw_exact
        bops = int(tf.reduce_sum(self.convolution_op(input_bw, kernel_bw)).numpy()) * int(self.parallel_factor.numpy()) / int(self.total_channels.numpy())  # type: ignore
        self.bops.assign(tf.constant(bops, dtype=tf.float32))
        return bops


@register_keras_serializable(package="HGQ")
class HConv2D(HConv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        kq_conf=None,
        paq_conf=None,
        beta=0.,
        parallel_factor: int = 1,
        **kwargs,
    ):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            kq_conf=kq_conf,
            paq_conf=paq_conf,
            beta=beta,
            parallel_factor=parallel_factor,
            **kwargs,
        )


@register_keras_serializable(package="HGQ")
class HConv1D(HConv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        kq_conf=None,
        paq_conf=None,
        beta=0.,
        parallel_factor: int = 1,
        **kwargs,
    ):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            kq_conf=kq_conf,
            paq_conf=paq_conf,
            beta=beta,
            parallel_factor=parallel_factor,
            **kwargs,
        )


class HConvBatchNorm(HConv, HBatchNormBase):
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        conv_op=None,
        kq_conf=None,
        paq_conf=None,
        beta=0.,
        parallel_factor: int = 1,
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
        channel_loc = 1 if data_format == "channels_first" else -1
        super().__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            conv_op=conv_op,
            kq_conf=kq_conf,
            paq_conf=paq_conf,
            beta=beta,
            parallel_factor=parallel_factor,
            axis=channel_loc,
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
            bops = tf.reduce_sum(self.convolution_op(self.input_bw, kernel_bw))  # type: ignore
            bops = bops * self.parallel_factor / self.total_channels
            self.bops.assign(bops)
            bops = tf.cast(bops, tf.float32) * self.beta
            self.add_loss(tf.convert_to_tensor(bops))
        return a

    @tf.function(jit_compile=True)
    def bn_train_jit_forward(self, x, training, record_minmax=None):

        if self.scale:
            kq = self.kernel
        else:
            kq = self.kq(self.kernel, training, False)  # type: ignore

        z = self.convolution_op(x, kq)  # type: ignore

        if self.center:
            mean = tf.math.reduce_mean(z, axis=self._reduction_axis) + self.bias
            self.moving_mean.assign(self.momentum * self.moving_mean + (1 - self.momentum) * mean)

        if self.scale:
            var = tf.math.reduce_variance(z, axis=self._reduction_axis)
            self.moving_variance.assign(self.momentum * self.moving_variance + (1 - self.momentum) * var)
            scale = self.bn_gamma * tf.math.rsqrt(var + self.epsilon)
            train_fused_kernel = self.kernel * scale
            kq = self.kq(train_fused_kernel, training, False)  # type: ignore
            z = self.convolution_op(x, kq)  # type: ignore
        else:
            scale = tf.constant(1.0, dtype=self.dtype)

        if self.center:
            train_fused_bias = self.bias - mean * scale  # type: ignore
            bq = self.paq.bias_forward(train_fused_bias, training, self.channel_loc)  # type: ignore
            z = tf.nn.bias_add(z, bq, data_format=self._tf_data_format)  # type: ignore

        z = self.paq(z, training, record_minmax)  # type: ignore
        a = self.activation(z)  # type: ignore

        return a, kq


@register_keras_serializable(package="HGQ")
class HConv2DBatchNorm(HConvBatchNorm):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        conv_op=None,
        kq_conf=None,
        paq_conf=None,
        beta=0.,
        parallel_factor: int = 1,
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
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            conv_op=conv_op,
            kq_conf=kq_conf,
            paq_conf=paq_conf,
            beta=beta,
            parallel_factor=parallel_factor,
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


@register_keras_serializable(package="HGQ")
class HConv1DBatchNorm(HConvBatchNorm):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        conv_op=None,
        kq_conf=None,
        paq_conf=None,
        beta=0.,
        parallel_factor: int = 1,
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
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            conv_op=conv_op,
            kq_conf=kq_conf,
            paq_conf=paq_conf,
            beta=beta,
            parallel_factor=parallel_factor,
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
