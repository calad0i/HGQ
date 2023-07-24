import tensorflow as tf
from keras.layers.convolutional.base_conv import Conv

from .base import HLayerBase
from ..utils import warn


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
        kernel_quantizer_config=None,
        pre_activation_quantizer_config=None,
        bops_reg_factor=0.,
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
            kernel_quantizer_config=kernel_quantizer_config,
            pre_activation_quantizer_config=pre_activation_quantizer_config,
            bops_reg_factor=bops_reg_factor,
            **kwargs,
        )
        if conv_op is not None:
            warn(r'Warning: conv_op is defined. This will override the default convolution operation.')
            self.convolution_op = conv_op
        self.parallel_factor = tf.constant(parallel_factor, dtype=tf.float32)

    def post_build(self, input_shape):
        r = super().post_build(input_shape)
        self.strides = list(self.strides)
        self.dilation_rate = list(self.dilation_rate)
        output_shape = self.compute_output_shape(input_shape)
        self.total_channels = tf.cast(tf.reduce_prod(output_shape[1:-1]), dtype=tf.float32)
        # self._bops_parallel_factor_modifier = self.parallel_factor
        self.channel_loc = 1 if self.data_format == "channels_first" else -1
        self.kernel_quantizer.degeneracy *= float(self.parallel_factor)
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
            bops = tf.cast(bops, tf.float32) * self.bops_reg_factor  # type: ignore
            if self.bops_reg_factor > 0:
                pass
                self.add_loss(tf.convert_to_tensor(bops))
        return a

    @tf.function(jit_compile=True)
    def jit_forward(self, x, training=None, record_minmax=None):

        kq = self.kernel_quantizer(self.kernel, training, False)  # type: ignore
        z = self.convolution_op(x, kq)  # type: ignore
        if self.use_bias:
            b = self.pre_activation_quantizer.bias_forward(self.bias, training)  # type: ignore
            # b = tf.reshape(b, (-1))
            z = tf.nn.bias_add(z, b, data_format=self._tf_data_format)
        z = self.pre_activation_quantizer(z, training, record_minmax)  # type: ignore
        a = self.activation(z)  # type: ignore

        return a, kq

    @property
    def compute_exact_bops(self):
        kernel_bw = tf.constant(self.kernel_bw_exact, dtype=tf.float32)
        input_bw = self.input_bw  # type: ignore
        bops = int(tf.reduce_sum(self.convolution_op(input_bw, kernel_bw)).numpy()) * int(self.parallel_factor.numpy()) / int(self.total_channels.numpy())  # type: ignore
        self.bops.assign(tf.constant(bops, dtype=tf.float32))
        return bops


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
        kernel_quantizer_config=None,
        pre_activation_quantizer_config=None,
        bops_reg_factor=0.,
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
            kernel_quantizer_config=kernel_quantizer_config,
            pre_activation_quantizer_config=pre_activation_quantizer_config,
            bops_reg_factor=bops_reg_factor,
            parallel_factor=parallel_factor,
            **kwargs,
        )


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
        kernel_quantizer_config=None,
        pre_activation_quantizer_config=None,
        bops_reg_factor=0.,
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
            kernel_quantizer_config=kernel_quantizer_config,
            pre_activation_quantizer_config=pre_activation_quantizer_config,
            bops_reg_factor=bops_reg_factor,
            parallel_factor=parallel_factor,
            **kwargs,
        )
