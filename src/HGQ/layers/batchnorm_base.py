import numpy as np
import tensorflow as tf
from keras.src.utils import tf_utils
from tensorflow import keras

from ..utils import warn
from .base import HLayerBase


class HBatchNormBase(HLayerBase):
    def __init__(
        self,
        axis=-1,
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

        super().__init__(**kwargs)
        self.axis = [axis] if not isinstance(axis, (list, tuple)) else list(axis)
        self.momentum = tf.constant(momentum, dtype=self.dtype)
        self.epsilon = tf.constant(epsilon, dtype=self.dtype)
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.moving_mean_initializer = tf.keras.initializers.get(
            moving_mean_initializer
        )
        self.moving_variance_initializer = tf.keras.initializers.get(
            moving_variance_initializer
        )
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.norm_shape = None

    def post_build(self, input_shape):
        super().post_build(input_shape)
        self.axis: list = tf_utils.validate_axis(self.axis, input_shape)
        self._post_build(input_shape)  # type: ignore

    def _post_build(self, input_shape):
        self._reduction_axis = tuple([i for i in range(len(input_shape)) if i not in self.axis])
        output_shape = self.compute_output_shape(input_shape)
        shape = self.norm_shape or tuple([output_shape[i] for i in self.axis])

        if self.center and not getattr(self, "use_bias", False):
            warn(f'`center` in fused BatchNorm can only be used if `use_bias` is True. Setting center to False.', stacklevel=3)
            self.center = False

        if self.center:
            self.bn_beta: tf.Variable = self.add_weight(
                name="bn_beta",
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
            self.moving_mean: tf.Variable = self.add_weight(
                name="moving_mean",
                shape=shape,
                initializer=self.moving_mean_initializer,
                trainable=False,
            )
        else:
            self.bn_beta = self.moving_mean = tf.Variable(0.0, trainable=False)
        if self.scale:
            self.bn_gamma: tf.Variable = self.add_weight(
                name="bn_gamma",
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
            self.moving_variance: tf.Variable = self.add_weight(
                name="moving_variance",
                shape=shape,
                initializer=self.moving_variance_initializer,
                trainable=False,
            )
        else:
            self.bn_gamma = self.moving_variance = tf.Variable(1.0, trainable=False)

    @property
    @tf.function(jit_compile=True)
    def fused_kernel(self):  # type: ignore
        if not self.scale:
            return self.kernel
        scale = self.bn_gamma * tf.math.rsqrt(self.moving_variance + self.epsilon)
        return self.kernel * scale

    @property
    @tf.function(jit_compile=True)
    def fused_bias(self):  # type: ignore
        if not self.center:
            return self.bias
        scale = self.bn_gamma * tf.math.rsqrt(self.moving_variance + self.epsilon)
        return self.bias - self.moving_mean * scale + self.bn_beta

    def adapt_fused_bn_kernel_bw_bits(self, x: tf.Tensor):
        """Adapt the bitwidth of the kernel quantizer to the input tensor, such that each input is represented with approximately the same number of bits after fused batchnormalization."""
        if not self.scale:
            self.kq.adapt_bw_bits(self.kernel)
            return

        fbw = tf.identity(self.kq.fbw)
        self.kq.fbw.assign(tf.ones_like(fbw) * 32)
        z = self.forward(x, training=False, record_minmax=False)
        self.kq.fbw.assign(fbw)

        var = tf.math.reduce_variance(z, axis=self._reduction_axis)
        scale = self.bn_gamma * tf.math.rsqrt(var + self.epsilon)
        fused_kernel = self.kernel * scale
        self.kq.adapt_bw_bits(fused_kernel)


class FakeObj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class HBatchNormalization(HBatchNormBase):

    def __init__(
        self,
        axis=-1,
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
            axis=axis,
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

    def post_build(self, input_shape):
        self.step_counter = tf.Variable(1, trainable=False, dtype=tf.int32, name="step_counter")
        self.use_bias = True
        axis: list = tf_utils.validate_axis(self.axis, input_shape)
        reduction_axis = tuple([i for i in range(len(input_shape)) if i not in axis])
        ker_shape = tuple(1 if i in reduction_axis else n_inp for i, n_inp in enumerate(input_shape))
        self._reduction_axis = reduction_axis
        self.norm_shape = ker_shape
        self.kernel = FakeObj(shape=ker_shape)
        r = super().post_build(input_shape)
        delattr(self, "use_bias")
        delattr(self, "kernel")
        return r

    def forward(self, x, training=None, record_minmax=False):

        if training:
            self.step_counter.assign_add(1)
            var = tf.math.reduce_variance(x, axis=self._reduction_axis, keepdims=True)
            mean = tf.math.reduce_mean(x, axis=self._reduction_axis, keepdims=True)
            self.moving_mean.assign_sub((self.moving_mean - mean) * (1 - self.momentum))
            self.moving_variance.assign_sub((self.moving_variance - var) * (1 - self.momentum))
        else:
            # correction = 1 - tf.pow(self.momentum, tf.cast(self.step_counter, self.dtype))
            mean = self.moving_mean
            var = self.moving_variance

        ker = self.bn_gamma * tf.math.rsqrt(var + self.epsilon)
        bias = self.bn_beta - mean / ker

        if self._do_adapt_kernel_bits and self._delayed_kernel_bits_adaption:
            self.kq.adapt_bw_bits(ker)
            self._delayed_kernel_bits_adaption = False

        qker = self.kq(ker, training=training)  # type: ignore
        qbias = self.paq(bias, training=training)  # type: ignore
        z = qker * x + qbias

        input_bw = self.input_bw
        if input_bw is not None:
            kernel_bw = self._kernel_bw(qker)
            bops = tf.reduce_sum(input_bw * kernel_bw)
            self.bops.assign(bops)
            bops_loss = tf.cast(bops, tf.float32) * self.beta
            self.add_loss(bops_loss)

        return self.paq(z, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def compute_exact_bops(self):
        mean = self.moving_mean
        var = self.moving_variance
        ker = self.bn_gamma * tf.math.rsqrt(var + self.epsilon)
        qker = self.kq(ker, training=False)  # type: ignore
        kn, int_bits, fb = self.kq.get_bits_exact(qker)
        kernel_bw = int_bits + fb  # sign not considered for kernel
        input_bw = self.input_bw_exact
        bops = int(np.sum(input_bw * kernel_bw))
        self.bops.assign(tf.constant(bops, dtype=tf.float32))
        return bops
