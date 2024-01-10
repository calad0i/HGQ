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

    def post_build(self, input_shape):
        super().post_build(input_shape)
        self.axis: list = tf_utils.validate_axis(self.axis, input_shape)
        self._post_build(input_shape)  # type: ignore

    def _post_build(self, input_shape):
        self._reduction_axis = tuple([i for i in range(len(input_shape)) if i not in self.axis])
        output_shape = self.compute_output_shape(input_shape)
        shape = tuple([output_shape[i] for i in self.axis])

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
    def fused_kernel(self):
        if not self.scale:
            return self.kernel
        scale = self.bn_gamma * tf.math.rsqrt(self.moving_variance + self.epsilon)
        return self.kernel * scale

    @property
    @tf.function(jit_compile=True)
    def fused_bias(self):
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
