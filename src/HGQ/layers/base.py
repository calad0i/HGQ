from collections.abc import Callable

import numpy as np
import tensorflow as tf
from keras.activations import linear, relu

from ..quantizer import HGQ
from ..utils import apf_to_tuple, get_default_kq_conf, get_default_paq_conf, tuple_to_apf, warn


@staticmethod  # type: ignore
@tf.function(jit_compile=True)
def scale_grad(x, scale):
    sx = x * scale
    return sx + tf.stop_gradient(x - sx)


class ABSBaseLayer(tf.keras.layers.Layer):

    @property
    def last_layer(self):
        assert len(self._inbound_nodes) == 1, f'input_container is only available for layers used only once. {self.name} is used {len(self._inbound_nodes)} times.'
        assert not isinstance(self._inbound_nodes[0].inbound_layers, list), f'input_container is only available for layers with a single input. {self.name} has {len(self._inbound_nodes[0].inbound_layers)} inputs.'
        return self._inbound_nodes[0].inbound_layers

    @property
    def input_bw(self):
        assert len(self._inbound_nodes) <= 1, f"Layer {self.name} is reused {len(self._inbound_nodes)} times. This is not allowed."
        try:
            return self.last_layer.act_bw
        except AssertionError:
            return None

    @property
    def input_bw_exact(self):
        assert len(self._inbound_nodes) <= 1, f"Layer {self.name} is reused {len(self._inbound_nodes)} times. This is not allowed."
        return self.last_layer.act_bw_exact.astype(np.float32)


class HLayerBase(ABSBaseLayer):
    """Abstract base class for all layers in the library. Child classes should call post_build() after calling their build() method.
    """

    def __init__(self, kq_conf=None, paq_conf=None, beta=0., **kwargs):
        self._has_kernel = None
        self._has_bias = None
        self.kq_config = kq_conf or get_default_kq_conf()
        "kernel quantizer config"
        self.paq_config = paq_conf or get_default_paq_conf()
        "pre-activation quantizer config"
        self.beta = tf.Variable(beta, dtype=tf.float32, name='beta', trainable=False)
        "BOPs-regularization strength"
        self.record_minmax = False
        self._has_last_layer = False
        self._do_adapt_kernel_bits = kwargs.pop('do_adapt_kernel_bits', True)
        self._delayed_kernel_bits_adaption = False

        activation = kwargs.get('activation', None)
        if activation is not None and type(self).__name__ != 'HActivation':
            if isinstance(activation, str):
                activation = activation.lower()
            elif isinstance(activation, Callable):
                activation = activation.__name__.lower()
            assert activation in ['relu', 'linear'], f'activation other than relu and linear are only supported for HActivation layer, but is defined for {type(self).__name__} layer.'

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.post_build(input_shape)

    @property
    def can_bias_cover_rnd(self):
        if not self._has_bias:
            return False
        quantizer_shape: tuple[int, ...] = tuple(self.paq.fbw.shape)
        bias_shape: tuple[int, ...] = tuple(self.bias.shape)
        if len(bias_shape) != 1:
            warn(f'bias shape {bias_shape} is not supported.')
            return False
        if np.prod(quantizer_shape) == 1:
            return True
        if self.channel_loc == -1:
            return np.prod(quantizer_shape[:-1]) == 1 and bias_shape == quantizer_shape[-1:]
        elif self.channel_loc == 1:
            return np.prod(quantizer_shape[1:]) == 1 and bias_shape == quantizer_shape[0:1]
        return False

    @property
    def _relu_act(self):
        return hasattr(self, 'activation') and self.activation is tf.keras.activations.relu

    def post_build(self, input_shape):
        """This method should be called after calling build() method of the child class. It initializes the quantizers and sets the bops variable, and set a few flags (_has_kernel, _has_bias, _relu_act) for convenience.)"""
        self._has_kernel = False
        self._has_bias = False
        if hasattr(self, 'kernel') and self.kernel is not None:
            self._has_kernel = True
        if hasattr(self, 'bias') and self.bias is not None:
            self._has_bias = True

        self.init_quantizers(input_shape)
        if not hasattr(self, 'channel_loc'):
            self.channel_loc = -1
        if self.paq_config['rnd_strategy'] == 'auto':
            self.paq.rnd_strategy = 0 if self.can_bias_cover_rnd else 3

        self.bops = tf.Variable(0, dtype=tf.float32, trainable=False, name='bops')

    def init_quantizers(self, input_shape):
        """Initializes the High Granularity Quantizers for the kernel and the pre-activation values. This method is called by post_build() method."""

        if self._has_kernel:
            kq = HGQ.from_config(self.kq_config)
            bw_shape, degeneracy = kq._compute_bw_shape_and_degeneracy(self.kernel.shape)
            fbw = self.add_weight(
                name='kernel_bw',
                shape=bw_shape,
                initializer=tf.keras.initializers.Constant(kq.init_bw),  # type: ignore
                trainable=kq.trainable and self.trainable,
                regularizer=kq.regularizer
            )
            kq.build(fbw)
            kq.degeneracy = degeneracy
            if self._do_adapt_kernel_bits and not self._delayed_kernel_bits_adaption:
                kq.adapt_bw_bits(self.kernel)
                self._do_adapt_kernel_bits = False
            self.kq = kq

        aq = HGQ.from_config(self.paq_config)
        output_shape = self.compute_output_shape(input_shape)
        bw_shape, degeneracy = aq._compute_bw_shape_and_degeneracy(output_shape)
        fbw = self.add_weight(
            name='activation_bw',
            shape=bw_shape,
            initializer=tf.keras.initializers.Constant(aq.init_bw),  # type: ignore
            trainable=aq.trainable and self.trainable,
            regularizer=aq.regularizer
        )

        aq.build(fbw)
        aq.degeneracy = degeneracy
        self.paq = aq

    def call(self, x, training=None, record_minmax=None):
        if not self.built:
            self.build(tuple(x.shape))

        if record_minmax is None:
            record_minmax = training or self.record_minmax

        dtype = self.dtype or tf.keras.backend.floatx()
        x = tf.cast(x, dtype)

        return self.forward(x, training=training, record_minmax=record_minmax)

    def forward(self, x, training=None, record_minmax=None):
        raise NotImplementedError

    @property
    @tf.function(jit_compile=True)
    def kernel_bw(self):
        """Returns (over) approximated bitwidth of the kernel. Differentiable."""
        int_bits, fp_bits, kn = self.kq.get_bits(self.fused_kernel)  # type: ignore
        k_bw = tf.nn.relu(int_bits + fp_bits)  # negative sign not considered for kernel
        k_bw = scale_grad(k_bw, tf.sqrt(1. / self.kq.degeneracy))  # type: ignore
        return tf.broadcast_to(k_bw, self.kernel.shape)

    @tf.function(jit_compile=True)
    def _kernel_bw(self, qk):
        """Returns (over) approximated bitwidth of the kernel. Differentiable. Takes the differentiable quantized kernel as input to avoid recomputing."""
        int_bits, fp_bits, kn = self.kq.get_bits(qk, quantized=True)  # type: ignore
        k_bw = tf.nn.relu(int_bits + fp_bits)  # negative sign not considered for kernel
        k_bw = scale_grad(k_bw, tf.sqrt(1. / self.kq.degeneracy))  # type: ignore
        return tf.broadcast_to(k_bw, qk.shape)

    @property
    def kernel_bw_exact(self):
        """Returns exact bitwidth of the kernel. Non-differentiable. For post-training use."""
        kn, int_bits, fb = self.kq.get_bits_exact(self.fused_kernel)
        return int_bits + fb  # sign not considered for kernel

    @property
    def act_bw(self):
        """Returns the bitwidth of the pre-activation values. Differentiable."""
        int_bits, fp_bits, kn = self.paq.get_bits(pos_only=self._relu_act)  # type: ignore
        bw = int_bits + fp_bits
        if not self._relu_act:
            bw = bw + kn
        bw = tf.nn.relu(bw)
        bw = scale_grad(bw, tf.sqrt(1. / self.paq.degeneracy))  # type: ignore
        return tf.broadcast_to(bw, (1,) + self.output_shape[1:])

    @property
    def act_bw_exact(self) -> np.ndarray:
        """Returns the exact bitwidth of the pre-activation values. Non-differentiable. For post-training use."""
        kn, int_bits, fb = self.paq.get_bits_exact(pos_only=self._relu_act)
        bw = int_bits + fb + kn
        return np.broadcast_to(bw, (1,) + self.output_shape[1:])

    @property
    def fused_bias(self):
        return self.bias

    @property
    def fused_kernel(self):
        return self.kernel

    @property
    @tf.function(jit_compile=True)
    def fused_qkernel(self):
        """Returns the final, quantized kernel for deployment. non-differentiable, should not be used for training."""
        return self.kq(self.fused_kernel, training=False)  # type: ignore

    @property
    @tf.function(jit_compile=True)
    def fused_qbias(self):
        """Returns the final, quantized bias for deployment. non-differentiable, should not be used for training. When using rounding to nearest and the bias can cover the rounding error, bias is pre-biased to cover the rounding shift 2^-fbw, and then TRN can be used instead RND without any loss of accuracy."""
        bias = self.paq.bias_forward(self.fused_bias, False, self.channel_loc)  # type: ignore

        fbw = self.paq.fbw
        if self.channel_loc == -1:
            dims = list(range(len(fbw.shape) - 1))
        elif self.channel_loc == 1:
            dims = [0] + list(range(2, len(fbw.shape)))
        else:
            raise ValueError('channel_loc must be -1 or 1')

        fbw = tf.reduce_max(self.paq.fbw, axis=dims, keepdims=False)
        fbw = tf.broadcast_to(fbw, bias.shape)
        mask = tf.reduce_max(self.act_bw, axis=dims, keepdims=False) > 0

        if self.paq.rnd_strategy != 3 and self.can_bias_cover_rnd:
            bias = tf.pow(2., -tf.round(fbw) - 1) + bias  # type: ignore

        return tf.where(mask, bias, 0.)

    def reset_minmax(self):
        """Resets the recorded minmax values for the pre-activation quantizer."""
        self.paq.minmax_reg_reset()  # type: ignore

    @property
    def compute_exact_bops(self):
        """Computes the exact bops for the layer. Non-differentiable. For post-training use."""
        self.bops.assign(tf.constant(0.))
        return np.float32(0.)

    def get_config(self):
        base_config = super().get_config()
        config = dict(
            kq_conf=self.kq_config,
            paq_conf=self.paq_config,
            beta=float(self.beta.numpy())  # type: ignore
        )
        return {**base_config, **config}

    def get_keras_config(self):
        config = super().get_config()
        return config
