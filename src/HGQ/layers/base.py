import numpy as np
import tensorflow as tf

from ..quantizer import HGQ
from ..utils import get_default_kernel_quantizer_config, get_default_pre_activation_quantizer_config
from ..utils import apf_to_tuple, tuple_to_apf


@staticmethod  # type: ignore
@tf.function(jit_compile=True)
def scale_grad(x, scale):
    sx = x * scale
    return sx + tf.stop_gradient(x - sx)


class HLayerBase(tf.keras.layers.Layer):
    """Abstract base class for all layers in the library. Child classes should call post_build() after calling their build() method.
    """

    def __init__(self, kernel_quantizer_config=None, pre_activation_quantizer_config=None, bops_reg_factor=0., **kwargs):
        super().__init__(**kwargs)
        self._has_kernel = None
        self._has_bias = None
        self.kernel_quantizer_config = kernel_quantizer_config or get_default_kernel_quantizer_config()
        self.pre_activation_quantizer_config = pre_activation_quantizer_config or get_default_pre_activation_quantizer_config()
        self.bops_reg_factor = tf.Variable(bops_reg_factor, dtype=tf.float32, trainable=False, name='bops_reg_factor')
        self.record_minmax = False
        self._has_last_layer = False

    def build(self, input_shape):
        super().build(input_shape)
        self.post_build(input_shape)

    def post_build(self, input_shape):
        """This method should be called after calling build() method of the child class. It initializes the quantizers and sets the bops variable, and set a few flags (_has_kernel, _has_bias, _relu_act) for convenience.)"""
        self._has_kernel = False
        self._has_bias = False
        if hasattr(self, 'kernel') and self.kernel is not None:
            self._has_kernel = True
        if hasattr(self, 'bias') and self.bias is not None:
            self._has_bias = True
        self._relu_act = hasattr(self, 'activation') and self.activation is tf.keras.activations.relu

        if self.pre_activation_quantizer_config['rnd_strategy'] == 'auto':
            strategy = 'floor' if not self._has_bias else 'standard_round'
            self.pre_activation_quantizer_config['rnd_strategy'] = strategy
        self.init_quantizers(input_shape)

        self.bops = tf.Variable(0, dtype=tf.float32, trainable=False, name='bops')
        self.channel_loc = -1

    def init_quantizers(self, input_shape):
        """Initializes the High Granularity Quantizers for the kernel and the pre-activation values. This method is called by post_build() method."""

        if self._has_kernel:
            kq = HGQ.from_config(self.kernel_quantizer_config)
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
            kq.adapt_bw_bits(self.kernel)
            self.kernel_quantizer = kq

        aq = HGQ.from_config(self.pre_activation_quantizer_config)
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
        self.pre_activation_quantizer = aq

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
        int_bits, fp_bits, kn = self.kernel_quantizer.get_bits(self.kernel)  # type: ignore
        k_bw = tf.nn.relu(int_bits + fp_bits)  # negative sign not considered for kernel
        k_bw = scale_grad(k_bw, tf.sqrt(1. / self.kernel_quantizer.degeneracy))  # type: ignore
        return tf.broadcast_to(k_bw, self.kernel.shape)

    @tf.function(jit_compile=True)
    def _kernel_bw(self, qk):
        """Returns (over) approximated bitwidth of the kernel. Differentiable. Takes the differentiable quantized kernel as input to avoid recomputing."""
        int_bits, fp_bits, kn = self.kernel_quantizer.get_bits(qk, quantized=True)  # type: ignore
        k_bw = tf.nn.relu(int_bits + fp_bits)  # negative sign not considered for kernel
        k_bw = scale_grad(k_bw, tf.sqrt(1. / self.kernel_quantizer.degeneracy))  # type: ignore
        return tf.broadcast_to(k_bw, qk.shape)

    @property
    def kernel_bw_exact(self):
        """Returns exact bitwidth of the kernel. Non-differentiable. For post-training use."""
        int_bits, fb, kn = self.kernel_quantizer.get_bits_exact(self.kernel)
        return int_bits + fb  # sign not considered for kernel

    @property
    # @tf.function(jit_compile=True)
    def act_bw(self):
        """Returns the bitwidth of the pre-activation values. Differentiable."""
        int_bits, fp_bits, kn = self.pre_activation_quantizer.get_bits(pos_only=self._relu_act)  # type: ignore
        bw = int_bits + fp_bits
        if not self._relu_act:
            bw = bw + kn
        bw = tf.nn.relu(bw)
        bw = scale_grad(bw, tf.sqrt(1. / self.pre_activation_quantizer.degeneracy))  # type: ignore
        return tf.broadcast_to(bw, (1,) + self.output_shape[1:])

    @property
    def input_bw(self):
        try:
            return self.last_layer.act_bw
        except ValueError:
            return None

    @property
    @tf.function(jit_compile=True)
    def qkernel(self):
        """Returns the quantized kernel. non-differentiable."""
        return self.kernel_quantizer(self.kernel, training=False)  # type: ignore

    @property
    @tf.function(jit_compile=True)
    def qbias(self):
        if not self._has_bias:
            return None
        """Returns the quantized bias. non-differentiable."""
        return self.pre_activation_quantizer.bias_forward(self.bias, False, self.channel_loc)  # type: ignore

    @property
    @tf.function(jit_compile=True)
    def fused_qbias(self):
        """Returns the adjusted quantized bias to bypass explicit rounding."""
        qbias = self.qbias
        if qbias is None:
            return None

        fbw = self.pre_activation_quantizer.fbw
        if self.channel_loc == -1:
            dims = list(range(len(fbw.shape) - 1))
        elif self.channel_loc == 1:
            dims = [0] + list(range(2, len(fbw.shape)))
        else:
            raise ValueError('channel_loc must be -1 or 1')

        fbw = tf.reduce_max(self.pre_activation_quantizer.fbw, axis=dims, keepdims=False)
        fbw = tf.broadcast_to(fbw, qbias.shape)
        mask = tf.reduce_max(self.act_bw, axis=dims, keepdims=False) > 0

        if self.pre_activation_quantizer.rnd_strategy != 3:
            qbias = tf.pow(2., -tf.floor(fbw + 0.5) - 1) + qbias  # type: ignore

        return tf.where(mask, qbias, 0.)

    def reset_minmax(self):
        """Resets the recorded minmax values for the pre-activation quantizer."""
        self.pre_activation_quantizer.minmax_reg_reset()  # type: ignore

    @property
    def compute_exact_bops(self):
        """Computes the exact bops for the layer. Non-differentiable. For post-training use."""
        self.bops.assign(tf.constant(0.))
        return np.float32(0.)

    @property
    def result_container(self) -> str:
        """Returns the ap representation of the quantizers as a string."""
        int_bits, fp_bits, kn = self.pre_activation_quantizer.get_bits(pos_only=False)  # type: ignore
        int_bits, fp_bits, kn = int_bits.numpy().astype(np.int8), fp_bits.numpy().astype(np.int8), kn.numpy().astype(np.int8)
        int_bits, fp_bits, kn = int_bits.ravel(), fp_bits.ravel(), kn.ravel()
        mask = self.act_bw.numpy() > 0  # type: ignore
        if skip_dims := self.pre_activation_quantizer.skip_dims:
            mask = np.any(mask, axis=tuple(skip_dims)).ravel()
        if int_bits[mask].size > 1:
            int_max, fp_max, kn_max = int_bits[mask].max(), fp_bits[mask].max(), kn[mask].max()
        else:
            int_max, fp_max, kn_max = int_bits[mask].item(), fp_bits[mask].item(), kn[mask].item()
        if self.pre_activation_quantizer.rnd_strategy != 3 and not self._has_bias:
            fp_max += 1
        assert np.sum(kn[int_bits + fp_bits <= 0]
                      ) == 0, f'Bit counting error at {self.name}. This should never happen. Please try again with cuda disabled (2^13 or above will may in error when tensorflow is run with cuda).'
        return tuple_to_apf((kn_max, int_max, fp_max))

    @property
    def last_layer(self):
        if not self._has_last_layer:
            if len(self._inbound_nodes) != 1:
                raise ValueError(f'input_container is only available for layers with a single input. {self.name} has {len(self._inbound_nodes)} inputs.')
            self._has_last_layer = True
        return self._inbound_nodes[0].inbound_layers

    @property
    def input_container(self):
        try:
            return self.last_layer.result_container
        except ValueError:
            return None

    @property
    def max_accum_fp_bits(self):
        ker_fp = apf_to_tuple(self.ker_container)[2]
        input_container = self.input_container
        assert input_container is not None, f'input_container is not available for {self.name}.'
        inp_fp = apf_to_tuple(input_container)[2]
        fp = inp_fp + ker_fp
        if self.pre_activation_quantizer.rnd_strategy != 3:
            fp = max(fp, 1)
        return fp

    @property
    def act_container(self) -> str:
        """Returns the minimal ap representation of the pre-activation quantizer that can represnet all values it have seen."""
        if not self._relu_act:
            return self.result_container
        int_bits, fp_bits, kn = self.pre_activation_quantizer.get_bits(pos_only=True)  # type: ignore
        int_bits, fp_bits, kn = int_bits.numpy().astype(np.int8), fp_bits.numpy().astype(np.int8), kn.numpy().astype(np.int8)
        mask = int_bits + fp_bits > 0
        int_max, fp_max, kn_max = int_bits[mask].max(), fp_bits[mask].max(), kn[mask].max()
        assert np.sum(
            kn[~mask]) == 0, f'Bit counting error at {self.name}. This should never happen. Please try again with cuda disabled (2^13 or above will may in error when tensorflow is run with cuda).'
        return tuple_to_apf((kn_max, int_max, fp_max))

    @property
    def ker_container(self):
        """Returns the minimal ap representation of the kernel quantizer that can represent every value."""
        int_bits, fp_bits, kn = self.kernel_quantizer.get_bits_exact(self.kernel)
        mask = int_bits + fp_bits > 0
        assert np.sum(
            kn[~mask]) == 0, f'Bit counting error at {self.name}. This should never happen. Please try again with cuda disabled (2^13 or above will may in error when tensorflow is run with cuda).'
        int_max, fp_max, kn_max = int_bits[mask].max(), fp_bits[mask].max(), kn[mask].max()
        return tuple_to_apf((kn_max, int_max, fp_max))

    def get_full_config(self):
        base_config = self.get_config()
        config = dict(
            kernel_quantizer_config=self.kernel_quantizer_config,
            pre_activation_quantizer_config=self.pre_activation_quantizer_config,
            bops_reg_factor=float(self.bops_reg_factor.numpy())
        )
        return {**base_config, **config}

    def get_config(self):
        return super().get_config()
