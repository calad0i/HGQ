from abc import ABCMeta
from collections import abc

import keras
import tensorflow as tf
from keras import constraints, initializers, regularizers
from keras.layers import Dense, Dropout, Layer, MultiHeadAttention, Permute, Reshape, Softmax
from keras.src.engine.functional import Functional
from keras.src.utils import tf_utils

from ..base import ABSBaseLayer, HLayerBase
from ..dense import HDense
from ..misc import HActivation, HAdd, HQuantize
from ..passive_layers import PDropout, PPermute, PReshape


class MatMul(Layer):
    def __init__(self, transpose_a=False, transpose_b=False, **kwargs):
        super().__init__(**kwargs)
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def call(self, x):
        return tf.matmul(x[0], x[1], transpose_a=self.transpose_a, transpose_b=self.transpose_b)

    def compute_output_shape(self, input_shape):
        left = input_shape[0][:-2]
        x1 = input_shape[0][-1] if self.transpose_a else input_shape[0][-2]
        x2 = input_shape[1][-2] if self.transpose_b else input_shape[1][-1]
        return [*left, x1, x2]

# @register_keras_serializable(package="HGQ")


class HMatMul(HLayerBase, MatMul):

    def call(self, x, training=None, record_minmax=None):
        if not self.built:
            self.build(tuple(x.shape))

        if record_minmax is None:
            record_minmax = training or self.record_minmax

        return self.forward(x, training=training, record_minmax=record_minmax)

    def forward(self, x, training=None, record_minmax=None):
        z = self.jit_forward(x, training=training, record_minmax=record_minmax)  # type: ignore
        if len(self._inbound_nodes) > 0:
            layer1, layer2 = self._inbound_nodes[0].inbound_layers
            inp_bw1 = layer1.act_bw
            inp_bw2 = layer2.act_bw
            bops = tf.reduce_sum(tf.matmul(inp_bw1, inp_bw2, transpose_a=self.transpose_a, transpose_b=self.transpose_b))
            self.bops.assign(bops)
            bops = tf.cast(bops, tf.float32) * self.beta  # type: ignore
            self.add_loss(bops)
        return z

    @tf.function(jit_compile=True)
    def jit_forward(self, x, training=None, record_minmax=None):

        z = tf.matmul(x[0], x[1], transpose_a=self.transpose_a, transpose_b=self.transpose_b)
        z = self.paq(z, training, record_minmax)  # type: ignore
        return z


class Scale(Layer):
    "Placeholder!!!!"

    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, x):
        return self.scale * x

    def compute_output_shape(self, input_shape):
        return input_shape


class HScale(HLayerBase):
    "Placeholder!!!!"

    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    @tf.function(jit_compile=True)
    def forward(self, x, training=None, record_minmax=None):  # type: ignore
        output = self.scale * x
        return self.paq(output, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape


class Div(Layer):
    "Placeholder!!!!"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x[0] / x[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class HDiv(HLayerBase):
    "Placeholder!!!!"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function(jit_compile=True)
    def forward(self, x, training=None, record_minmax=None):  # type: ignore
        output = x[0] / x[1]
        return self.paq(output, training=training, record_minmax=record_minmax)  # type: ignore

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class HMultiHeadAttention(Layer, metaclass=ABCMeta):
    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._built = False
        self.supports_masking = True
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._dropout = dropout
        self._use_bias = use_bias
        self._output_shape = output_shape
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        if attention_axes is not None and not isinstance(
            attention_axes, abc.Sized
        ):
            self._attention_axes = (attention_axes,)
        else:
            self._attention_axes = attention_axes
        self._built_from_signature = False
        self._query_shape, self._key_shape, self._value_shape = None, None, None

    def attn_model_build(self, q_dim, k_dim, v_dim):
        with tf_utils.maybe_init_scope(self):  # type: ignore
            to_q = HDense(self._key_dim * self._num_heads, use_bias=self._use_bias, name=f"{self.name}_to_q")
            to_k = HDense(self._key_dim * self._num_heads, use_bias=self._use_bias, name=f"{self.name}_to_k")
            to_v = HDense(self._value_dim * self._num_heads, use_bias=self._use_bias, name=f"{self.name}_to_v")
            to_out = HDense(v_dim, use_bias=self._use_bias, name=f"{self.name}_to_out")
            dropout = PDropout(rate=self._dropout)
            norm = HActivation('softmax')
        self._query_shape = q_dim
        self._key_shape = k_dim
        self._value_shape = v_dim
        self._built_from_signature = True

        q = keras.Input(shape=(50, q_dim))
        k = keras.Input(shape=(50, k_dim))
        v = keras.Input(shape=(50, v_dim))
        mask = keras.Input(shape=(50, 50))
        _mask = tf.repeat(mask[:, None], self._num_heads, axis=1)
        _q = HQuantize()(q)
        _k = HQuantize()(k)
        _v = HQuantize()(v)

        # [B, N, HD]
        Q = to_q(_q)
        K = to_k(_k)
        V = to_v(_v)

        # [B, N, H, D]
        Q = PReshape([-1, self._num_heads, self._key_dim])(Q)
        K = PReshape([-1, self._num_heads, self._key_dim])(K)
        V = PReshape([-1, self._num_heads, self._value_dim])(V)

        # [B, H, N, D]
        Q = PPermute([2, 1, 3])(Q)
        K = PPermute([2, 1, 3])(K)
        V = PPermute([2, 1, 3])(V)

        # [B, H, NQ, NK]
        QK = HMatMul(transpose_b=True)([Q, K])
        QK = HScale(float(1 / tf.math.sqrt(tf.cast(K.shape[-1], tf.float32)).numpy()))(QK)  # Broken
        QK = HAdd()([QK, _mask])
        QK = norm(QK)
        QK = dropout(QK)
        out = HMatMul()([QK, V])
        out = PPermute([2, 1, 3])(out)
        out = PReshape([-1, self._value_dim * self._num_heads])(out)
        out = to_out(out)

        model = keras.Model([q, k, v, mask], [out, QK])
        self.model = model
        return model

    def __call__(self, inputs):
        if not self._built:
            self.build(inputs)
        q, k, v, mask = inputs
        q_dim = q.shape[-1]
        k_dim = k.shape[-1]
        v_dim = v.shape[-1]
        if not self._built_from_signature:
            self.attn_model_build(q_dim, k_dim, v_dim)
        return self.model([q, k, v, mask])
