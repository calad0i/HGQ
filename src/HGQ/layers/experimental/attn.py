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


class MatMul(Layer):
    def __init__(self, transpose_a=False, transpose_b=False, **kwargs):
        super().__init__(**kwargs)
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def call(self, x):
        return tf.matmul(x[0], x[1], transpose_a=self.transpose_a, transpose_b=self.transpose_b)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + input_shape[1][-1:]

# @register_keras_serializable(package="HGQ")


class HMatMul(HLayerBase, MatMul):
    def forward(self, x, training=None, record_minmax=None):
        z = self.jit_forward(x, training=training, record_minmax=record_minmax)  # type: ignore
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


class _MultiHeadAttention(Layer, metaclass=ABCMeta):
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
            to_q = Dense(self._key_dim * self._num_heads, use_bias=self._use_bias, name=f"{self.name}_to_q")
            to_k = Dense(self._key_dim * self._num_heads, use_bias=self._use_bias, name=f"{self.name}_to_k")
            to_v = Dense(self._value_dim * self._num_heads, use_bias=self._use_bias, name=f"{self.name}_to_v")
            to_out = Dense(v_dim, use_bias=self._use_bias, name=f"{self.name}_to_out")
            dropout = Dropout(rate=self._dropout)
            norm = Softmax(axis=-1)
        self._query_shape = q_dim
        self._key_shape = k_dim
        self._value_shape = v_dim
        self._built_from_signature = True

        q = keras.Input(shape=(None, q_dim))
        k = keras.Input(shape=(None, k_dim))
        v = keras.Input(shape=(None, v_dim))
        mask = keras.Input(shape=(None, None, None))

        # [B, N, HD]
        Q = to_q(q)
        K = to_k(k)
        V = to_v(v)

        # [B, N, H, D]
        Q = Reshape([-1, self._num_heads, self._key_dim])(Q)
        K = Reshape([-1, self._num_heads, self._key_dim])(K)
        V = Reshape([-1, self._num_heads, self._value_dim])(V)

        # [B, H, N, D]
        Q = Permute([2, 1, 3])(Q)
        K = Permute([2, 1, 3])(K)
        V = Permute([2, 1, 3])(V)

        # [B, H, NQ, NK]
        QK = MatMul(transpose_b=True)([Q, K])
        QK = Scale(float(1 / tf.math.sqrt(tf.cast(K.shape[-1], tf.float32)).numpy()))(QK)  # Broken
        QK = keras.layers.Add()([QK, mask])
        QK = norm(QK)
        QK = dropout(QK)
        out = MatMul()([QK, V])
        out = Permute([2, 1, 3])(out)
        out = Reshape([-1, self._value_dim * self._num_heads])(out)
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
