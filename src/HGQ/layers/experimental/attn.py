from collections import abc

import tensorflow as tf
from keras import constraints, initializers, regularizers
from keras.layers import Dense, Dropout, Layer, MultiHeadAttention, Softmax
from keras.src.utils import tf_utils

from ..base import ABSBaseLayer, HLayerBase
from ..dense import HDense


class HMultiHeadAttention(Layer):
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
        **kwargs,
    ):
        super().__init__(**kwargs)
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

    def attn_build(self, q_dim, k_dim, v_dim):
        with tf_utils.maybe_init_scope(self):  # type: ignore
            self.to_q = Dense(self._key_dim * self._num_heads, use_bias=self._use_bias, name=f"{self.name}_to_q")
            self.to_k = Dense(self._key_dim * self._num_heads, use_bias=self._use_bias, name=f"{self.name}_to_k")
            self.to_v = Dense(self._value_dim * self._num_heads, use_bias=self._use_bias, name=f"{self.name}_to_v")
            self.to_out = Dense(v_dim, use_bias=self._use_bias, name=f"{self.name}_to_out")
            self.dropout = Dropout(rate=self._dropout)
            self.norm = Softmax(axis=-1)
        self._query_shape = q_dim
        self._key_shape = k_dim
        self._value_shape = v_dim
        self._built_from_signature = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((None, None, self._value_dim * self._num_heads))

    def call(self, q, k, v, attention_mask=None, training=None, return_attention_scores=False):
        if not self._built_from_signature:
            self.attn_build(q.shape[-1], k.shape[-1], v.shape[-1])

        # [B, N, HD]
        Q = self.to_q(q)
        K = self.to_k(k)
        V = self.to_v(v)

        # [B, N, H, D]
        Q = tf.reshape(Q, [*Q.shape[:2], self._num_heads, self._key_dim])
        K = tf.reshape(K, [*K.shape[:2], self._num_heads, self._key_dim])
        V = tf.reshape(V, [*V.shape[:2], self._num_heads, self._value_dim])

        # [B, H, N, D]
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])
        print(Q.shape, K.shape, V.shape)

        # [B, H, NQ, NK]
        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK / tf.math.sqrt(tf.cast(K.shape[-1], tf.float32))
        print(QK.shape)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None]
            if attention_mask.dtype == tf.bool:
                attention_mask = tf.cast(~attention_mask, QK.dtype) * -1e9  # type: ignore
            QK = QK + attention_mask

        QK = self.norm(QK)

        # [B, H, NQ, DV]
        out = tf.matmul(QK, V)
        print(out.shape)

        # [B, N, H, D]
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [*out.shape[:2], self._value_dim * self._num_heads])
        out = self.to_out(out)
        if return_attention_scores:
            return out, QK
        return out
