import numpy as np
import tensorflow as tf
from functools import singledispatchmethod

from .utils import strategy_dict


two = tf.constant(2, dtype=tf.float32)
log2 = tf.constant(np.log(2), dtype=tf.float32)


@tf.function(jit_compile=True)
def q_round(x: tf.Tensor, strategy: int = 0):
    """Round the tensor.

    strategy:
        0: standard round (default, round to nearest, 0.5 to even)
        1: stochastic round
        2: fast uniform noise injection (uniform noise in [-0.5, 0.5])
        3: floor

    """
    if strategy == 0:  # standard round
        return tf.floor(x + 0.5)  # type: ignore
    if strategy == 1:  # stochastic round
        _floor = tf.floor(x)
        noise = tf.random.uniform(tf.shape(x))
        return tf.where(noise < x - _floor, _floor + 1, _floor)
    if strategy == 2:  # fast uniform noise injection
        noise = tf.random.uniform(tf.shape(x), minval=-0.5, maxval=0.5)  # type: ignore
        noise = tf.where(tf.abs(x) <= 0.5, -x, noise)  # type: ignore
        return tf.stop_gradient(noise) + x  # type: ignore
    if strategy == 3:
        return tf.floor(x)
    raise ValueError(f"Unknown strategy {strategy}")


class HGQ:
    def __init__(self, init_bw: float, skip_dims, rnd_strategy: str | int = 'floor', exact_q_value=True, dtype=None, bw_clip=(-23, 23), trainable=True, regularizer=None, minmax_record=False):
        self.init_bw = init_bw
        self.skip_dims = skip_dims
        self.rnd_strategy = strategy_dict[rnd_strategy] if isinstance(rnd_strategy, str) else rnd_strategy
        self.exact_q_value = exact_q_value
        self.dtype = dtype
        self.bw_clip = bw_clip
        self.trainable = trainable
        self.regularizer = regularizer
        self.minmax_record = minmax_record
        self.built = False
        self.degeneracy = 1

    def _compute_bw_shape_and_degeneracy(self, input_shape):
        if isinstance(self.skip_dims, str):
            if self.skip_dims == 'all':
                self.skip_dims = tuple(range(len(input_shape)))
            elif self.skip_dims == 'batch':
                self.skip_dims = (0,)
            elif self.skip_dims == 'none':
                self.skip_dims = None
            elif self.skip_dims == 'except_last':
                self.skip_dims = tuple(range(len(input_shape) - 1))
            elif self.skip_dims == 'except_1st':
                self.skip_dims = (0,) + tuple(range(2, len(input_shape)))
            else:
                raise ValueError('skip_dims must be tuple or str in ["all", "except_last", "batch", "except_last", "none"]')
        _input_shape = list(input_shape)
        degeneracy = 1
        if self.skip_dims:
            for d in self.skip_dims:
                degeneracy *= _input_shape[d] if _input_shape[d] is not None else 1
                _input_shape[d] = 1
        return _input_shape, degeneracy

    def init_minmax(self):
        if self.minmax_record:
            dtype = self.dtype or tf.keras.backend.floatx()
            self._min = tf.Variable(tf.zeros_like(self.fbw), trainable=False, name='min', dtype=dtype)
            self._max = tf.Variable(tf.zeros_like(self.fbw), trainable=False, name='max', dtype=dtype)
            self.minmax_reg_reset()  # type: ignore

    @singledispatchmethod
    def build(self, x, name=None):
        self.built = True
        self.fbw = x
        self.init_minmax()

    @build.register
    def _(self, input_shape: tuple, name: str | None = None):
        self.built = True
        _input_shape, degeneracy = self._compute_bw_shape_and_degeneracy(input_shape)
        self.degeneracy = degeneracy

        self.fbw = tf.Variable(tf.ones(_input_shape) * self.init_bw, trainable=self.trainable, name=name, dtype=tf.float32)

        self.init_minmax()

    @tf.function(jit_compile=True)
    def minmax_reg_reset(self):
        assert self.built
        inf = tf.ones(self.fbw.shape) * float('inf')
        self._min.assign(inf)
        self._max.assign(-inf)

    @tf.function(jit_compile=True)
    def __call__(self, x, training=None, record_minmax=None):
        if not self.built:
            self.build(tuple(x.shape), name=None)
        if self.bw_clip:
            self.fbw.assign(tf.clip_by_value(self.fbw, *self.bw_clip))
        return self.forward(x, training, record_minmax)  # type: ignore

    @tf.function(jit_compile=True)
    def forward(self, x, training=None, record_minmax=None):
        if self.exact_q_value or not training:
            scale = tf.pow(two, tf.round(self.fbw))
        else:
            scale = tf.pow(two, self.fbw)

        xq = q_round(x * scale, self.rnd_strategy) / scale  # type: ignore
        delta = tf.stop_gradient(xq - x)
        if training:
            prod = delta * self.fbw * log2  # type: ignore
            delta = tf.stop_gradient(delta + prod) - prod

        if not record_minmax:
            return x + delta
        xq = x + delta
        if self.skip_dims:
            min_xq = tf.reduce_min(xq, axis=self.skip_dims)
            max_xq = tf.reduce_max(xq, axis=self.skip_dims)
        else:
            min_xq = max_xq = xq

        self._min.assign(tf.minimum(min_xq, self._min))
        self._max.assign(tf.maximum(max_xq, self._max))

        return xq

    @tf.function(jit_compile=True)
    def bias_forward(self, x, training=None, channel_loc=-1):

        if channel_loc == -1:
            dims = list(range(len(self.fbw.shape) - 1))
        elif channel_loc == 1:
            dims = [0] + list(range(2, len(self.fbw.shape)))
        else:
            raise ValueError('channel_loc must be -1 or 1')

        fbw = tf.reduce_max(self.fbw, axis=dims, keepdims=False)

        if self.exact_q_value or not training:
            scale = tf.pow(two, tf.round(fbw))
        else:
            scale = tf.pow(two, fbw)

        xq = q_round(x * scale, self.rnd_strategy) / scale  # type: ignore
        delta = tf.stop_gradient(xq - x)

        if training:
            prod = delta * fbw * log2  # type: ignore
            delta = tf.stop_gradient(delta + prod) - prod

        return x + delta

    @tf.function(jit_compile=True)
    def get_bits(self, ref=None, quantized=None, pos_only=False):
        fp_bits = tf.round(self.fbw)
        fp_bits = self.fbw + tf.stop_gradient(fp_bits - self.fbw)  # type: ignore
        if ref is not None:
            if quantized is not None:
                _ref = ref
            else:
                _ref = self.forward(ref)  # type: ignore
            kn = tf.keras.backend.cast_to_floatx(_ref < 0)
            _ref = tf.abs(_ref)
            int_bits = tf.floor(tf.math.log(_ref) / log2) + 1
            if self.skip_dims:
                int_bits = tf.reduce_max(int_bits, axis=self.skip_dims)
                kn = tf.reduce_max(kn, axis=self.skip_dims)
        else:
            assert self.minmax_record
            if pos_only:
                _ref = tf.maximum(self._max, 0.)
                kn = tf.zeros_like(self._max)
            else:
                _ref = tf.maximum(tf.abs(self._min), tf.abs(self._max))
                kn = tf.keras.backend.cast_to_floatx(self._min < 0)
            int_bits = tf.floor(tf.math.log(_ref) / log2) + 1
        return int_bits, fp_bits, kn

    def get_bits_exact(self, ref=None):

        if ref is None and self.minmax_record:
            int_bits, fp_bits, kn = self.get_bits()  # type: ignore
            return int_bits.numpy().astype(np.int8), fp_bits.numpy().astype(np.int8), kn.numpy().astype(np.int8)

        assert ref is not None
        w = self.forward(ref).numpy()  # type: ignore
        kn = (w < 0).astype(np.int8)
        high, low = self._get_arr_bits(w)
        fb = int(max(*self.bw_clip)) + 2 - low  # type: ignore
        int_bits = high - low - fb
        return int_bits.astype(np.int8), fb.astype(np.int8), kn.astype(np.int8)

    def _get_arr_bits(self, k: np.ndarray):
        mul = int(max(*self.bw_clip)) + 2  # type: ignore
        k = k * 2**mul
        k = np.abs(k)[..., None]  # type: ignore
        n = int(np.ceil(np.max(np.log2(k + 1))))  # type: ignore
        divisor = 2**np.arange(1, n)[None, ...]  # type: ignore
        low_pos = np.sum(k % divisor == 0, axis=-1) + (k[..., 0] == 1)
        with np.errstate(divide='ignore'):
            high_pos = np.where(k[..., 0] != 0, np.floor(np.log2(k[..., 0]) + 1), low_pos).astype(np.int8)
        # print(high_pos, low_pos)
        return high_pos.astype(np.int8), low_pos.astype(np.int8)

    def adapt_bw_bits(self, ref: tf.Tensor):
        if not self.built:
            self.build(tuple(ref.shape), name=None)
        new_fbw = self.fbw - (tf.math.log(tf.abs(ref)) / log2)
        if self.skip_dims:
            new_fbw = tf.reduce_min(new_fbw, axis=self.skip_dims, keepdims=True)
        self.fbw.assign(new_fbw)

    @classmethod
    def from_config(cls, config):
        return cls(**config)
