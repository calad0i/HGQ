from collections.abc import Callable
from functools import singledispatch

import numpy as np
import tensorflow as tf
from keras.layers import AvgPool1D, AvgPool2D, AvgPool3D, Concatenate, Flatten, MaxPool1D, MaxPool2D, MaxPool3D, Reshape
from keras.src.layers.convolutional.base_conv import Conv
from keras.src.layers.pooling.base_pooling1d import Pooling1D
from keras.src.layers.pooling.base_pooling2d import Pooling2D
from keras.src.layers.pooling.base_pooling3d import Pooling3D
from tensorflow import keras

from ..utils import apf_to_tuple, tuple_to_apf, warn
from .fixed_point_quantizer import FixedPointQuantizer

STREAM = False
"This variable is not used for now."


def get_arr_container(arr: np.ndarray, silent=False):
    """ Get the minimal fixed integer that can represent the array (kif format). If the result is greater than ~30, consider that as inf. (Not representable by fixed point with reasonable bitwidth.)"""
    k = arr < 0
    lf, hf = -32, 32
    if np.all(arr == 0):
        warn('All zero array is detected.')
        return 0, 1, 0  # All zero array is special. Though, (u)fixed<0,...> will lead to a crash, thus ufixed<1,...> is used.
    while True:
        if hf - lf <= 1:
            break
        f = (lf + hf) // 2
        _arr = arr * 2**f
        if np.all(_arr.astype(np.int64) == _arr):
            hf = f
        else:
            lf = f
    f = int(hf)
    if hf == 32 and not silent:
        warn('Failed to derive fractional bits. Is the array really quantized with less than 32 fp bits?')
    with np.errstate(divide='ignore'):
        i1, i2 = -np.inf, -np.inf
        if (~k).any():
            i1 = np.floor(1 + np.log2(arr[~k] * 2.**f)).max()
        if k.any():
            i2 = np.ceil(np.log2(-arr[k] * 2.**f)).max()
        i = int(max(i1, i2))
    i -= f
    k = int(k.any())
    return k, i, f


def activation_kif_forward(func: Callable, k: int, i: int, f: int):
    """Given the input bitwidth (kif) of an activation function, get the output bitwidth (kif)."""
    assert k + i + f > 0, 'Activation function is applied to an zero array. Something is wrong.'
    N = min(2**(k + i + f), 2**20)
    arr = np.array(np.linspace(-2.**i * k, 2.**i - 2.**-f, N), dtype=np.float64)
    arr: np.ndarray = np.array(func(arr), dtype=np.float32)
    K, I, F = get_arr_container(arr, silent=True)
    return K, I, F


def get_input_kifs(layer: keras.layers.Layer) -> tuple[tuple[int, int, int] | np.ndarray, ...]:
    """Get the input bitwidth of a layer, as a tuple of (k, i, f)."""
    parents: keras.layers.Layer | list[keras.layers.Layer] = layer._inbound_nodes[0].inbound_layers

    # As any layer that *changes* bitwidth will be followed by a quantizer, assume all layers we will meet here are "passive" layers.
    if isinstance(parents, keras.layers.Layer):
        return (get_produced_kif(parents),)
    else:
        # Passive layers with multiple inputs. Likely only Concatenate layer.
        return tuple(get_produced_kif(parent) for parent in parents)


# ===================================================================================================
# Given a layer, get the produced bitwidth of the layer, as a tuple of (k, i, f).
# ===================================================================================================

@singledispatch
def get_produced_kif(layer) -> tuple[int, int, int]:
    """Get the produced bitwidth of a layer, as a tuple of (k, i, f)."""
    warn(f'Layer {layer.name} is unknown. Assuming infinite produced bitwidth.')
    return 1, 65535, 65535


@get_produced_kif.register
def _(layer: FixedPointQuantizer):
    kifs = get_input_kifs(layer)
    assert len(kifs) == 1, f'Quantizer {layer.name} has more than one input. This is not supported.'
    _ik, _ii, _if = kifs[0]

    # If the quantizer bit width is unnecessarily large, reduce it.
    ok, oi, of = layer.result_t_kif
    k, i, f = min(_ik, ok), min(_ii, oi), min(_if, of)
    if _ii <= oi and _if > f:
        # May overflow otherwise, like (8,4) & (8,5) should result in (9,4) instead of (8,4) to avoid overflow.
        i += 1
    if k != ok or i != oi or f != of:
        layer.keep_negative.assign(tf.minimum(layer.keep_negative, k))
        layer.integers.assign(tf.minimum(layer.integers, k + i))
        layer.bits.assign(tf.minimum(layer.bits, k + i + f))

    return k, i, f


@get_produced_kif.register
def _(layer: keras.layers.Activation | keras.layers.ReLU | keras.layers.LeakyReLU | keras.layers.Softmax):
    kifs = get_input_kifs(layer)
    assert len(kifs) == 1, f'Activation layer {layer.name} has more than one input. This is not supported.'
    if isinstance(layer, keras.layers.Activation):
        if layer.activation is tf.keras.activations.softmax:
            return 0, 1, 65535
    if isinstance(layer, keras.layers.Softmax):
        return 0, 1, 65535
    k, i, f = activation_kif_forward(layer, *np.max(kifs, axis=0))
    return k, i, f


@get_produced_kif.register
def _(layer: AvgPool1D | AvgPool2D | AvgPool3D):
    kifs = get_input_kifs(layer)
    assert len(kifs) == 1, f'Pooling layer {layer.name} has more than one input. This is not supported.'
    k, i, f = kifs[0]
    pool_size = np.prod(layer.pool_size)
    f += int(np.ceil(np.log2(pool_size)))
    return k, i, f


@get_produced_kif.register
def _(layer: keras.layers.Add):
    kifs = get_input_kifs(layer)
    k, i, f = np.max(kifs, axis=0)
    # being lazy here. But this will never overflow.
    i += int(np.ceil(np.log2(len(kifs))))
    return k, i, f


@get_produced_kif.register
def _(layer: keras.layers.Concatenate):
    kifs = get_input_kifs(layer)
    k, i, f = np.max(kifs, axis=0)
    return k, i, f


@get_produced_kif.register
def _(layer: keras.layers.Dense | Conv):
    kifs = get_input_kifs(layer)
    assert len(kifs) == 1, f'Dense layer {layer.name} has more than one input. This is not supported.'
    k, i, f = kifs[0]
    w_k, w_i, w_f = get_arr_container(layer.kernel.numpy())
    k = int(k or w_k or np.any(layer.use_bias and layer.bias < 0))  # type: ignore
    i, f = i + w_i, f + w_f
    if isinstance(layer, Conv):
        div = layer.filters
    else:
        div = layer.units

    multiplicity = np.prod(layer.kernel.shape) / div
    i += int(np.ceil(np.log2(multiplicity)))
    return k, i, f


@get_produced_kif.register
def _(layer: Reshape | Flatten | MaxPool3D | MaxPool2D | MaxPool1D):
    kifs = get_input_kifs(layer)
    assert len(kifs) == 1, f'Flatten/Reshape layer {layer.name} has more than one input. This is not supported.'
    k, i, f = kifs[0]
    return k, i, f


@get_produced_kif.register
def _(layer: keras.layers.InputLayer):
    return 1, 65535, 65535


# ===================================================================================================


def get_requested_kif(layer: keras.layers.Layer | FixedPointQuantizer) -> tuple[int, int, int]:
    """Get the bitwidth requested by downstream layers, as a tuple of (k, i, f). By requested, it means the maximum bitwidth that downstream layers may make use of."""
    out_layers = [node.outbound_layer for node in layer._outbound_nodes]
    if len(out_layers) == 0:
        # Terminal case
        return (1, 65535, 65535)
    else:
        # Main case.
        requested_kifs = [get_request_kif(out_layer) for out_layer in out_layers]
        k, i, f = np.max(requested_kifs, axis=0)
        return k, i, f

# ===================================================================================================
# Given a layer, get the bitwidth it requests (maximum bitwidth it can make use of), as a tuple of (k, i, f).
# ===================================================================================================


@singledispatch
def get_request_kif(layer: keras.layers.Layer) -> tuple[int, int, int]:
    """Get the requested bitwidth of a layer, as a tuple of (k, i, f)"""
    if isinstance(layer, (Concatenate, Reshape, Flatten)):
        out_layers: list[keras.layers.Layer] = [node.outbound_layer for node in layer._outbound_nodes]
        if out_layers:
            # Layers that does nothing. Pass through.
            requested_kifs = [get_request_kif(out_layer) for out_layer in out_layers]
            k, i, f = np.max(requested_kifs, axis=0)
            return k, i, f
    return 1, 65535, 65535


@get_request_kif.register
def _(layer: FixedPointQuantizer):
    k, i, f = layer.result_t_kif
    rnd = layer.RND
    sat = layer.SAT
    if rnd.upper() == 'TRN':
        f += 0
    elif rnd.upper() == 'RND':
        f += 1
    else:
        f += 2
    if 'WRAP' not in sat.upper():
        i = 65535
    return k, i, f


# ===================================================================================================


def merge_precision(available: tuple[int, int, int], request: tuple[int, int, int]):
    """Given available precision and the maximum precision can be accepted by downstream, return the precision should be allocated for the data path."""
    k0, i0, f0 = available
    k1, i1, f1 = request

    k = k0
    i = min(i0, i1)
    f = min(f0, f1)
    if f0 > f1 and i0 <= i1:
        i += 1
    return k, i, f


def result_kifRS_layer_with_fusible_quantizer(layer: keras.layers.Layer):
    """Get the result bitwidth of a layer that has a fusible quantizer following immediately, as a tuple of (k, i, f, RND, SAT). When the layer has exactly one quantizer following it, and the quantizer is not heterogenous, the quantizer will be purged during synthesis, and the result bitwidth of the layer will be the same as the quantizer."""
    out_nodes = layer._outbound_nodes
    if len(out_nodes) != 1:
        return
    out_layer = out_nodes[0].outbound_layer
    if not isinstance(out_layer, FixedPointQuantizer):
        return
    if not out_layer.fusible:
        return
    k, i, f = out_layer.result_t_kif
    SAT = out_layer.SAT
    RND = out_layer.RND
    return k, i, f, RND, SAT


def derive_result_kifRS_from_next_quantizers(layer: keras.layers.Layer) -> tuple[int, int, int, str, str]:
    """Get the result bitwidth of a layer that has a quantizer following immediately, as a tuple of (k, i, f, RND, SAT). In general, any InputLayer or layers with kernels will have a quantizer following immediately."""
    Ks, Is, Fs, RNDs, SATs = [], [], [], [], []
    out_layers: list[FixedPointQuantizer] = [node.outbound_layer for node in layer._outbound_nodes]
    for out_layer in out_layers:
        assert isinstance(out_layer, FixedPointQuantizer), f'Layer {layer.name} has a non-quantizer layer {out_layer.name} following it. This is not supported.'
    kifRSs = np.array(list(map(lambda x: (*x.result_t_kif, x.RND, x.SAT), out_layers)), dtype=object)
    Ks, Is, Fs, RNDs, SATs = kifRSs.T
    assert np.all(SATs == SATs[0]), f'Input layer {layer.name} has different SATs. This is not supported.'
    SAT = SATs[0].upper()
    if SAT == 'WRAP_SM':
        warn('WRAP_SM for input layer may lead bitmismatch between proxy and hls4ml due to overflow on th first layer.')
    k = np.max(Ks)
    i = np.max(Is)
    f = np.max(Fs + 2 * (RNDs != 'TRN') - 1 * (RNDs == 'RND'))
    return k, i, f, 'TRN', SAT


def get_result_kifRS(layer: keras.layers.Layer) -> tuple[int, int, int, str, str]:
    """Get the result bitwidth of a layer, as a tuple of (k, i, f, RND, SAT)."""
    result_t = result_kifRS_layer_with_fusible_quantizer(layer)
    if result_t is not None:
        return result_t
    if isinstance(layer, keras.layers.InputLayer):
        return derive_result_kifRS_from_next_quantizers(layer)
    if STREAM and (hasattr(layer, 'kernel')):
        return derive_result_kifRS_from_next_quantizers(layer)

    produced_kif = get_produced_kif(layer)
    requested_kif = get_requested_kif(layer)
    k, i, f = merge_precision(produced_kif, requested_kif)
    return k, i, f, 'TRN', 'WRAP'


def get_config_wight_accum_result_bias(layer: keras.layers.Layer, accum_fp_max_offset: None | int = None):
    """Get the quantization configuration for a layer with kernel in the proxy model."""
    assert hasattr(layer, 'kernel'), f'Layer {layer.name} does not have kernel.'
    r_k, r_i, r_f, RND, SAT = get_result_kifRS(layer)
    p_k, p_i, p_f = get_produced_kif(layer)
    k_k, k_i, k_f = get_arr_container(layer.kernel.numpy())
    weight_t = tuple_to_apf((k_k, k_i, k_f))
    result_t = tuple_to_apf((r_k, r_i, r_f), RND, SAT)
    if accum_fp_max_offset is None:
        a_f = p_f
    else:
        a_f = min(r_f + accum_fp_max_offset, p_f)

    if SAT.upper() == 'WRAP':
        accum_t = tuple_to_apf((r_k, r_i, a_f))
    else:
        k, i, f = p_k, p_i, a_f
        if accum_fp_max_offset is not None:
            warn(f'Layer {layer.name} has SAT={SAT}, and has accum_fp_max_offset set. Proceed only if you know what you are doing.')
            f = min(accum_fp_max_offset + r_f, f)
        accum_t = tuple_to_apf((k, i, f))
    bias_t = accum_t
    return weight_t, accum_t, result_t, bias_t


def get_config_table_tablesize_result(layer: keras.layers.Activation):
    """Get the quantization configuration for a activation layer in the proxy model."""
    assert isinstance(layer, keras.layers.Activation), f'Layer {layer.name} is not an activation layer.'
    k, i, f, RND, SAT = get_result_kifRS(layer)
    result_t = tuple_to_apf((k, i, f), RND, SAT)
    table_t = result_t
    if layer.activation in (keras.activations.tanh, keras.activations.sigmoid):
        result_t = tuple_to_apf((k, i, f), 'TRN', 'WRAP')  # These activations are purely LUT based, this rounding/saturating in the table only would be enough.
    i_kifs = get_input_kifs(layer)
    if len(i_kifs) > 1:
        warn(f'Activation layer {layer.name} has more than one input. Did you just make a activation with multiple inputs? table_size set in this way may make no sense. Proceed only if you know what you are doing.')
    i_k, i_i, i_f = np.max(i_kifs, axis=0)
    table_size = int(2**(i_k + i_i + i_f))  # ...the ideal case. Will be the case if we have universal LUT-based activation.
    if layer.activation is keras.activations.tanh:
        table_size = int(8 / 2.**-i_f)  # LUT Range hardcoded to -4 ~ 4, match #fractional bits
    elif layer.activation is tf.keras.activations.sigmoid:
        table_size = int(16 / 2.**-i_f)  # LUT Range hardcoded to -8 ~ 8, match #fractional bits
    return table_t, table_size, result_t


def get_config(layer: keras.layers.Layer, accum_fp_max_offset: None | int = None):
    """Get the quantization configuration for a layer in the proxy model."""
    if hasattr(layer, 'kernel'):
        weight_t, accum_t, result_t, bias_t = get_config_wight_accum_result_bias(layer, accum_fp_max_offset)
        conf = {'weight_t': weight_t, 'accum_t': accum_t, 'result_t': result_t, 'bias_t': bias_t}
    elif isinstance(layer, keras.layers.Activation):
        table_t, table_size, result_t = get_config_table_tablesize_result(layer)
        conf = {'table_t': table_t, 'table_size': table_size, 'result_t': result_t}
    else:
        k, i, f, RND, SAT = get_result_kifRS(layer)
        result_t = tuple_to_apf((k, i, f), RND, SAT)
        conf = {'result_t': result_t}
    return conf


def _get_next_quantizer(layer: keras.layers.Layer):
    """Get the next quantizer after the layer. Return None if there is no quantizer after the layer."""
    out_layers = [node.outbound_layer for node in layer._outbound_nodes]
    for out_layer in out_layers:
        if isinstance(out_layer, FixedPointQuantizer):
            return out_layer
        r = _get_next_quantizer(out_layer)
        if r is not None:
            return r
    return None


def _get_last_quantizer(layer: keras.layers.Layer):
    """Get the last quantizer before the layer. Return None if there is no quantizer before the layer."""
    assert len(layer._inbound_nodes) <= 1, f'Layer {layer.name} has more than one inbound nodes. This is not supported.'
    in_layers = layer._inbound_nodes[0].inbound_layers
    if not isinstance(in_layers, list):
        in_layers = [in_layers]
    for in_layer in in_layers:
        if isinstance(in_layer, FixedPointQuantizer):
            return in_layer
        r = _get_last_quantizer(in_layer)
        if r is not None:
            return r
    return None


def get_whatever_quantizer(layer: keras.layers.Layer):
    """Find the quantizer before or after the layer."""
    if isinstance(layer, FixedPointQuantizer):
        return layer
    if isinstance(layer, keras.layers.InputLayer):
        pass
    q = _get_next_quantizer(layer) or _get_last_quantizer(layer)
    assert q is not None, f'Layer {layer.name} has no quantizer before or after it. Did you create a valid proxy model?'
    return q


def register_qconf(layer: keras.layers.Layer, accum_fp_max_offset: None | int = None):
    """Get and register quantization configuration for a layer in the proxy model."""
    if hasattr(layer, 'proxy_ready'):
        return
    q = get_whatever_quantizer(layer)
    conf = get_config(layer, accum_fp_max_offset=accum_fp_max_offset)
    overrides = q.overrides or {}
    overrides['layers'].setdefault(layer.name, {}).update(conf)
