from dataclasses import dataclass, field
from functools import singledispatch
from pathlib import Path

import h5py as h5
import keras
import numpy as np
import pandas as pd

from HGQ.proxy import FixedPointQuantizer, UnaryLUT
from HGQ.quantizer.quantizer import get_arr_bits
from HGQ.utils import apf_to_tuple, tuple_to_apf, warn

co = {'FixedPointQuantizer': FixedPointQuantizer, 'UnaryLUT': UnaryLUT}

GRID_SIZE = 12, 16  # fixed_bw, variable_bw


def get_hls_config(model: keras.Model):
    "Get hls config from proxy keras model"
    layer_config = {}
    for layer in model.layers:
        if isinstance(layer, FixedPointQuantizer):
            _layer_config = layer.overrides['layers']
            layer_config.update(_layer_config)
            continue

        if isinstance(layer, UnaryLUT):
            k, i, f = layer.kif_out
            result_t = tuple_to_apf((k, i, f))
            layer_config[layer.name] = {'result_t': result_t, 'table_size': layer.table.shape[0]}

    return layer_config


def get_last_layer(layer: keras.layers.Layer):
    if isinstance(layer, keras.layers.InputLayer):
        raise ValueError(f"Unsupported layer: layer {layer.name} is an input layer")
    inbound_node = layer._inbound_nodes[0]
    if isinstance(inbound_node.inbound_layers, list):
        raise ValueError(f"Unsupported layer: layer {layer.name} has multiple inbound layers")
    return inbound_node.inbound_layers


def get_input_bw(layer: keras.layers.Layer):
    assert not isinstance(layer, keras.layers.InputLayer)
    assert not isinstance(layer, FixedPointQuantizer)
    inbound_layer = layer
    while not isinstance(inbound_layer, FixedPointQuantizer):
        inbound_layer = get_last_layer(inbound_layer)
    bits = np.array(inbound_layer.bits)
    shape = inbound_layer.output_shape[1:]
    if bits.shape == ():
        return np.broadcast_to(bits, shape)
    return bits.reshape(shape)


def get_ker_bw(layer: keras.layers.Layer):
    kernel = layer.kernel.numpy()
    bits = np.sum(get_arr_bits(kernel)[1:], axis=0)
    return bits


# @singledispatch
# def get_mul_oprs_grid(layer: keras.layers.Layer) -> np.ndarray:
#     "Get mul feature grid (fixed_bits x variable_bits) from a layer"
#     raise ValueError(f"Unsupported layer: layer {layer.__class__} is not supported")


def mat_vec_mul_feature(mat_bw: np.ndarray, vec_bw: np.ndarray):
    assert mat_bw.ndim == 2
    assert vec_bw.ndim == 1
    assert mat_bw.shape[0] == vec_bw.shape[0]
    grid = np.zeros((GRID_SIZE[0] + 1, GRID_SIZE[1] + 1), dtype=np.int32)
    for var_bw, fixed_bws in zip(vec_bw, mat_bw):
        var_bw = np.clip(var_bw, 0, GRID_SIZE[1])
        fixed_bws = np.clip(fixed_bws, 0, GRID_SIZE[0])
        loc, inc = np.unique(fixed_bws, return_counts=True)
        grid[loc, var_bw] += inc
    return grid[1:, 1:]


# @get_mul_oprs_grid.register
# def _(layer: keras.layers.Dense):
#     input_bw = get_input_bw(layer)
#     ker_bw = get_ker_bw(layer)
#     return mat_vec_mul_feature(ker_bw, input_bw)


# @get_mul_oprs_grid.register
# def _(layer: keras.layers.Conv1D):
#     # *kernel_sizes, in_ch, out_ch
#     padding = layer.padding.upper()
#     input_bw = get_input_bw(layer)
#     ker_bw = get_ker_bw(layer)
#     ker_size = ker_bw.shape[0]
#     seq_len = input_bw.shape[0]
#     if padding == 'SAME':
#         _input_bw = np.zeros((ker_size + seq_len - 1, *input_bw.shape[1:]), dtype=np.int32)
#         _input_bw[ker_size // 2: ker_size // 2 + seq_len] = input_bw
#         input_bw = _input_bw
#     elif padding == 'CAUSAL':
#         _input_bw = np.zeros((ker_size + seq_len - 1, *input_bw.shape[1:]), dtype=np.int32)
#         _input_bw[-seq_len:] = input_bw
#         input_bw = _input_bw

#     seq_len += ker_size - 1
#     grid = np.zeros((GRID_SIZE[0], GRID_SIZE[1]), dtype=np.int32)
#     for ker_loc in range(ker_size):
#         var_bws = input_bw[ker_loc: ker_loc + input_bw.shape[0] - ker_size + 1]
#         ker_slice_bw = ker_bw[ker_loc]
#         for var_bw in var_bws:
#             grid += mat_vec_mul_feature(ker_slice_bw, var_bw)

#     return grid


@dataclass
class LayerSurrogateFeature:
    name: str
    reuse_factor: int
    n_partition: int = 1
    mul_oprs: np.ndarray = field(default_factory=lambda: np.array(0))
    '[fixed_bits, variable_bits]'
    cmp_oprs: np.ndarray = field(default_factory=lambda: np.array(0))
    '[variable_bits]'
    lut_oprs: np.ndarray = field(default_factory=lambda: np.array(0))
    '[variable_bits]'

    def save(self, f: h5.Group):
        g = f.create_group(self.name)
        g.attrs['reuse_factor'] = self.reuse_factor
        g.attrs['n_partition'] = self.n_partition
        compression = 'gzip' if self.mul_oprs.size > 1 else None  # type:ignore
        g.create_dataset('mul_oprs', data=self.mul_oprs, compression=compression)
        compression = 'gzip' if self.cmp_oprs.size > 1 else None  # type:ignore
        g.create_dataset('cmp_oprs', data=self.cmp_oprs, compression=compression)
        compression = 'gzip' if self.lut_oprs.size > 1 else None  # type:ignore
        g.create_dataset('lut_oprs', data=self.lut_oprs, compression=compression)

    @classmethod
    def load(cls, f: h5.Group, name: str):
        g: h5.Group = f[name]  # type:ignore
        if 'lut_oprs' in g.keys():
            lut = np.array(g['lut_oprs'])  # type:ignore
        else:
            lut = np.array(0)
        return cls(name,
                   g.attrs['reuse_factor'],  # type:ignore
                   g.attrs['n_partition'],  # type:ignore
                   np.array(g['mul_oprs']),  # type:ignore
                   np.array(g['cmp_oprs']),  # type:ignore
                   lut  # type:ignore
                   )

    @property
    def bops(self):
        a, b = np.arange(GRID_SIZE[0]) + 1, np.arange(GRID_SIZE[1]) + 1
        s = np.sum(self.mul_oprs * a[:, None] * b[None, :]).astype(int)
        return s / self.n_partition

    def flatten(self):
        mul_oprs = np.broadcast_to(self.mul_oprs, GRID_SIZE)
        cmp_oprs = np.broadcast_to(self.cmp_oprs, GRID_SIZE[1])
        lut_oprs = np.broadcast_to(self.lut_oprs, GRID_SIZE[1])
        return np.hstack([mul_oprs.ravel(), cmp_oprs, lut_oprs])


@singledispatch
def get_layer_feature(layer, hls_conf: dict) -> LayerSurrogateFeature | None:
    return None


@get_layer_feature.register
def _(layer: keras.layers.Dense, hls_conf: dict):
    input_bw = get_input_bw(layer)
    ker_bw = get_ker_bw(layer)
    mul_oprs = mat_vec_mul_feature(ker_bw, input_bw)
    n_partition = 1
    reuse_factor = hls_conf[layer.name].get('reuse_factor', 1)
    return LayerSurrogateFeature(layer.name, reuse_factor, n_partition, mul_oprs=mul_oprs)


@get_layer_feature.register
def _(layer: keras.layers.Conv1D, hls_conf: dict):
    padding = layer.padding.upper()
    input_bw = get_input_bw(layer)
    ker_bw = get_ker_bw(layer)
    ker_size = ker_bw.shape[0]
    seq_len = input_bw.shape[0]

    # Pad input_bw and and merge all conds to padding='VALID'
    if padding == 'SAME':
        _input_bw = np.zeros((ker_size + seq_len - 1, *input_bw.shape[1:]), dtype=np.int32)
        _input_bw[ker_size // 2: ker_size // 2 + seq_len] = input_bw
        input_bw = _input_bw
    elif padding == 'CAUSAL':
        _input_bw = np.zeros((ker_size + seq_len - 1, *input_bw.shape[1:]), dtype=np.int32)
        _input_bw[-seq_len:] = input_bw
        input_bw = _input_bw

    seq_len += ker_size - 1
    mul_oprs = np.zeros((GRID_SIZE[0], GRID_SIZE[1]), dtype=np.int32)
    for ker_loc in range(ker_size):
        var_bws = input_bw[ker_loc: ker_loc + input_bw.shape[0] - ker_size + 1]
        ker_slice_bw = ker_bw[ker_loc]
        for var_bw in var_bws:
            mul_oprs += mat_vec_mul_feature(ker_slice_bw, var_bw)

    parallelization_factor = hls_conf[layer.name]['parallelization_factor']
    n_partition = (input_bw.shape[0] - ker_size + 1) // parallelization_factor
    reuse_factor = hls_conf[layer.name].get('reuse_factor', 1)

    return LayerSurrogateFeature(layer.name, reuse_factor, n_partition, mul_oprs=mul_oprs)


@get_layer_feature.register
def _(layer: keras.layers.ReLU, hls_conf: dict):
    input_bw = get_input_bw(layer)
    arr = np.zeros(GRID_SIZE[1] + 1, dtype=np.int32)
    loc, inc = np.unique(np.minimum(input_bw, GRID_SIZE[1]), return_counts=True)
    arr[loc] = inc
    reuse_factor = hls_conf[layer.name].get('reuse_factor', 1)
    cmp = LayerSurrogateFeature(layer.name, reuse_factor, 1, cmp_oprs=arr[1:])
    return cmp


@get_layer_feature.register
def _(layer: keras.layers.Activation, hls_conf: dict):
    input_bw = get_input_bw(layer)
    if layer.activation is keras.activations.relu:
        return get_layer_feature.registry[keras.layers.ReLU](layer, hls_conf)

    if layer.activation in (keras.activations.tanh, keras.activations.sigmoid):
        arr = np.zeros(GRID_SIZE[1] + 1, dtype=np.int32)
        loc, inc = np.unique(np.minimum(input_bw, GRID_SIZE[1]), return_counts=True)
        arr[loc] = inc * 2

        name = layer.name
        table_size = hls_conf[name]['table_size']
        out_bw = sum(apf_to_tuple(hls_conf[name]['result_t']))
        lut_oprs = np.zeros(GRID_SIZE[1], dtype=np.int32)
        lut_oprs[out_bw - 1] = table_size
        return LayerSurrogateFeature(layer.name, 1, 1, cmp_oprs=arr[1:], lut_oprs=lut_oprs)

    warn(f"Unsupported activation: {layer.activation}")
    return None


@get_layer_feature.register
def _(layer: UnaryLUT, hls_conf: dict):
    name = layer.name
    assert layer.built, f"Layer {name} is not built"
    table_size = layer.table.shape[0]  # type:ignore
    out_bw = sum(layer.kif_out)
    bw = np.sum(get_arr_bits(layer.table.numpy()), axis=0)  # type:ignore
    lut_oprs = np.zeros(GRID_SIZE[1] + 1, dtype=np.int32)
    loc, inc = np.unique(np.minimum(bw, GRID_SIZE[1]), return_counts=True)
    lut_oprs[loc] = inc
    reuse_factor = hls_conf[layer.name].get('reuse_factor', 1)
    return LayerSurrogateFeature(layer.name, reuse_factor, 1, lut_oprs=lut_oprs[1:])
