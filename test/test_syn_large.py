import os
import random

import numpy as np
import pytest
import tensorflow as tf
from helpers import get_test_dir, run_model_test, set_seed
from tensorflow import keras

from HGQ import get_default_paq_conf, set_default_paq_conf
from HGQ.layers import HActivation, HAdd, HConv1D, HConv2D, HDense, HQuantize, PConcatenate, PDropout, PFlatten, PMaxPool1D, PMaxPool2D, PReshape


def create_model(rnd_strategy: str, io_type: str):

    pa_config = get_default_paq_conf()
    pa_config['skip_dims'] = 'all' if io_type == 'io_stream' else 'batch'
    pa_config['rnd_strategy'] = rnd_strategy
    set_default_paq_conf(pa_config)

    inp = keras.Input(shape=(10, 10))
    qinp = HQuantize()(inp)
    x = PReshape((10, 10, 1))(qinp)  # 10x10
    x = HConv2D(5, (3, 3), activation='relu', padding='valid', parallel_factor=64)(x)  # 5x8x8
    x = PMaxPool2D((1, 2))(x)  # 5x8x4
    x = HActivation('relu')(x)
    mp = PMaxPool2D((4, 4))(x)  # 5x2x1
    x = PReshape((10,))(mp)  # 5
    x = HDense(10)(x)
    y = PReshape((20, 5))(qinp)  # 20x5
    y = HConv1D(5, 5, strides=5, padding='valid', parallel_factor=4)(y)  # 4x5
    po = PMaxPool1D(2)(y)  # 2x5
    ay = HActivation('relu')(po)
    y = PFlatten()(ay)  # 5
    y = PDropout(0.5)(y)
    y = HDense(10)(y)
    z = PFlatten()(qinp)  # 5
    x = HAdd()([x, y])  # 5
    xy = PConcatenate()([x, y])  # 5
    z = HDense(10, activation='relu')(z)  # 5
    z = HDense(10, activation=None)(z)  # 5
    out = PConcatenate()([xy, z])  # 5
    out = HActivation('relu')(out)
    horrible_model = keras.Model(inp, out)

    for layer in horrible_model.layers:
        # Randomize weight bitwidths
        if hasattr(layer, 'kq'):
            fbw: tf.Variable = layer.kq.fbw
            fbw.assign(tf.constant(np.random.uniform(2, 8, fbw.shape).astype(np.float32)))
        # And activation bitwidths
        if hasattr(layer, 'paq'):
            fbw: tf.Variable = layer.paq.fbw
            fbw.assign(tf.constant(np.random.uniform(2, 6, fbw.shape).astype(np.float32)))
        if hasattr(layer, 'bias') and layer.bias is not None:
            bias: tf.Variable = layer.bias
            bias.assign(tf.constant(np.random.uniform(-4, 4, layer.bias.shape).astype(np.float32)))
    return horrible_model


def get_data(N: int, sigma: float, max_scale: float, seed):
    rng = np.random.default_rng(seed)
    a1 = rng.normal(0, sigma, (N, 10, 10, 1)).astype(np.float32)
    a2 = rng.uniform(0, max_scale, (1, 10, 10, 1)).astype(np.float32)
    return (a1 * a2).astype(np.float32)


@pytest.mark.parametrize("N", [50000, 10])
@pytest.mark.parametrize("rnd_strategy", ['auto'])
@pytest.mark.parametrize("io_type", ['io_parallel', 'io_stream'])
@pytest.mark.parametrize("cover_factor", [0.25, 1.0])
@pytest.mark.parametrize("aggressive", [True, False])
@pytest.mark.parametrize("backend", ['vivado', 'vitis'])
@pytest.mark.parametrize("seed", [42])
def test_syn_large(N: int, rnd_strategy: str, io_type: str, cover_factor: float, aggressive: bool, backend: str, seed: int):
    dir = get_test_dir()
    set_seed(seed)
    model = create_model(rnd_strategy=rnd_strategy, io_type=io_type)
    data = get_data(N, 8, 1, seed)

    run_model_test(model, cover_factor, data, io_type, backend, dir, aggressive)
