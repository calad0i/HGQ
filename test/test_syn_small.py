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

    model = keras.Sequential([
        HQuantize(input_shape=(6, 6, 1), name='inp_q'),
        HConv2D(4, 2, activation='relu'),
        PMaxPool2D(2),
        PReshape((8, 2)),
        HConv1D(4, 3, activation='relu', padding='same'),
        PMaxPool1D(2),
        PFlatten(),
        HDense(10),
        HActivation('sigmoid')
    ])

    for layer in model.layers:
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
    return model


def get_data(N: int, sigma: float, max_scale: float, seed):
    rng = np.random.default_rng(seed)
    a1 = rng.normal(0, sigma, (N, 6, 6, 1)).astype(np.float32)
    a2 = rng.uniform(0, max_scale, (1, 6, 6, 1)).astype(np.float32)
    return (a1 * a2).astype(np.float32)


@pytest.mark.parametrize("N", [50000, 10])
@pytest.mark.parametrize("rnd_strategy", ['auto', 'standard_round', 'floor'])
@pytest.mark.parametrize("io_type", ['io_parallel', 'io_stream'])
@pytest.mark.parametrize("cover_factor", [0.25, 1.0])
@pytest.mark.parametrize("aggressive", [True, False])
@pytest.mark.parametrize("backend", ['vivado', 'vitis'])
@pytest.mark.parametrize("seed", [114514, 42, 1919810])
def test_syn_small(N: int, rnd_strategy: str, io_type: str, cover_factor: float, aggressive: bool, backend: str, seed: int):
    dir = get_test_dir()
    set_seed(seed)
    model = create_model(rnd_strategy=rnd_strategy, io_type=io_type)
    data = get_data(N, 8, 1, seed)

    run_model_test(model, cover_factor, data, io_type, backend, dir, aggressive)


if __name__ == '__main__':
    test_syn_small(10, 'auto', 'io_parallel', 0.49, True, 'vivado', 1919810)
