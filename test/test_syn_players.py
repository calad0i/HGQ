import numpy as np
import pytest
import tensorflow as tf
from helpers import get_test_dir, run_model_test, set_seed
from tensorflow import keras

import HGQ
from HGQ import get_default_paq_config, set_default_paq_conf
from HGQ.layers import HConv1D, HDense, HQuantize, PAvgPool1D, PAvgPool2D, PConcatenate, PFlatten, PMaxPool1D, PMaxPool2D, PReshape


def create_model(layer: str, rnd_strategy: str, io_type: str):
    pa_config = get_default_paq_config()
    pa_config['rnd_strategy'] = rnd_strategy
    pa_config['skip_dims'] = 'all' if io_type == 'io_stream' else 'batch'
    set_default_paq_conf(pa_config)

    inp = keras.Input(shape=(16))
    if 'PConcatenate' in layer:
        _inp = [HQuantize()(inp)] * 2
    elif 'Pool2D' in layer:
        _inp = PReshape((4, 4, 1))(HQuantize()(inp))
    elif 'Pool1D' in layer:
        _inp = PReshape((16, 1))(HQuantize()(inp))
    elif 'Dense' in layer or 'Activation' in layer:
        _inp = HQuantize()(inp)
    elif 'Flatten' in layer:
        out = HQuantize()(inp)
        out = PReshape((4, 4))(out)
        out = HConv1D(2, 2)(out)
        out = eval(layer)(out)
        out = HDense(16)(out)
        return keras.Model(inp, out)
    else:
        raise Exception(f'Please add test for {layer}')

    out = eval(layer)(_inp)
    model = keras.Model(inp, out)

    for layer in model.layers:
        # No weight bitwidths to randomize
        # And activation bitwidths
        if hasattr(layer, 'paq'):
            fbw: tf.Variable = layer.paq.fbw
            fbw.assign(tf.constant(np.random.uniform(4, 8, fbw.shape).astype(np.float32)))

    return model


def get_data(N: int, sigma: float, max_scale: float, seed):
    rng = np.random.default_rng(seed)
    a1 = rng.normal(0, sigma, (N, 16)).astype(np.float32)
    a2 = rng.uniform(0, max_scale, (1, 16)).astype(np.float32)
    return (a1 * a2).astype(np.float32)


@pytest.mark.parametrize('layer',
                         [
                             "PConcatenate()",
                             "PMaxPool1D(2, padding='same')",
                             "PMaxPool2D((2,2), padding='same')",
                             "PMaxPool1D(2, padding='valid')",
                             "PMaxPool2D((2,2), padding='valid')",
                             #  "PAvgPool1D(2, padding='same')",
                             #  "PAvgPool2D((1,2), padding='same')",
                             #  "PAvgPool2D((2,2), padding='same')",
                             #  "PAvgPool1D(2, padding='valid')",
                             #  "PAvgPool2D((1,2), padding='valid')",
                             #  "PAvgPool2D((2,2), padding='valid')",
                             "PFlatten()",
                         ]
                         )
@pytest.mark.parametrize("N", [1000, 10])
@pytest.mark.parametrize("rnd_strategy", ['auto', 'standard_round', 'floor'])
@pytest.mark.parametrize("io_type", ['io_parallel', 'io_stream'])
@pytest.mark.parametrize("cover_factor", [0.5, 1.0])
@pytest.mark.parametrize("aggressive", [True, False])
@pytest.mark.parametrize("backend", ['vivado'])
@pytest.mark.parametrize("seed", [42])
def test_syn_players(layer, N: int, rnd_strategy: str, io_type: str, cover_factor: float, aggressive: bool, backend: str, seed: int):
    dir = get_test_dir()
    set_seed(seed)
    model = create_model(layer=layer, rnd_strategy=rnd_strategy, io_type=io_type)
    data = get_data(N, 1, 1, seed)

    run_model_test(model, cover_factor, data, io_type, backend, dir, aggressive)


if __name__ == '__main__':
    test_syn_players('PFlatten()', 10, 'floor', 'io_parallel', 0.5, True, 'vivado', 42)
