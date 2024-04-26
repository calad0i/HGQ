import numpy as np
import pytest
import tensorflow as tf
from helpers import get_test_dir, run_model_test, set_seed
from tensorflow import keras

import HGQ
from HGQ import get_default_paq_conf, set_default_paq_conf
from HGQ.layers import HConv1D, HDense, HQuantize, PAvgPool1D, PAvgPool2D, PConcatenate, PFlatten, PMaxPool1D, PMaxPool2D, PReshape, Signature
from HGQ.proxy.fixed_point_quantizer import gfixed


def create_model(layer: str, rnd_strategy: str, io_type: str):
    pa_config = get_default_paq_conf()
    pa_config['rnd_strategy'] = rnd_strategy
    pa_config['skip_dims'] = 'all' if io_type == 'io_stream' else 'batch'
    set_default_paq_conf(pa_config)

    inp = keras.Input(shape=(15))
    if 'PConcatenate' in layer:
        _inp = [HQuantize()(inp)] * 2
        out = eval(layer)(_inp)
        out = HDense(15)(out)
        return keras.Model(inp, out)
    elif 'Signature' in layer:
        _inp = eval(layer)(inp)
        out = HDense(15)(_inp)
        return keras.Model(inp, out)
    elif 'Pool2D' in layer:
        _inp = PReshape((3, 5, 1))(HQuantize()(inp))
    elif 'Pool1D' in layer:
        _inp = PReshape((5, 3))(HQuantize()(inp))
    elif 'Dense' in layer or 'Activation' in layer:
        _inp = HQuantize()(inp)
    elif 'Flatten' in layer:
        out = HQuantize()(inp)
        out = PReshape((3, 5))(out)
        out = HConv1D(2, 2)(out)
        out = eval(layer)(out)
        out = HDense(15)(out)
        return keras.Model(inp, out)
    else:
        raise Exception(f'Please add test for {layer}')

    out = eval(layer)(_inp)
    if 'maxpool' in layer.lower():
        out = HQuantize()(out)
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
    a1 = rng.normal(0, sigma, (N, 15)).astype(np.float32)
    a2 = rng.uniform(0, max_scale, (1, 15)).astype(np.float32)
    return (a1 * a2).astype(np.float32)


@pytest.mark.parametrize('layer',
                         [
                             "PConcatenate()",
                             "PMaxPool1D(2, padding='same')",
                             "PMaxPool1D(4, padding='same')",
                             "PMaxPool2D((5,3), padding='same')",
                             "PMaxPool1D(2, padding='valid')",
                             "PMaxPool2D((2,3), padding='valid')",
                             "Signature(1,6,3)",
                             "PAvgPool1D(2, padding='same')",
                             "PAvgPool2D((1,2), padding='same')",
                             "PAvgPool2D((2,2), padding='same')",
                             "PAvgPool1D(2, padding='valid')",
                             "PAvgPool2D((1,2), padding='valid')",
                             "PAvgPool2D((2,2), padding='valid')",
                             "PFlatten()",
                         ]
                         )
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("rnd_strategy", ['auto', 'standard_round', 'floor'])
@pytest.mark.parametrize("io_type", ['io_parallel', 'io_stream'])
@pytest.mark.parametrize("cover_factor", [0.25, 1.0])
@pytest.mark.parametrize("aggressive", [True, False])
@pytest.mark.parametrize("backend", ['vivado', 'vitis'])
@pytest.mark.parametrize("seed", [42])
def test_syn_players(layer, N: int, rnd_strategy: str, io_type: str, cover_factor: float, aggressive: bool, backend: str, seed: int):
    dir = get_test_dir()
    set_seed(seed)
    model = create_model(layer=layer, rnd_strategy=rnd_strategy, io_type=io_type)
    data = get_data(N, 1, 1, seed)

    if 'Signature' in layer:
        q = gfixed(1, 6, 3)
        data = q(data).numpy()
    if "padding='same'" in layer and io_type == 'io_stream':
        pytest.skip("io_stream does not support padding='same' for pools at the moment")

    run_model_test(model, cover_factor, data, io_type, backend, dir, aggressive)


if __name__ == '__main__':
    test_syn_players('PFlatten()', 10, 'floor', 'io_parallel', 0.5, True, 'vivado', 42)
