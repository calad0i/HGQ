import pytest

import random
import re
sub = re.compile(r'[\(\)\'\[\]]')
from pathlib import Path
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras

test_root_path = Path(__file__).parent
test_root_path = Path('/tmp/unit_test')

try:
    import HGQ
except ImportError:
    pytest.skip("HGQ not installed, skipping tests", allow_module_level=True)

from HGQ.layers.passive_layers import PAvgPool1D, PMaxPool1D, PAvgPool2D, PMaxPool2D, PFlatten, PConcatenate, PReshape
from HGQ.layers.passive_layers import PPool1D, PPool2D
from HGQ import HDense, HQuantize
from HGQ import get_default_pre_activation_quantizer_config, set_default_pre_activation_quantizer_config
from HGQ import trace_minmax
from HGQ.proxy import generate_proxy_model

seed = 42
os.environ['RANDOM_SEED'] = f'{seed}'
np.random.seed(seed)
tf.random.set_seed(seed)
tf.get_logger().setLevel('ERROR')
random.seed(seed)


def get_model(layer: str, rnd_strategy: str, io_type: str):
    pa_config = get_default_pre_activation_quantizer_config()
    pa_config['rnd_strategy'] = rnd_strategy
    pa_config['skip_dims'] = 'all' if io_type == 'io_stream' else 'batch'
    set_default_pre_activation_quantizer_config(pa_config)

    if 'PConcatenate' in layer:
        inp = [keras.Input(shape=(16)), keras.Input(shape=(16))]
        out = [HQuantize()(inp[0]), HQuantize()(inp[1])]
        out = PConcatenate()(out)
    elif 'Pool2D' in layer:
        inp = keras.Input(shape=(4, 4, 1))
        out = HQuantize()(inp)
        out = eval(layer)(out)
    elif 'Pool1D' in layer:
        inp = keras.Input(shape=(8, 2))
        out = HQuantize()(inp)
        out = eval(layer)(out)
    elif 'Flatten' in layer or 'Reshape' in layer:
        inp = keras.Input(shape=(2, 2, 2, 2))
        out = HQuantize()(inp)
        out = eval(layer)(out)
        out = HDense(16)(out)  # Needed to actually do something, or will crash hls4ml
    else:
        raise Exception(f'Please add test for {layer}')
    model = keras.Model(inp, out)

    for layer in model.layers:
        # Randomize weight bitwidths
        if hasattr(layer, 'kernel_quantizer'):
            fbw: tf.Variable = layer.kernel_quantizer.fbw
            fbw.assign(tf.constant(np.random.uniform(4, 10, fbw.shape).astype(np.float32)))
        # And activation bitwidths
        if hasattr(layer, 'pre_activation_quantizer'):
            fbw: tf.Variable = layer.pre_activation_quantizer.fbw
            fbw.assign(tf.constant(np.random.uniform(4, 4.05, fbw.shape).astype(np.float32)))

    return model


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (1000, 16)) * rng.uniform(0, 3, (1, 16))


@pytest.mark.parametrize('rnd_strategy', ['auto', 'floor', 'standard_round'])
@pytest.mark.parametrize('io_type', ['io_stream', 'io_parallel'])
@pytest.mark.parametrize('layer',
                         [
                             "PConcatenate()",
                             "PMaxPool1D(2, padding='same')",
                             "PMaxPool2D((2,2), padding='same')",
                             "PMaxPool1D(2, padding='valid')",
                             "PMaxPool2D((2,2), padding='valid')",
                             "PAvgPool1D(2, padding='same')",
                             "PAvgPool2D((1,2), padding='same')",
                             "PAvgPool2D((2,2), padding='same')",
                             "PAvgPool1D(2, padding='valid')",
                             "PAvgPool2D((1,2), padding='valid')",
                             "PAvgPool2D((2,2), padding='valid')",
                             "PFlatten()",
                             "PReshape((16,))",
                         ]
                         )
def test_layer(data: np.ndarray, layer: str, rnd_strategy: str, io_type: str):
    model = get_model(layer, rnd_strategy, io_type)
    io_type = io_type
    if model.input.__class__ is list:
        dataset = [data, data]
    else:
        dataset = data.reshape(-1, *model.input.shape[1:])  # type: ignore
    trace_minmax(model, dataset, cover_factor=1.0)
    proxy = generate_proxy_model(model)
    r_keras = model.predict(dataset, verbose=0)  # type: ignore
    r_hls: np.ndarray = proxy.predict(dataset).reshape(r_keras.shape)  # type: ignore

    mismatch = r_keras != r_hls
    # if 'AvgPool' in layer:
    #     assert np.allclose(r_hls, r_keras, atol=2**-4, rtol=0), f"Error margin exceeded for AvgPool layer. Max difference: {np.max(np.abs(r_keras - r_hls))}; Avg difference: {np.mean(np.abs(r_keras - r_hls))}; expected error margin: {2**-4}."
    #     # Pool by non-2^n: always non-exact
    #     # Pool by 2^n: still non-exact in general, as accumulator size is NOT configurable (straightforwardly). Is exact in standard_round & io_parallel with pool size 2 by accident, as accumulator size is increased by 1 bit effectively...
    # else:
    assert np.sum(mismatch) == 0, f"Results do not match for {layer} layer: {np.sum(mismatch)} out of {np.prod(mismatch)} entries are different. Sample: {r_keras[mismatch].ravel()[:10] - r_hls[mismatch].ravel()[:10]}"


if __name__ == '__main__':
    test_layer(data(), 'Vivado', "PAvgPool2D((2,2), padding='same', strides=(2,2))", 'standard_round')
