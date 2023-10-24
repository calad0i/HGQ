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
from keras.layers.merging.base_merge import _Merge

test_root_path = Path(__file__).parent
test_root_path = Path('/tmp/unit_test')

try:
    import HGQ
except ImportError:
    pytest.skip("HGQ not installed, skipping tests", allow_module_level=True)

from HGQ.layers import HDense, HConv1D, HConv2D, HActivation, HAdd, HQuantize
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

    if 'Add' in layer:
        inp = [keras.Input(shape=(16)), keras.Input(shape=(16))]
        out = [HQuantize()(inp[0]), HQuantize()(inp[1])]
    elif 'Conv2D' in layer:
        inp = keras.Input(shape=(4, 4, 1))
        out = HQuantize()(inp)
    elif 'Conv1D' in layer:
        inp = keras.Input(shape=(16, 1))
        out = HQuantize()(inp)
    elif 'Dense' in layer or 'Activation' in layer:
        inp = keras.Input(shape=(16,))
        out = HQuantize()(inp)
    else:
        raise Exception(f'Please add test for {layer}')
    out = eval(layer)(out)
    model = keras.Model(inp, out)

    for layer in model.layers:
        # Randomize weight bitwidths
        if hasattr(layer, 'kernel_quantizer'):
            fbw: tf.Variable = layer.kernel_quantizer.fbw
            fbw.assign(tf.constant(np.random.uniform(4, 10, fbw.shape).astype(np.float32)))
        # And activation bitwidths
        if hasattr(layer, 'pre_activation_quantizer'):
            fbw: tf.Variable = layer.pre_activation_quantizer.fbw
            fbw.assign(tf.constant(np.random.uniform(4, 7, fbw.shape).astype(np.float32)))

    return model


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (1000, 16)) * rng.uniform(0, 6, (1, 16))


@pytest.mark.parametrize('rnd_strategy', ['standard_round', 'floor'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('layer',
                         ["HDense(10)",
                          "HConv1D(2, 3, padding='same')",
                          "HConv1D(2, 3, padding='valid')",
                          "HConv2D(2, (3,3), padding='same')",
                          "HConv2D(2, (3,3), padding='valid')",
                          "HAdd()",
                          "HActivation('relu')",
                          "HActivation('leaky_relu')",
                          "HActivation('relu6')",
                          "HActivation('tanh')",
                          "HActivation('sigmoid')",
                          "HActivation('softmax')",
                          ]
                         )
def test_layer(data: np.ndarray, layer: str, rnd_strategy: str, io_type: str):
    model = get_model(layer, rnd_strategy, io_type)
    print(model.layers[-1].pre_activation_quantizer.fbw)
    io_type = io_type
    if model.input.__class__ is list:
        dataset = [data, data]
    else:
        dataset = data.reshape(-1, *model.input.shape[1:])  # type: ignore
    trace_minmax(model, dataset, cover_factor=1.0)
    proxy = generate_proxy_model(model)
    model.save(test_root_path / f'hls4ml_prj_hgq_{layer}_{rnd_strategy}_{io_type}.h5')
    proxy.save(test_root_path / f'hls4ml_prj_hgq_{layer}_{rnd_strategy}_{io_type}_proxy.h5')
    r_keras = model.predict(dataset, verbose=0)  # type: ignore
    r_hls = proxy.predict(dataset, verbose=0).reshape(r_keras.shape)  # type: ignore

    mismatch = r_keras != r_hls

    assert np.sum(mismatch) == 0, f"Results do not match for {layer} layer: {np.sum(mismatch,axis=1)} out of {1000} samples are different. Sample: {r_keras[mismatch].ravel()[:10]} vs {r_hls[mismatch].ravel()[:10]}"


if __name__ == '__main__':
    test_layer(data(), "HAdd()", 'standard_round', 'io_parallel')
