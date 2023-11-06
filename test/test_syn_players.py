import pytest

import numpy as np
from helpers import set_seed, run_model_test, get_test_dir

import tensorflow as tf
from tensorflow import keras

from HGQ.layers import HDense, HConv1D, HQuantize, PReshape, PConcatenate, PAvgPool1D,PAvgPool2D, PMaxPool1D, PMaxPool2D, PFlatten
from HGQ import get_default_pre_activation_quantizer_config, set_default_pre_activation_quantizer_config
import HGQ

def create_model(layer: str, rnd_strategy: str, io_type: str):
    pa_config = get_default_pre_activation_quantizer_config()
    pa_config['rnd_strategy'] = rnd_strategy
    pa_config['skip_dims'] = 'all' if io_type == 'io_stream' else 'batch'
    set_default_pre_activation_quantizer_config(pa_config)

    inp = keras.Input(shape=(16))
    if 'PConcatenate' in layer:
        _inp = [HQuantize()(inp)] * 2
    elif 'Pool2D' in layer:
        _inp = PReshape((4,4,1))(HQuantize()(inp))
    elif 'Pool1D' in layer:
        _inp = PReshape((16,1))(HQuantize()(inp))
    elif 'Dense' in layer or 'Activation' in layer:
        _inp = HQuantize()(inp)
    elif 'Flatten' in layer:
        out = HQuantize()(inp)
        out = PReshape((4,4))(out)
        out = HConv1D(2,2)(out)
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
        if hasattr(layer, 'pre_activation_quantizer'):
            fbw: tf.Variable = layer.pre_activation_quantizer.fbw
            fbw.assign(tf.constant(np.random.uniform(4, 8, fbw.shape).astype(np.float32)))

    return model

def get_data(N:int, sigma:float, max_scale:float, seed):
    rng = np.random.default_rng(seed)
    a1 = rng.normal(0, sigma, (N,16)).astype(np.float32)
    a2 = rng.uniform(0, max_scale, (1,16)).astype(np.float32)
    return (a1*a2).astype(np.float32)

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
@pytest.mark.parametrize("io_type", ['io_parallel','io_stream'])
@pytest.mark.parametrize("cover_factor", [0.5, 1.0])
@pytest.mark.parametrize("aggressive", [True, False])
@pytest.mark.parametrize("backend", ['vivado'])
@pytest.mark.parametrize("seed", [42])
def test_end2end(layer, N:int, rnd_strategy:str, io_type:str, cover_factor:float, aggressive:bool, backend:str, seed:int):
    dir = get_test_dir()
    set_seed(seed)
    model = create_model(layer=layer,rnd_strategy=rnd_strategy, io_type=io_type)
    data = get_data(N, 1, 1, seed)
    
    run_model_test(model, cover_factor, data, io_type, backend, dir, aggressive)


if __name__ == '__main__':
    test_end2end('PFlatten()', 10, 'floor', 'io_parallel', 0.5, True, 'vivado', 42)