import pytest

import random
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

from HGQ.layers import HQuantize, HDense, HConv2D, HConv1D, PMaxPool2D, PReshape, PMaxPool1D, PFlatten, HActivation
from HGQ import get_default_pre_activation_quantizer_config, set_default_pre_activation_quantizer_config
from HGQ import trace_minmax
from HGQ.proxy import generate_proxy_model

seed = 42
os.environ['RANDOM_SEED'] = f'{seed}'
np.random.seed(seed)
tf.random.set_seed(seed)
tf.get_logger().setLevel('ERROR')
random.seed(seed)


def create_model(rnd_strategy:str, io_type:str):

    pa_config = get_default_pre_activation_quantizer_config()
    pa_config['skip_dims'] = 'all' if io_type == 'io_stream' else 'batch'
    pa_config['rnd_strategy'] = rnd_strategy
    set_default_pre_activation_quantizer_config(pa_config)
    
    model = keras.Sequential([
        HQuantize(input_shape=(10,10,1), name='inp_q'), # 10x10x1
        HConv2D(2, 3, activation='relu'), # 8x8x2
        PMaxPool2D(2), # 4x4x2
        PReshape((8,4)), # 8x4
        HConv1D(4, 3, activation='relu'), # 6x4
        PMaxPool1D(2), # 3x4    2
        PFlatten(), # 12
        HDense(10), # 10
        HActivation('sigmoid')
    ])

    for layer in model.layers:
        # Randomize weight bitwidths
        if hasattr(layer, 'kernel_quantizer'):
            fbw:tf.Variable = layer.kernel_quantizer.fbw
            fbw.assign(tf.constant(np.random.uniform(4,10,fbw.shape).astype(np.float32)))
        # And activation bitwidths
        if hasattr(layer, 'pre_activation_quantizer'):
            fbw:tf.Variable = layer.pre_activation_quantizer.fbw
            fbw.assign(tf.constant(np.random.uniform(2,6,fbw.shape).astype(np.float32)))
    return model

def get_data(N:int, sigma:float, max_scale:float):
    rng = np.random.default_rng(42)
    a1 = rng.normal(0, sigma, (N,10,10,1)).astype(np.float32)
    a2 = rng.uniform(0, max_scale, (1,10,10,1)).astype(np.float32)
    return a1*a2
    
    
@pytest.mark.parametrize("N", [10, 50000])
@pytest.mark.parametrize("rnd_strategy", ['auto', 'standard_round', 'floor'])
@pytest.mark.parametrize("io_type", ['io_parallel','io_stream'])
def test_end2end(N:int, rnd_strategy:str, io_type:str):
    model = create_model(rnd_strategy=rnd_strategy, io_type=io_type)
    data = get_data(N, 1, 3)
    trace_minmax(model, data, cover_factor=1.0)
    r_keras = model.predict(data[1:2], verbose=0) # type: ignore
    print(f"Testing {rnd_strategy}_{io_type}")
    proxy = generate_proxy_model(model)
    model.save(test_root_path / f'hls4ml_prj_hgq_{N}_{rnd_strategy}_{io_type}.h5')
    proxy.save(test_root_path / f'hls4ml_prj_hgq_{N}_{rnd_strategy}_{io_type}_proxy.h5')
    r_hls = proxy.predict(data[1:2], verbose=0) # type: ignore

    mismatch = r_keras != r_hls
    assert np.sum(mismatch)==0, f"Results do not match: {np.sum(np.any(mismatch,axis=1))} out of {N} samples are different. Sample: {r_keras[mismatch].ravel()[:10]} vs {r_hls[mismatch].ravel()[:10]}"

if __name__ == '__main__':
    test_end2end(10, 'standard_round', 'io_parallel')
    