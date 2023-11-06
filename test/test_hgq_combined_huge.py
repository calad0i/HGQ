import os
import pytest

import random
from pathlib import Path
import numpy as np

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow import keras
from tempfile import TemporaryDirectory
import hls4ml

test_root_path = Path('/tmp/unit_test')

try:
    import HGQ
except ImportError:
    pytest.skip("HGQ not installed, skipping tests", allow_module_level=True)

from HGQ.layers import HQuantize, HDense, HConv2D, HConv1D, PMaxPool2D, PReshape, PMaxPool1D, PConcatenate, PFlatten, HActivation, HAdd, PDropout
from HGQ import get_default_pre_activation_quantizer_config, set_default_pre_activation_quantizer_config
from HGQ import trace_minmax
from HGQ.proxy import to_proxy_model


def create_model(rnd_strategy:str, io_type:str, seed:int=42):
    seed = seed
    os.environ['RANDOM_SEED'] = f'{seed}'
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    pa_config = get_default_pre_activation_quantizer_config()
    pa_config['skip_dims'] = 'all' if io_type == 'io_stream' else 'batch'
    pa_config['rnd_strategy'] = rnd_strategy
    set_default_pre_activation_quantizer_config(pa_config)
    
    inp = keras.Input(shape=(10,10,))
    qinp = HQuantize()(inp)
    x = PReshape((10,10,1))(qinp) #10x10
    x = HConv2D(5, (3,3), activation='relu', padding='valid', parallel_factor=64)(x) #5x8x8
    x = PMaxPool2D((1,2))(x) #5x8x4
    x = HActivation('relu')(x)
    mp = PMaxPool2D((4,4))(x) #5x2x1
    x = PReshape((10,))(mp) #5
    x = HDense(10)(x)
    y = PReshape((20,5))(qinp) #20x5
    y = HConv1D(5, 5, strides=5, padding='valid', parallel_factor=4)(y) #4x5
    po = PMaxPool1D(2)(y) #2x5
    ay = HActivation('relu')(po)
    y = PFlatten()(ay) #5
    y = PDropout(0.5)(y)
    y = HDense(10)(y)
    z = PFlatten()(qinp) # 5
    x = HAdd()([x,y]) # 5
    xy = PConcatenate()([x,y]) # 5
    z = HDense(10, activation='relu')(z) # 5
    z = HDense(10, activation=None)(z) # 5
    out = PConcatenate()([xy,z]) # 5
    out = HActivation('relu')(out)
    horrible_model = keras.Model(inp, out)

    for layer in horrible_model.layers:
        # Randomize weight bitwidths
        if hasattr(layer, 'kernel_quantizer'):
            fbw:tf.Variable = layer.kernel_quantizer.fbw
            fbw.assign(tf.constant(np.random.uniform(2,8,fbw.shape).astype(np.float32)))
        # And activation bitwidths
        if hasattr(layer, 'pre_activation_quantizer'):
            fbw:tf.Variable = layer.pre_activation_quantizer.fbw
            fbw.assign(tf.constant(np.random.uniform(2,6,fbw.shape).astype(np.float32)))
    return horrible_model

def get_data(N:int, sigma:float, max_scale:float, seed):
    rng = np.random.default_rng(seed)
    a1 = rng.normal(0, sigma, (N,10,10,1)).astype(np.float32)
    a2 = rng.uniform(0, max_scale, (1,10,10,1)).astype(np.float32)
    return (a1*a2).astype(np.float32)


@pytest.mark.parametrize("N", [50000, 10])
@pytest.mark.parametrize("rnd_strategy", ['auto', 'standard_round', 'floor'])
@pytest.mark.parametrize("io_type", ['io_parallel','io_stream'])
@pytest.mark.parametrize("cover_factor", [0.49, 1.0])
@pytest.mark.parametrize("aggressive", [True, False])
@pytest.mark.parametrize("backend", ['vivado'])
@pytest.mark.parametrize("seed", [1919810, 1919, 910, 114514, 42])
def test_end2end(N:int, rnd_strategy:str, io_type:str, cover_factor:float, aggressive:bool, backend:str, seed:int):
    model = create_model(rnd_strategy=rnd_strategy, io_type=io_type, seed=seed)
    data = get_data(N, 8, 1, seed)
    trace_minmax(model, data, cover_factor=cover_factor)
    proxy = to_proxy_model(model, aggressive=aggressive)
    output_dir=test_root_path / f'{seed}-{backend}-{aggressive}-{cover_factor}-{io_type}-{rnd_strategy}-{N}'
    model.save(output_dir / 'model.h5')
    proxy.save(output_dir / 'proxy.h5')
    r_keras = model(data).numpy() # type: ignore
    r_proxy = proxy(data).numpy() # type: ignore

    mismatch_proxy = r_keras != r_proxy
    if cover_factor >= 1.0:
        assert np.sum(mismatch_proxy)==0, f"Proxy-Keras mismatch: {np.sum(np.any(mismatch_proxy,axis=1))} out of {N} samples are different. Sample: {r_keras[mismatch_proxy].ravel()[:5]} vs {r_proxy[mismatch_proxy].ravel()[:5]}"
    else:
        assert np.sum(mismatch_proxy)>0, f"Proxy-Keras perfect match when overflow should happen: cover_factor={cover_factor}."

    hls_config = {'LayerName':{}, 'Model': {'Precision': 'ap_fixed<32,16>', 'ReuseFactor': 1}}
    for layer in proxy.layers:
        name = layer.name
        hls_config['LayerName'][name] = {'Trace': False}

    model_hls = hls4ml.converters.convert_from_keras_model(proxy, hls_config=hls_config, output_dir=str(output_dir/'hls4ml_prj'), io_type=io_type, backend=backend)
    
    model_hls.compile()
    r_hls:np.ndarray = model_hls.predict(data).reshape(r_proxy.shape) # type: ignore
    mismatch_hls = r_hls != r_proxy
    assert np.sum(mismatch_hls)==0, f"Keras-HLS4ML mismatch: {np.sum(np.any(mismatch_hls,axis=1))} out of {N} samples are different. Sample: {r_proxy[mismatch_hls].ravel()[:5]} vs {r_hls[mismatch_hls].ravel()[:5]}"

if __name__ == '__main__':
    test_end2end(10, 'auto', 'io_stream', 0.49, False, 'vivado', 1919)
