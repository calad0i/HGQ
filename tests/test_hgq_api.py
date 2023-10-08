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

import hls4ml

test_root_path = Path(__file__).parent
test_root_path = Path('/tmp/unit_test')

try:
    import HGQ
except ImportError:
    pytest.skip("HGQ not installed, skipping tests", allow_module_level=True)

seed = 42
os.environ['RANDOM_SEED'] = f'{seed}'
np.random.seed(seed)
tf.random.set_seed(seed)
tf.get_logger().setLevel('ERROR')
random.seed(seed)



def create_model(rnd_strategy:str, io_type:str):
    from HGQ.layers import HQuantize, HDense, HConv2D, HConv1D, PMaxPool2D, PReshape, PMaxPool1D, PFlatten
    from HGQ import get_default_pre_activation_quantizer_config, set_default_pre_activation_quantizer_config
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
        PMaxPool1D(2), # 3x4
        PFlatten(), # 12
        HDense(10), # 10
    ])

    for layer in model.layers:
        # Randomize weight bitwidths
        if hasattr(layer, 'kernel_quantizer'):
            fbw:tf.Variable = layer.kernel_quantizer.fbw
            fbw.assign(tf.constant(np.random.uniform(4,10,fbw.shape).astype(np.float32)))
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
@pytest.mark.parametrize('backend', ['Vitis', 'Vivado'])
def test_end2end(N:int, rnd_strategy:str, io_type:str, backend:str):
    from HGQ import compute_bops
    model = create_model(rnd_strategy=rnd_strategy, io_type=io_type)
    data = get_data(N, 1, 3)
    compute_bops(model, data, cover_factor=1.0)
    r_keras = model.predict(data[1:2], verbose=0) # type: ignore
    from HGQ.hls4ml_hook import convert_from_hgq_model
    model.save(test_root_path / f'hls4ml_prj_hgq_{N}_{rnd_strategy}_{io_type}_{backend}.h5')
    print(f"Testing {rnd_strategy}_{io_type}_{backend}")
    print(f"  {model.layers[-1].pre_activation_quantizer._max}")
    print(f"  {model.layers[-1].pre_activation_quantizer._min}")
    model_hls = convert_from_hgq_model(
        model,
        hls_config=None,
        output_dir=str(test_root_path / f'hls4ml_prj_hgq_{N}_{rnd_strategy}_{io_type}_{backend}'),
        io_type=io_type,
        backend=backend,
        )
    model_hls.compile()
    r_hls = model_hls.predict(data[1:2]).reshape((1,10))

    print(r_keras)
    print(r_hls)
    mask = r_keras != r_hls
    assert np.sum(mask)==0, f"Results do not match: {np.sum(np.any(mask,axis=1))} out of {N} samples are different. Sample: {r_keras[mask].ravel()[:10]} vs {r_hls[mask].ravel()[:10]}"

# def get_curated_data(N:int):
#     rng = np.random.default_rng(42)
#     a1 = rng.uniform(0, 1, (N, 10)).astype(np.float32)
#     a2 = np.array()    

# @pytest.mark.parametrize("layer", ["HConv2D", "HConv1D", "HDense", "PMaxPool2D", "PMaxPool1D", "PFlatten", "PReshape", "HQuantize"])
# def test_layer(layer):
#     import HGQ.layers
#     layer = getattr(HGQ.layers, layer)
    
#     model = keras.Sequential([
#         HGQ.layers.Signature()
#     ])

if __name__ == '__main__':
    test_end2end(10, 'standard_round', 'io_parallel', 'Vitis')
    