import pytest

try:
    import qkeras
    from qkeras import quantizers
except ImportError:
    pytest.skip('qkeras not installed', allow_module_level=True)

import numpy as np
from helpers import get_test_dir, run_model_test, set_seed
from tensorflow import keras


def create_model():

    l = lambda X: qkeras.QDense(X, kernel_quantizer='quantized_bits(8, 3, alpha=1.)', activation='quantized_bits(5, 3, alpha=1.)', bias_quantizer='quantized_bits(8, 3, alpha=1)')
    c = lambda X: qkeras.QConv2D(X, 2, kernel_quantizer=quantizers.quantized_bits(8, 3, alpha=1.), activation=quantizers.quantized_bits(6, 1, alpha=1., symmetric=True), bias_quantizer=quantizers.quantized_bits(8, 3, alpha=1))
    r = lambda X: keras.layers.Activation('relu')(X)
    rr = lambda X: qkeras.QActivation('quantized_relu(5, 3)')(X)

    L = l(10)

    inp = keras.layers.Input((10,))
    q_inp = qkeras.QActivation(qkeras.quantized_bits(8, 4, alpha=1.))(inp)
    out = L(q_inp)
    out1 = L(out)
    out1 = r(out1)
    out1 = L(out1)
    out2 = l(20)(out1)
    out2 = r(out2)
    out3 = l(20)(out2)
    out2 = keras.layers.Reshape((2, 5, 2))(out2)
    out2 = c(2)(out2)
    out2 = rr(out2)
    out2 = keras.layers.Flatten()(out2)
    out = keras.layers.Concatenate()([out2, out3])
    model = keras.models.Model(inp, out)
    return model


def get_data(N: int, sigma: float, max_scale: float, seed):
    rng = np.random.default_rng(seed)
    a1 = rng.normal(0, sigma, (N, 10)).astype(np.float32)
    a2 = rng.uniform(0, max_scale, (1, 10)).astype(np.float32)
    return (a1 * a2).astype(np.float32)


@pytest.mark.parametrize("N", [50000, 10])
@pytest.mark.parametrize("io_type", ['io_parallel', 'io_stream'])
@pytest.mark.parametrize("aggressive", [False])
@pytest.mark.parametrize("backend", ['vivado', 'vitis', 'Quartus'])
@pytest.mark.parametrize("seed", [42, 114514, 1919810])
def test_syn_qkeras(N: int, io_type: str, aggressive: bool, backend: str, seed: int):
    dir = get_test_dir()
    set_seed(seed)
    model = create_model()
    data = get_data(N, 8, 1, seed)

    run_model_test(model, None, data, io_type, backend, dir, aggressive, skip_sl_test=True)


# if __name__ == '__main__':
#     test_syn_small(10, 'auto', 'io_parallel', 0.49, True, 'vivado', 1919810)
