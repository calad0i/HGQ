import numpy as np
import pytest
import tensorflow as tf
from helpers import get_test_dir, run_model_test, set_seed
from tensorflow import keras

import HGQ
from HGQ import get_default_paq_conf, set_default_paq_conf
from HGQ.layers import HQuantize, PReshape


def create_model(layer: str, rnd_strategy: str, io_type: str):
    pa_config = get_default_paq_conf()
    pa_config['rnd_strategy'] = rnd_strategy
    pa_config['skip_dims'] = 'all' if io_type == 'io_stream' else 'batch'
    set_default_paq_conf(pa_config)

    inp = keras.Input(shape=(16))
    if 'Add' in layer:
        _inp = [HQuantize()(inp)] * 2
    elif 'Conv2D' in layer:
        _inp = PReshape((4, 4, 1))(HQuantize()(inp))
    elif 'Conv1D' in layer:
        _inp = PReshape((16, 1))(HQuantize()(inp))
    elif 'Dense' in layer or 'Activation' in layer:
        _inp = HQuantize()(inp)
    else:
        raise Exception(f'Please add test for {layer}')

    _layer = eval('HGQ.layers.' + layer)
    if hasattr(_layer, 'bias') and _layer.bias is not None:
        bias: tf.Variable = _layer.bias
        bias.assign(tf.constant(np.random.uniform(-4, 4, _layer.bias.shape).astype(np.float32)))

    out = _layer(_inp)
    model = keras.Model(inp, out)

    for layer in model.layers:
        # Randomize weight bitwidths
        if hasattr(layer, 'kq'):
            fbw: tf.Variable = layer.kq.fbw
            fbw.assign(tf.constant(np.random.uniform(2, 8, fbw.shape).astype(np.float32)))
        # And activation bitwidths
        if hasattr(layer, 'paq'):
            fbw: tf.Variable = layer.paq.fbw
            fbw.assign(tf.constant(np.random.uniform(2, 8, fbw.shape).astype(np.float32)))

    return model


def get_data(N: int, sigma: float, max_scale: float, seed):
    rng = np.random.default_rng(seed)
    a1 = rng.normal(0, sigma, (N, 16)).astype(np.float32)
    a2 = rng.uniform(0, max_scale, (1, 16)).astype(np.float32)
    return (a1 * a2).astype(np.float32)


def softmax_cond(proxy, hls):
    match_precent = np.mean(np.argmax(proxy, axis=1) == np.argmax(hls, axis=1))
    assert match_precent > 0.90, f"Proxy-HLS4ML mismatch: {(1-match_precent) * 100}% of samples are different. Sample: {proxy[:5]} vs {hls[:5]}"


def custom_fn(x):
    return tf.sin(x)**2. - x  # type: ignore


@pytest.mark.parametrize('layer',
                         ["HDense(10)",
                          "HDense(10, use_bias=False)",
                          "HDenseBatchNorm(10)",
                          "HConv1D(2, 3, padding='same')",
                          "HConv1D(2, 3, padding='valid')",
                          "HConv1D(2, 3, padding='valid', use_bias=False)",
                          "HConv1D(2, 3, padding='valid', strides=2)",
                          "HConv1D(2, 3, padding='same', strides=2)",
                          "HConv1DBatchNorm(2, 3, padding='valid')",
                          "HConv2D(2, (3,3), padding='same')",
                          "HConv2D(2, (3,3), padding='valid')",
                          "HConv2D(2, (3,3), padding='valid', use_bias=False)",
                          "HConv2D(2, (3,3), padding='valid', strides=2)",
                          "HConv2D(2, (3,3), padding='same', strides=2)",
                          "HConv2DBatchNorm(2, (3,3), padding='valid')",
                          "HAdd()",
                          "HActivation('relu')",
                          #   "HActivation('leaky_relu')",
                          "HActivation('tanh')",
                          "HActivation('sigmoid')",
                          "HActivation('softmax')",
                          "HActivation(custom_fn)",
                          ]
                         )
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("rnd_strategy", ['auto', 'standard_round', 'floor'])
@pytest.mark.parametrize("io_type", ['io_parallel', 'io_stream'])
@pytest.mark.parametrize("cover_factor", [0.25, 1.0])
@pytest.mark.parametrize("aggressive", [True, False])
@pytest.mark.parametrize("backend", ['vivado', 'vitis'])
@pytest.mark.parametrize("seed", [42])
def test_syn_hlayers(layer, N: int, rnd_strategy: str, io_type: str, cover_factor: float, aggressive: bool, backend: str, seed: int):
    dir = get_test_dir()
    set_seed(seed)
    model = create_model(layer=layer, rnd_strategy=rnd_strategy, io_type=io_type)
    data = get_data(N, 1, 1, seed)

    test_grad = N > 100
    cond = None
    if 'softmax' in layer:
        cond = softmax_cond
        cover_factor = max(cover_factor, 0.5)  # Softmax table size is sensitive to range

    skip_sl_test = 'custom' in layer
    run_model_test(model,
                   cover_factor, data,
                   io_type,
                   backend,
                   dir,
                   aggressive,
                   test_gard=test_grad,
                   cond=cond,
                   skip_sl_test=skip_sl_test,
                   )


if __name__ == '__main__':
    test_syn_hlayers('HActivation("tanh")', 10, 'floor', 'io_parallel', 0.5, True, 'vivado', 42)
