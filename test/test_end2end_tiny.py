import keras
import numpy as np
import pytest
from helpers import get_test_dir, run_model_test, set_seed

from HGQ import FreeBOPs, HDense, HDenseBatchNorm, HQuantize, ResetMinMax, get_default_kq_conf, get_default_paq_conf, set_default_kq_conf, set_default_paq_conf, trace_minmax


def get_model(io_type: str, rnd_strategy) -> keras.models.Model:
    beta = 5e-5

    paq_conf = get_default_paq_conf()

    paq_conf['skip_dims'] = 'all' if io_type == 'io_stream' else 'batch'
    paq_conf['rnd_strategy'] = rnd_strategy
    paq_conf['init_bw'] = 8

    set_default_paq_conf(paq_conf)

    kq_conf = get_default_kq_conf()

    kq_conf['init_bw'] = 8

    set_default_kq_conf(kq_conf)

    model = keras.models.Sequential([
        HQuantize(input_shape=(1,)),
        HDense(16, activation='relu', beta=beta),
        HDense(16, activation='relu', beta=beta),
        HDense(16, activation='relu', beta=beta),
        HDense(1),
    ])

    loss = keras.losses.MSE
    opt = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(loss=loss, optimizer=opt)
    return model


def data():
    x = np.linspace(-np.pi, np.pi, 10000).astype(np.float32).reshape(-1, 1)
    y = np.sin(x)
    return x, y


@pytest.mark.parametrize('io_type', ['io_stream', 'io_parallel'])
@pytest.mark.parametrize('seed', [42, 114514, 1919810])
@pytest.mark.parametrize('rnd_strategy', ['standard_round', 'stochastic_round', 'fast_uniform_noise_injection', 'floor'])
def test_end2end(io_type: str, seed: int, rnd_strategy: str):

    set_seed(seed)

    model = get_model(io_type, rnd_strategy)
    x, y = data()

    callbacks = [FreeBOPs(), ResetMinMax()]
    logs = model.fit(x, y, epochs=10, batch_size=500, callbacks=callbacks, verbose=0)  # type: ignore

    mse = np.mean((model(x) - y)**2)
    assert mse < 0.01 if 'io_stream' else 0.005, f'Obtained MSE={mse}, expected < {0.01 if "io_stream" else 0.005}'

    bops: float = trace_minmax(model, x, verbose=False)  # type: ignore
    assert bops < 25000 if 'io_stream' else 20000, f'Obtained BOPs={bops}, expected < {25000 if "io_stream" else 20000}'
    assert np.mean(np.sign(np.diff(logs.history['bops']))) < -0.5, 'BOPs should decrease during training, but it does not'

    run_model_test(model, 1.0, x, io_type, 'vivado', get_test_dir(), True)
