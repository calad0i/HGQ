import json
import os
import random
import shutil
from warnings import warn

import numpy as np
import pytest
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)  # noqa
tf.config.threading.set_inter_op_parallelism_threads(1)  # noqa
tf.config.threading.set_intra_op_parallelism_threads(1)  # noqa

from hls4ml.converters import convert_from_keras_model
from tensorflow import keras

from HGQ import trace_minmax
from HGQ.proxy import FixedPointQuantizer, to_proxy_model

tf.get_logger().setLevel('ERROR')


def get_test_dir() -> str:
    cur_test = os.environ.get('PYTEST_CURRENT_TEST')
    if not cur_test:
        cur_test = 'test_run'
    cur_test = cur_test.replace('::', '_').replace('(call)', '')
    test_root = os.environ.get('TEST_ROOT_DIR')
    if not test_root:
        test_root = '/tmp/unit_test'
    test_dir = os.path.join(test_root, cur_test)
    os.makedirs(test_dir, exist_ok=True)
    return test_dir


def _run_synth_match_test(proxy: keras.Model, data, io_type: str, backend: str, dir: str, cond=None):

    output_dir = dir + '/hls4ml_prj'
    hls_model = convert_from_keras_model(
        proxy,
        io_type=io_type,
        output_dir=output_dir,
        backend=backend,
        hls_config={'Model': {'Precision': 'fixed<1,0>', 'ReuseFactor': 1}}
    )
    hls_model.compile()

    data_len = data.shape[0] if isinstance(data, np.ndarray) else data[0].shape[0]
    # Multiple output case. Check each output separately
    if len(proxy.outputs) > 1:  # type: ignore
        r_proxy: list[np.ndarray] = [x.numpy() for x in proxy(data)]  # type: ignore
        r_hls: list[np.ndarray] = hls_model.predict(data)  # type: ignore
        r_hls = [x.reshape(r_proxy[i].shape) for i, x in enumerate(r_hls)]
    else:
        r_proxy: list[np.ndarray] = [proxy(data).numpy()]  # type: ignore
        r_hls: list[np.ndarray] = [hls_model.predict(data).reshape(r_proxy[0].shape)]  # type: ignore

    errors = []
    for i, (p, h) in enumerate(zip(r_proxy, r_hls)):
        try:
            if cond is None:
                mismatch_ph = p != h
                assert np.sum(mismatch_ph) == 0, f"Proxy-HLS4ML mismatch for out {i}: {np.sum(np.any(mismatch_ph,axis=1))} out of {data_len} samples are different. Sample: {p[mismatch_ph].ravel()[:5]} vs {h[mismatch_ph].ravel()[:5]}"
            else:
                cond(p, h)
        except AssertionError as e:
            errors.append(e)
    if len(errors) > 0:
        msgs = [str(e) for e in errors]
        raise AssertionError('\n'.join(msgs))


def _run_model_sl_test(model: keras.Model, proxy: keras.Model, data, output_dir: str):
    model.save(output_dir + '/keras.h5')
    model_loaded: keras.Model = keras.models.load_model(output_dir + '/keras.h5')  # type: ignore

    proxy.save(output_dir + '/proxy.h5')
    proxy_loaded: keras.Model = keras.models.load_model(output_dir + '/proxy.h5')  # type: ignore
    for l1, l2 in zip(proxy.layers, proxy_loaded.layers):
        if not isinstance(l1, FixedPointQuantizer):
            continue
        assert l1.overrides == l2.overrides, f"Overrides mismatch for layer {l1.name}"

    assert tf.reduce_all(model(data) == model_loaded(data)), f"Model premdiction mismatch"
    assert tf.reduce_all(proxy(data) == proxy_loaded(data)), f"Proxy prediction mismatch"


def _run_model_proxy_match_test(model: keras.Model, proxy: keras.Model, data, cover_factor: float | None):
    nof_outputs = len(model.outputs)  # type: ignore
    if nof_outputs > 1:
        r_keras: list[np.ndarray] = [x.numpy() for x in model(data)]  # type: ignore
        r_proxy: list[np.ndarray] = [x.numpy() for x in proxy(data)]  # type: ignore
    else:
        r_keras: list[np.ndarray] = [model(data).numpy()]  # type: ignore
        r_proxy: list[np.ndarray] = [proxy(data).numpy()]  # type: ignore

    errors = []
    for i, (k, p) in enumerate(zip(r_keras, r_proxy)):
        mismatch_kp = k != p
        try:
            assert np.sum(mismatch_kp) == 0, f"Keras-Proxy mismatch for out {i}: {np.sum(np.any(mismatch_kp,axis=1))} out of {data.shape[0]} samples are different. Sample: {k[mismatch_kp].ravel()[:5]} vs {p[mismatch_kp].ravel()[:5]}"
        except AssertionError as e:
            errors.append(e)

    if cover_factor is None or cover_factor >= 1.0:
        if not len(errors) == 0:
            raise AssertionError('\n'.join([str(e) for e in errors]))
    else:
        if len(errors) == 0:
            warn(f"Keras-Proxy perfect match when overflow should happen: cover_factor={cover_factor}.")


def _run_gradient_test(model, data):
    with tf.GradientTape() as tape:
        out = model(data, training=True)
        if isinstance(out, list):
            loss = tf.reduce_mean([tf.reduce_sum(tf.abs(x)) for x in out])
        else:
            loss = tf.reduce_sum(tf.abs(out))

    trainable_weights = model.trainable_weights
    grad = tape.gradient(loss, trainable_weights)

    for w, g in zip(trainable_weights, grad):
        assert tf.reduce_any(g != 0), f"Gradient for {w.name} is zero"


def run_model_test(model: keras.Model, cover_factor: float | None, data, io_type: str, backend: str, dir: str, aggressive: bool, cond=None, skip_sl_test=False, test_gard=False):
    data_len = data.shape[0] if isinstance(data, np.ndarray) else data[0].shape[0]
    if test_gard:
        _run_gradient_test(model, data)
    if cover_factor is not None:
        trace_minmax(model, data, cover_factor=cover_factor, bsz=data_len)
    proxy = to_proxy_model(model, aggressive=aggressive, unary_lut_max_table_size=4096)
    try:
        if not skip_sl_test:
            _run_model_sl_test(model, proxy, data, dir)
        _run_model_proxy_match_test(model, proxy, data, cover_factor)
        _run_synth_match_test(proxy, data, io_type, backend, dir, cond=cond)
    except AssertionError as e:
        raise e
    except Warning as w:
        warn(w)
    else:
        shutil.rmtree(dir)


def set_seed(seed: int):
    seed = seed
    os.environ['RANDOM_SEED'] = f'{seed}'
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
