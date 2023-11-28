import warnings

import numpy as np
import pytest
import tensorflow as tf
from helpers import get_test_dir, run_model_test, set_seed
from tensorflow import keras

from HGQ import get_default_paq_conf, set_default_paq_conf
from HGQ.layers import HDense, Signature


def create_model():
    # Here is an extremely terrible example of how to use Signature layer.
    # This is for test purpose only. Do NOT use it as a reference.

    dense = keras.layers.Dense(8)
    inp = keras.Input(shape=(8,))
    out = Signature(0, 4, 4)(inp)
    out = dense(out)
    model = keras.Model(inp, out)
    dense.kernel.assign(tf.constant(np.eye(8).astype(np.float32)))
    return model


def get_data(N: int, seed):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 2**4, (N, 8)).astype(np.float32)
    return data


@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("io_type", ['io_parallel', 'io_stream'])
@pytest.mark.parametrize("aggressive", [True, False])
@pytest.mark.parametrize("backend", ['vivado'])
@pytest.mark.parametrize("seed", [42])
def test_syn_hlayers(N: int, io_type: str, aggressive: bool, backend: str, seed: int):
    dir = get_test_dir()
    set_seed(seed)
    model = create_model()
    data = get_data(N, seed)

    # with pytest.warns(UserWarning, match=r'Layer batch_normalization(?:|_\d+) is unknown. Assuming infinite produced bitwidth') as record:
    run_model_test(model,
                   None,
                   data,
                   io_type,
                   backend,
                   dir,
                   aggressive,
                   test_gard=False,
                   )
    # assert record
