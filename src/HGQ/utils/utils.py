import re
from warnings import warn as _warn

import keras
import tensorflow as tf


@keras.utils.register_keras_serializable(package='HGQ')
class MonoL1:
    def __init__(self, l1=0.):
        assert l1 >= 0, f'l1 must be non-negative, got {l1}'
        self.l1 = l1

    def __call__(self, x):
        return tf.reduce_sum(x) * self.l1

    def get_config(self):
        return {'l1': self.l1}


DEFAULT_KERNEL_QUANTIZER_CONFIG = \
    dict(init_bw=2,
         skip_dims=None,
         rnd_strategy='standard_round',
         exact_q_value=True,
         dtype=None,
         bw_clip=(-23, 23),
         trainable=True,
         regularizer=MonoL1(1e-6),
         )


DEFAULT_PRE_ACTIVATION_QUANTIZER_CONFIG = \
    dict(init_bw=2,
         skip_dims=(0,),
         rnd_strategy='standard_round',  # 'auto': 'floor' for layer without bias or with 'relu' activation, 'standard_round' otherwise
         exact_q_value=True,
         dtype=None,
         bw_clip=(-23, 23),
         trainable=True,
         regularizer=MonoL1(1e-6),
         minmax_record=True
         )


def get_default_kq_conf():
    return DEFAULT_KERNEL_QUANTIZER_CONFIG.copy()


def get_default_paq_conf():
    return DEFAULT_PRE_ACTIVATION_QUANTIZER_CONFIG.copy()


def set_default_kq_conf(config):
    global DEFAULT_KERNEL_QUANTIZER_CONFIG
    assert isinstance(config, dict), f'config must be a dict, got {config.__class__.__name__}'
    assert set(config.keys()) == set(DEFAULT_KERNEL_QUANTIZER_CONFIG.keys()), \
        f'config must have keys {DEFAULT_KERNEL_QUANTIZER_CONFIG.keys()}, got {config.keys()}'
    DEFAULT_KERNEL_QUANTIZER_CONFIG = config


def set_default_paq_conf(config):
    global DEFAULT_PRE_ACTIVATION_QUANTIZER_CONFIG
    assert isinstance(config, dict), f'config must be a dict, got {config.__class__.__name__}'
    assert set(config.keys()) == set(DEFAULT_PRE_ACTIVATION_QUANTIZER_CONFIG.keys()), \
        f'config must have keys {DEFAULT_PRE_ACTIVATION_QUANTIZER_CONFIG.keys()}, got {config.keys()}'
    DEFAULT_PRE_ACTIVATION_QUANTIZER_CONFIG = config


strategy_dict = {
    'standard_round': 0,
    'stochastic_round': 1,
    'fast_uniform_noise_injection': 2,
    'floor': 3,
}

_apf_m = re.compile(r'\s*(?:|)(u?)fixed<\s*(\d+)\s*,\s*(-?\d+)[\s,_\w]*>\s*')


def apf_to_tuple(apf: str):
    """Convert fixed format to tuple of (keep_negative, int_bits, fp_bits)"""
    m = _apf_m.match(apf)
    assert m is not None, f'Unable to parse "{apf}". The format should be "()(u)fixed<m,n...>"'
    u, b, i = m.groups()
    b, i = int(b), int(i)
    kn = 0 if u else 1
    f = b - i
    i = i - kn
    return kn, i, f


def tuple_to_apf(t: tuple, rnd='TRN', warp='WARP', keep_zeros=True):
    """Convert tuple of (keep_negative, int_bits, fp_bits) to fixed format"""
    kn, i, f = t
    if not keep_zeros and i + f + kn <= 0:
        return 'nuke'
    if rnd != 'TRN' and warp == 'WARP':
        return f'{"u" if kn==0 else ""}fixed<{max(i+f+kn,1)},{i+kn},{rnd}>'
    if warp != 'WARP':
        return f'{"u" if kn==0 else ""}fixed<{max(i+f+kn,1)},{i+kn},{rnd},{warp}>'
    return f'{"u" if kn==0 else ""}fixed<{max(i+f+kn,1)},{i+kn}>'


def warn(msg: str, stacklevel=2):
    # print in yellow
    _warn(f'\033[93m{msg}\033[0m', stacklevel=stacklevel)
