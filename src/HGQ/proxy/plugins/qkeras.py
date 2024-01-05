import abc

import keras
import numpy as np
import qkeras
from keras import backend as K
from qkeras import quantizers
from qkeras.quantizers import BaseQuantizer
from qkeras.utils import _add_supported_quantized_objects
from tensorflow import keras

from HGQ.proxy.fixed_point_quantizer import FixedPointQuantizer
from HGQ.utils import warn

from ..convert import ProxyLayerXFormer
from ..precision_derivation import get_produced_kif

qkeras_objects = {}
_add_supported_quantized_objects(qkeras_objects)

qkeras_layers = {v for v in qkeras_objects.values() if issubclass(v, keras.layers.Layer)}
qkeras_quantizers = {v for v in qkeras_objects.values() if issubclass(v, BaseQuantizer)}
real_qkeras_quantizers = {
    quantizers.quantized_bits,
    quantizers.quantized_relu,
    quantizers.stochastic_binary,
    quantizers.stochastic_ternary,
    quantizers.ternary,
    quantizers.binary,
    quantizers.quantized_po2,
    quantizers.quantized_relu_po2,
}

from functools import singledispatch


@singledispatch
def qkeras_quantizer_to_layers(quantizer, SAT) -> tuple[keras.layers.Layer, ...]:
    """
    Map qkeras quantizer to a tuple of equivalent proxy layers.
        e.g.: quantized_bits -> (fixed,)
              quantized_relu -> (fixed, relu, fixed)
    """
    raise TypeError(f"Unknown quantizer type {type(quantizer)}")


@qkeras_quantizer_to_layers.register
def _(quantizer: quantizers.quantized_bits, SAT):
    assert quantizer.alpha == 1, f"alpha != 1 is not currently supported, got {quantizer.alpha}."
    k = quantizer.keep_negative
    i = quantizer.integer + k
    b = quantizer.bits
    SYM = bool(quantizer.symmetric)
    if SYM:
        if SAT != 'SAT':
            warn(f'Symmetric quantizer is only possible with SAT=SAT_SYM, but got SAT={SAT}. Enforcing SAT=SAT.')
        SAT = 'SAT_SYM'
    if k == b == 1:
        warn(f"Qkeras's implementation for quantizers with keep_negative = bits = 1 is flawed as of 2023/11. If you believe they have fixed it, please ignore this warning.")
    if quantizer.symmetric != 0 and quantizer.symmetric != 1:
        raise ValueError(f"Expecting symmetric to be 0 or 1 for quantized_bits, got {quantizer.symmetric}.")

    return FixedPointQuantizer(k, b, i, 'RND_CONV', SAT),


@qkeras_quantizer_to_layers.register
def _(quantizer: quantizers.quantized_relu, SAT):
    i = quantizer.integer
    b = quantizer.bits
    q0 = FixedPointQuantizer(1, b + 1, i + 1, 'RND_CONV', SAT)
    q1 = FixedPointQuantizer(0, b, i, 'RND_CONV', 'WRAP')
    relu = keras.layers.Activation('relu')
    return q0, relu, q1


@qkeras_quantizer_to_layers.register
def _(quantizer: quantizers.stochastic_binary | quantizers.binary, SAT):
    if quantizer.use_01:
        raise NotImplementedError("No support is currently available for use_01=True in stochastic_binary or binary on hls4ml.")
        return FixedPointQuantizer(0, 1, 1, 'RND', 'SAT'),
    else:
        return qkeras.QActivation(qkeras.binary()), FixedPointQuantizer(1, 2, 2, 'RND', 'WRAP')


@qkeras_quantizer_to_layers.register
def _(quantizer: quantizers.stochastic_ternary | quantizers.ternary):
    return FixedPointQuantizer(1, 2, 1, 'RND', 'SAT')


@qkeras_quantizer_to_layers.register
def _(quantizer: quantizers.quantized_po2 | quantizers.quantized_relu_po2):
    raise ValueError("quantized_po2 family cannot be implemented in hls4ml as activation.")


def qlayer_to_keras_layer(layer: keras.layers.Layer) -> keras.layers.Layer | None:
    base_cls_name = layer.__class__.__name__[1:]
    if not hasattr(keras.layers, base_cls_name):
        raise ValueError(f"Cannot find corresponding keras layer for {layer.__class__.__name__}")
    base_cls = getattr(keras.layers, base_cls_name)
    conf = layer.get_config()

    # For some ridiculous reason, QAactivation's activation is called quantizer
    # Yet, all other layers are call their activation activation.
    if isinstance(layer, (qkeras.QActivation, keras.layers.Activation)):
        quantizer = layer.quantizer if isinstance(layer, qkeras.QActivation) else layer.activation
        if quantizer.__class__ in qkeras_quantizers:
            return None

    blacklist = ('quantize', 'kernel_range', 'bias_range', 'constraint', 'activation', 'initializer')
    for k in list(conf.keys()):
        for k_rm in blacklist:
            if k_rm in k:
                del conf[k]
                continue

    klayer = base_cls(**conf)
    klayer.trainable = False
    klayer.build(layer.input_shape)
    if hasattr(layer, 'kernel'):
        q = layer.kernel_quantizer_internal or (lambda x: x)
        klayer.kernel.assign(q(layer.kernel))
    if hasattr(layer, 'bias'):
        q = layer.bias_quantizer_internal or (lambda x: x)
        klayer.bias.assign(q(layer.bias))
    return klayer


def qlayer_to_proxy_layer(self: ProxyLayerXFormer, layer: keras.layers.Layer) -> tuple[keras.layers.Layer, ...]:

    SAT = self.SAT
    klayer = qlayer_to_keras_layer(layer)
    if not hasattr(layer, 'activation'):
        assert klayer is not None
        return klayer

    activation = layer.quantizer if hasattr(layer, 'quantizer') else layer.activation
    act_layers = qkeras_quantizer_to_layers(activation, SAT)

    input_shape = layer.input_shape
    if isinstance(input_shape, tuple):
        inputs = keras.layers.Input(shape=input_shape[1:])
    else:
        inputs = [keras.layers.Input(shape=shape[1:]) for shape in input_shape]

    out = inputs
    if klayer is not None:
        out = klayer(out)
    for layer in act_layers:
        out = layer(out)
    return keras.Model(inputs, out)


class QKerasBaseLayer(metaclass=abc.ABCMeta):
    pass


for layer_cls in qkeras_layers:
    QKerasBaseLayer.register(layer_cls)


def init():
    ProxyLayerXFormer.__call__.register(QKerasBaseLayer, qlayer_to_proxy_layer)
    get_produced_kif.register(qkeras.QActivation, get_produced_kif.registry[keras.layers.Activation])
