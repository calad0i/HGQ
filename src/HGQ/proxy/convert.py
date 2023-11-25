from functools import singledispatch

import numpy as np
import tensorflow as tf
from keras.src.engine.keras_tensor import KerasTensor
from keras.src.engine.node import Node
from tensorflow import keras

from ..layers import HLayerBase, HQuantize, PDropout, PLayerBase, Signature
from ..layers.base import ABSBaseLayer
from ..utils import warn
from .fixed_point_quantizer import FixedPointQuantizer
from .precision_derivation import register_qconf


def get_all_nodes(model: keras.Model) -> set[Node]:
    """Get all nodes in the model as a set."""
    nodes = set()
    for layer in model.layers:
        for node in layer._inbound_nodes:
            nodes.add(node)
        for node in layer._outbound_nodes:
            nodes.add(node)
    return nodes


def solve_dependencies(model: keras.Model):
    """Given a keras model, return the input nodes, output nodes and a list of (layer, requires, provides) tuples. Requires is a list of nodes that are parents of the layer, provides is the node that is the output of the layer."""
    inp_tensors: list[KerasTensor] = model.inputs  # type:ignore
    out_tensors: list[KerasTensor] = model.outputs  # type:ignore

    input_nodes: list[Node] = [t.node for t in inp_tensors]  # type:ignore
    output_nodes: list[Node] = [t.node for t in out_tensors]  # type:ignore

    nodes = get_all_nodes(model)

    dependencies_list: list[tuple[keras.layers.Layer, list[Node], Node]] = []
    "List of (layer, requires, provides) tuples; requires is a list of nodes, provides is a single node."

    for node in nodes:
        if node.is_input:
            continue
        layer = node.layer
        requires = list(node.parent_nodes)
        provides = node
        dependencies_list.append((layer, requires, provides))
    return input_nodes, output_nodes, dependencies_list


def get_weight(layer: keras.layers.Layer, name: str):
    """Given a layer and a weight name, return the weight. The weight name may or may not contain the layer name. If the number index is missing, it is assumed to be 0."""
    if '/' in name:
        name = name.split('/')[-1]
    if ':' not in name:
        name = f'{name}:0'
    for w in layer.weights:
        if name in w.name:
            return w
    return None


def copy_fused_weights(src: keras.layers.Layer, dst: keras.layers.Layer):
    """For HGQ layers, some layers may have different fused weights for kernel and bias (Processed weights are deployment). This function copies the fused kernel and bias to the keras proxy."""
    if hasattr(dst, 'kernel'):
        if (k := getattr(src, 'fused_qkernel', None)) is not None:
            dst.kernel.assign(k)
        else:
            dst.kernel.assign(src.kernel)
    if hasattr(dst, 'bias'):
        if (b := getattr(src, 'fused_qbias', None)) is not None:
            dst.bias.assign(b)
        elif (b := getattr(src, 'bias', None)) is not None:
            dst.bias.assign(b)
        else:
            warn(f'No bias found for layer {src.name}.')
    for dst_w in dst.weights:
        if 'kernel:0' in dst_w.name or 'bias:0' in dst_w.name:
            continue
        src_w = get_weight(src, dst_w.name)
        if src_w is not None:
            dst_w.assign(src_w)
        else:
            warn(f'No weight found for layer {src.name} corresponding to {dst_w.name}.')


class Namer:
    """Helper class to generate unique names for layers, if one being used multiple times."""

    def __init__(self):
        self.used_names: set[str] = set()

    def next_name(self, name: str):
        i = 0
        if name in self.used_names:
            while f'{name}_{i}' in self.used_names:
                i += 1
            name = f'{name}_{i}'
        self.used_names.add(name)
        return name


def extract_keras_layers(layer: HLayerBase | PLayerBase, name: str) -> tuple[keras.layers.Layer, ...]:
    """Given a HGQ layer, return a tuple of keras layers in corresponding order. The tuple may be empty if the layer is a quantizer, or containing only the corresponding layer if the layer is an activation or does not have a non-linear activation, or containing the corresponding layer and the activation layer otherwise.

    Example:
        HQuantize -> ()
        HDense[linaer] -> (Dense,)
        HDense[relu] -> (Dense, Activation)
    """

    if isinstance(layer, (HQuantize, Signature)):
        return tuple()

    if hasattr(layer, 'get_keras_config'):
        conf = layer.get_keras_config()
    else:
        conf = layer.get_config()
    conf['name'] = name
    activation_layer = None

    if not isinstance(layer, keras.layers.Activation):
        if 'activation' in conf and conf['activation'] != 'linear':
            activation = conf['activation']
            if isinstance(activation, str):
                activation_name = activation
            else:
                activation_name = activation.__name__
            activation_layer = keras.layers.Activation(activation, name=f'{name}_{activation_name}')
            conf['activation'] = 'linear'
    cls_name = layer.__class__.__name__[1:]
    if hasattr(keras.layers, cls_name):
        layer_cls = getattr(keras.layers, cls_name)
    elif hasattr(keras.layers, cls_name.replace('BatchNorm', '')):
        layer_cls = getattr(keras.layers, cls_name.replace('BatchNorm', ''))
    else:
        raise RuntimeError(f'Unknown layer type {layer.__class__.__name__}: no corresponding keras layer found.')
    klayer = layer_cls.from_config(conf)
    klayer.build(layer.input_shape)
    copy_fused_weights(layer, klayer)
    if activation_layer is not None:
        return klayer, activation_layer
    else:
        return klayer,


def extract_quantizers(layer: HLayerBase | Signature, name: str, SAT='WRAP') -> tuple[FixedPointQuantizer, ...]:
    """Given a HGQ layer, return a tuple of quantizers that are used in the layer."""
    if isinstance(layer, Signature):
        return FixedPointQuantizer(layer.keep_negative, layer.bits, layer.int_bits, 'TRN', SAT),

    quantizer = layer.paq
    if quantizer.rnd_strategy != 3 and not layer.can_bias_cover_rnd:
        RND = 'RND'
    else:
        RND = 'TRN'

    relu_act = layer._relu_act
    overriddes = None
    if layer._has_kernel:
        if hasattr(layer, 'parallel_factor'):
            parallel_factor = int(layer.parallel_factor)
            overriddes = {'layers': {name: {'parallelization_factor': parallel_factor}}}

    int_bits, fp_bits, kn = quantizer.get_bits_exact(pos_only=False)  # type: ignore

    if not relu_act:
        k, b, i = kn, kn + int_bits + fp_bits, kn + int_bits
        return FixedPointQuantizer(k, b, i, RND, SAT, name=f'{name}_quantizer', overrides=overriddes),

    mask = int_bits + fp_bits + kn > 0

    r_int_bits, r_fp_bits, rk = quantizer.get_bits_exact(pos_only=True)
    rk, rb, ri = rk, r_int_bits + r_fp_bits, r_int_bits
    relu_quantizer = FixedPointQuantizer(rk, rb, ri, RND, SAT, name=f'{name}_relu_quantizer')

    if isinstance(layer, keras.layers.Activation):
        return relu_quantizer,

    k = tf.reduce_max(kn[mask], keepdims=True)
    i = tf.reduce_max(int_bits[mask], keepdims=True)
    f = tf.reduce_max(fp_bits[mask], keepdims=True)
    k, b, i = k, k + i + f, k + i

    # If there is a rounding following the layer, keep one or two extra bit and do NOT round perserve bit accuracy.
    if RND == 'RND':
        b += 1
    elif RND != 'TRN':
        b += 2

    layer_quantizer = FixedPointQuantizer(k, b, i, 'TRN', SAT, name=f'{name}_quantizer', overrides=overriddes)

    return layer_quantizer, relu_quantizer


@singledispatch
def to_proxy_layers(layer, name, SAT: str) -> tuple[keras.layers.Layer, ...]:
    """Given a layer, return a tuple of keras layers and quantizers that are equivalent to the layer when applied in order. (When it doesn't overflow, and up to fp precision)"""
    if hasattr(keras.layers, layer.__class__.__name__):
        # Is already vanilla keras layer
        new_layer = layer.__class__.from_config(layer.get_config())
        new_layer.build(layer.input_shape)
        for w1, w2 in zip(new_layer.weights, layer.weights):
            w1.assign(w2)
        return new_layer,

    raise TypeError(f'No matching overload for layer type {type(layer)}. Signatures available: {to_proxy_layers.registry.keys()}')


@to_proxy_layers.register
def _(layer: ABSBaseLayer, name: str, SAT: str):
    proxy_quantizer_layers = ()
    layers = []
    proxy_layers = list(extract_keras_layers(layer, name))

    if hasattr(layer, 'paq'):
        proxy_quantizer_layers = list(extract_quantizers(layer, name, SAT))
    if len(proxy_layers) > len(proxy_quantizer_layers) and isinstance(layer, HLayerBase):
        warn(f'Layer {layer.name} does not have a quantizer attached!')

    while proxy_layers or proxy_quantizer_layers:
        if proxy_layers:
            layers.append(proxy_layers.pop(0))
        if proxy_quantizer_layers:
            layers.append(proxy_quantizer_layers.pop(0))

    assert layers, f'Failed to convert layer {layer.name}: layer not mapped to anything.'
    return tuple(layers)


SKIP_LAYERS = (PDropout,)


def apply_proxy_layers(layer: keras.layers.Layer, tensor, namer: Namer | None = None, SAT='WRAP'):
    """Given a HGQ-competible layer and a tensor, return the output of the layer when applied to the tensor. Used in builing the proxy model."""
    if isinstance(layer, SKIP_LAYERS):
        return tensor
    if namer is not None:
        name = namer.next_name(layer.name)
    else:
        name = layer.name
    for l in to_proxy_layers(layer, name, SAT):
        tensor = l(tensor)
    return tensor


def to_proxy_model(model: keras.Model, aggressive: bool = True, accum_fp_max_offset: int | None = None):
    """Given a HGQ model, return a hls4ml-ready keras model.

    Args:
        model: The HGQ model to be converted.

        aggressive (default: True): If True, use WRAP overflow mode. Sigificant performance degradation may occur if overflow occurs, but latency may be reduced. If False, use SAT overflow mode. Performance is more stable when it overflows, but latency may be increased.

        accum_fp_max_offset (default: None): If not None, autoset accumlator such that the model is bit accurate (when no overflow occurs and up to fp precision). If set, use the specified number of floating bits plus result float bits as accumlator float bits. May improve latency in some rare cases, not recommended in general.

    """
    input_nodes, output_nodes, dependencies_list = solve_dependencies(model)

    if accum_fp_max_offset is not None and not aggressive:
        warn('You are using bitgrowth (aggressive=False) together with bias_accum_bits set. This is not recommended. If you are sure what you are doing, ignore this warning.')
    if accum_fp_max_offset is not None and accum_fp_max_offset < 0:
        warn('You are using a negative value for bias_accum_bits. Please make sure you know what you are doing.')

    nof_output = len(output_nodes)
    inputs = [keras.layers.Input(shape=node.input_shapes[0][1:]) for node in input_nodes]
    satisfied = {node: tensor for node, tensor in zip(input_nodes, inputs)}
    outputs = []
    namer = Namer()
    i = 0
    if aggressive:
        SAT = 'WRAP'
    else:
        SAT = 'SAT'
    while dependencies_list and not len(outputs) == nof_output:
        layer, requires, provides = dependencies_list.pop(0)
        if set(requires).issubset(satisfied):
            inps = [satisfied[node] for node in requires]
            if len(inps) == 1:
                inps = inps[0]
            out = apply_proxy_layers(layer, inps, namer=namer, SAT=SAT)
            satisfied[provides] = out
            if provides in output_nodes:
                outputs.append(out)
        else:
            i += 1
            if i == 65535:
                raise RuntimeError('Infinite loop detected.')
            dependencies_list.append((layer, requires, provides))

    if len(outputs) == 1:
        outputs = outputs[0]
    if len(inputs) == 1:
        inputs = inputs[0]
    proxy = keras.Model(inputs=inputs, outputs=outputs)
    for layer in proxy.layers:
        register_qconf(layer, accum_fp_max_offset=accum_fp_max_offset)
    return proxy
