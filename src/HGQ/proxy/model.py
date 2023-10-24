import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.engine.node import Node
from keras.layers.convolutional.base_conv import Conv as BaseConv

from .fixed_point_quantizer import FixedPointQuantizer
from ..layers import HLayerBase, PLayerBase, Signature
from ..utils import warn


def get_all_nodes(model: keras.Model) -> set[Node]:
    nodes = set()
    for layer in model.layers:
        for node in layer._inbound_nodes:
            nodes.add(node)
        for node in layer._outbound_nodes:
            nodes.add(node)
    return nodes


def solve_dependencies(model: keras.Model):
    inp_tensors = model.inputs
    out_tensors = model.outputs
    inp_layers = [tensor._keras_history.layer for tensor in inp_tensors]  # type:ignore
    out_layers = [tensor._keras_history.layer for tensor in out_tensors]  # type:ignore

    input_nodes: list[Node] = [inp_layer._inbound_nodes[0] for inp_layer in inp_layers]
    output_nodes: list[Node] = [out_layer._inbound_nodes[0] for out_layer in out_layers]

    nodes = get_all_nodes(model)

    dependencies_list: list[tuple[keras.layers.Layer, set[Node], Node]] = []

    for node in nodes:
        if node.is_input:
            continue
        layer = node.layer
        requires = set(node.parent_nodes)
        provides = node
        dependencies_list.append((layer, requires, provides))
    return input_nodes, output_nodes, dependencies_list


def get_weight(layer: keras.layers.Layer, name: str):
    if '/' in name:
        name = name.split('/')[-1]
    if ':' not in name:
        name = f'{name}:0'
    for w in layer.weights:
        if name in w.name:
            return w
    return None


def copy_fused_weights(src: keras.layers.Layer, dst: keras.layers.Layer):
    if hasattr(dst, 'kernel'):
        if (k := getattr(src, 'fused_qkernel', None)) is not None:
            dst.kernel.assign(k)
        elif (k := getattr(src, 'qkernel', None)) is not None:
            dst.kernel.assign(k)
        else:
            dst.kernel.assign(src.kernel)
    if hasattr(dst, 'bias'):
        if (b := getattr(src, 'fused_qbias', None)) is not None:
            dst.bias.assign(b)
        elif (b := getattr(src, 'qbias', None)) is not None:
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


def to_keras_layers(layer: HLayerBase | PLayerBase, name: str) -> tuple[keras.layers.Layer, ...]:
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
    layer_cls = getattr(keras.layers, layer.__class__.__name__[1:])
    klayer = layer_cls.from_config(conf)
    klayer.build(layer.input_shape)
    copy_fused_weights(layer, klayer)
    if activation_layer is not None:
        return klayer, activation_layer
    else:
        return klayer,


def extract_quantizers(layer: HLayerBase | Signature, name: str, SAT='WRAP') -> tuple[FixedPointQuantizer, ...]:
    if isinstance(layer, Signature):
        return FixedPointQuantizer(layer.keep_negative, layer.bits, layer.int_bits, 'TRN', SAT),

    quantizer = layer.pre_activation_quantizer

    relu_act = layer._relu_act
    hls_config = None
    if layer._has_kernel:
        if isinstance(layer, keras.layers.Dense):
            multiplicity = np.prod(layer.kernel.shape) / layer.units
        elif isinstance(layer, BaseConv):
            multiplicity = np.prod(layer.kernel.shape) / layer.filters
        else:
            multiplicity = 1024
            warn(f'Unknown layer type {layer.__class__.__name__} to compute accumlation multiplicity for bitgrowth. If you are not using bitgrowth, ignore this warning.')
        hls_config = {'last_layer': {'weight_t': layer.ker_container, '_accum_multiplicity': multiplicity}}

    int_bits, fp_bits, kn = quantizer.get_bits_exact(pos_only=False)  # type: ignore
    if quantizer.rnd_strategy != 3 and not layer.can_bias_cover_rnd:
        RND = 'RND'
    else:
        RND = 'TRN'

    if not relu_act:
        k, b, i = kn, kn + int_bits + fp_bits, kn + int_bits
        return FixedPointQuantizer(k, b, i, RND, SAT, name=f'{name}_quantizer', hls_config=hls_config),

    k, i, f = tf.reduce_max(kn, keepdims=True), tf.reduce_max(int_bits, keepdims=True), tf.reduce_max(fp_bits, keepdims=True)
    k, b, i = k, k + i + f, k + i

    # If there is a rounding following the layer, keep one extra bit and do NOT round perserve bit accuracy.
    if RND != 'TRN':
        b += 1
    layer_quantizer = FixedPointQuantizer(k, b, i, 'TRN', SAT, name=f'{name}_quantizer', hls_config=hls_config)
    int_bits, fp_bits, kn = quantizer.get_bits_exact(pos_only=True)  # type: ignore
    if quantizer.rnd_strategy != 3 and not layer.can_bias_cover_rnd:
        RND = 'RND'
    else:
        RND = 'TRN'
    k, b, i = kn, kn + int_bits + fp_bits, kn + int_bits
    relu_quantizer = FixedPointQuantizer(k, b, i, RND, SAT, name=f'{name}_relu_quantizer')

    return layer_quantizer, relu_quantizer


def apply_proxy_layers(layer: keras.layers.Layer, tensor, namer: Namer | None = None, SAT='WRAP'):
    if namer is not None:
        name = namer.next_name(layer.name)
    else:
        name = layer.name
    proxy_layers = ()
    proxy_quantizer_layers = ()
    if hasattr(keras.layers, layer.__class__.__name__[1:]):
        proxy_layers = to_keras_layers(layer, name)
    if hasattr(layer, 'pre_activation_quantizer'):
        proxy_quantizer_layers = extract_quantizers(layer, name, SAT)
    if len(proxy_layers) > len(proxy_quantizer_layers) and isinstance(layer, HLayerBase):
        warn(f'Layer {layer.name} does not have a quantizer attached!')
    assert proxy_layers or proxy_quantizer_layers, f'Failed to convert layer {layer.name}: layer not mapped to anything.'
    for l1, l2 in zip(proxy_layers, proxy_quantizer_layers):
        tensor = l2(l1(tensor))
    if len(proxy_layers) < len(proxy_quantizer_layers):
        for l2 in proxy_quantizer_layers[len(proxy_layers):]:
            tensor = l2(tensor)
    else:
        for l1 in proxy_layers[len(proxy_quantizer_layers):]:
            tensor = l1(tensor)
    return tensor


def generate_proxy_model(model: keras.Model, aggressive: bool = True, accum_bits_bias=None):

    input_nodes, output_nodes, dependencies_list = solve_dependencies(model)

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
        if requires.issubset(satisfied):
            inps = [satisfied[node] for node in requires]
            if len(inps) == 1:
                inps = inps[0]
            out = apply_proxy_layers(layer, inps, namer=namer)
            satisfied[provides] = out
            if provides in output_nodes:
                outputs.append(out)
        else:
            i += 1
            if i == 65535:
                raise RuntimeError('Infinite loop detected.')
            dependencies_list.append((layer, requires, provides))

    return keras.Model(inputs=inputs, outputs=outputs)
