import tensorflow as tf
from tensorflow import keras

from . import layers
from .utils import warn


def copy_weights(target: keras.Model, source: keras.Model, force=False):
    for layer in source.layers:
        name = layer.name
        if hasattr(layer, 'kernel') or hasattr(layer, 'bias'):
            replica_layer: keras.layers.Layer = target.get_layer(name)
            if hasattr(layers, replica_layer.__class__.__name__):
                if not force:
                    raise ValueError(f'Layer {name} from the replica model is a custom layer, but the original model is not.')
                need_warn = True
            if hasattr(layer, 'kernel'):
                assert hasattr(replica_layer, 'kernel'), f'Layer {name} from the replica model does not have a kernel, but the original model does.'
                if hasattr(layer, 'fused_qkernel'):
                    replica_layer.kernel.assign(layer.fused_qkernel)  # type: ignore
                elif hasattr(layer, 'qkernel'):
                    replica_layer.kernel.assign(layer.qkernel)  # type: ignore
                else:
                    replica_layer.kernel.assign(layer.kernel)  # type: ignore
            if hasattr(layer, 'bias'):
                assert hasattr(replica_layer, 'bias'), f'Layer {name} from the replica model does not have a bias, but the original model does.'
                if hasattr(layer, 'fused_qbias'):
                    replica_layer.bias.assign(layer.fused_qbias)  # type: ignore
                elif hasattr(layer, 'qbias'):
                    replica_layer.bias.assign(layer.qbias)  # type: ignore
                else:
                    replica_layer.bias.assign(layer.bias)  # type: ignore
            continue

        if hasattr(layer, 'layers'):
            copy_weights(target.get_layer(name), layer)
            continue


def get_replica_config(model: tf.keras.Model):
    conf = model.get_config()

    layer_configs = {c['config']['name']: c for c in conf['layers']}
    names = list(layer_configs.keys())

    for i, name in enumerate(names):
        lc = layer_configs[name]

        cls_name = lc['class_name']

        if lc['class_name'] in ('HQuantize', 'Signature'):
            if 'inbound_nodes' in lc:
                inbound_node = lc['inbound_nodes']
                assert len(inbound_node) == 1 and \
                    len(inbound_node[0]) == 1 and \
                    len(inbound_node[0][0]) == 4 and \
                    inbound_node[0][0][0] in names, \
                    f"Layer {name}'s inbound node is not valid"
                inp_name = inbound_node[0][0][0]
            else:
                inp_name = names[i - 1]
            assert layer_configs[inp_name]['class_name'] == 'InputLayer'
            layer_configs[inp_name]['config']['name'] = name
            if 'name' in layer_configs[inp_name]:
                layer_configs[inp_name]['name'] = lc
            del layer_configs[name]

        if cls_name.startswith('H') or cls_name.startswith('P'):
            lc['class_name'] = cls_name[1:]  # map all H/P layers to their keras counterparts

    layer_configs = list(layer_configs.values())
    conf['layers'] = layer_configs
    return conf

_copy_weights = copy_weights

def create_replica(model: tf.keras.Model, copy_weights=True):
    """Create a replica model from a model with FHQ layers to be passed to hls4ml. The configs should be generated separately."""
    conf = get_replica_config(model)
    replica = model.__class__.from_config(conf)
    if copy_weights:
        _copy_weights(replica, model)
    return replica
