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

            if hasattr(layer, 'kernel') and layer.kernel is not None:
                assert hasattr(replica_layer, 'kernel'), f'Layer {name} from the replica model does not have a kernel, but the original model does.'
                if hasattr(layer, 'fused_qkernel'):
                    replica_layer.kernel.assign(layer.fused_qkernel)  # type: ignore
                elif hasattr(layer, 'qkernel'):
                    replica_layer.kernel.assign(layer.qkernel)  # type: ignore
                else:
                    replica_layer.kernel.assign(layer.kernel)  # type: ignore
            if hasattr(layer, 'bias') and layer.bias is not None:
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

    replaced_names = {}
    for i, name in enumerate(names):
        lc = layer_configs[name]

        cls_name = lc['class_name']

        if lc['class_name'] in ('HQuantize', 'Signature'):
            if 'inbound_nodes' in lc:  # functional model
                inbound_node = lc['inbound_nodes']
                assert len(inbound_node) == 1 and \
                    len(inbound_node[0]) == 1 and \
                    len(inbound_node[0][0]) == 4 and \
                    inbound_node[0][0][0] in names, \
                    f"Layer {name}'s inbound node is not valid"
                inp_name = inbound_node[0][0][0]
            else:
                inp_name = names[i - 1]  # sequential model
            assert layer_configs[inp_name]['class_name'] == 'InputLayer'
            layer_configs[inp_name]['config']['name'] = name  # rename the input layer to the name of the quantize layer
            if 'name' in layer_configs[inp_name]:
                layer_configs[inp_name]['name'] = name
            del layer_configs[name]
            replaced_names[inp_name] = name
            continue

        if cls_name.startswith('H') or cls_name.startswith('P'):
            lc['class_name'] = cls_name[1:]  # map all H/P layers to their keras counterparts

        for k, v in lc['config'].items():  # purge all constraint/regularization configs, as they are not relevant during the inference time
            if 'constraint' in k or 'regularizer' in k:
                lc['config'][k] = None

    layer_configs = list(layer_configs.values())
    conf['layers'] = layer_configs

    if 'input_layers' in conf:
        for v in conf['input_layers']:
            assert v[0] in replaced_names
            v[0] = replaced_names[v[0]]

    return conf


_copy_weights = copy_weights


def create_replica(model: tf.keras.Model, copy_weights=True):
    """Create a replica model from a model with HGQ layers to be passed to hls4ml. The configs should be generated separately."""
    conf = get_replica_config(model)
    replica = model.__class__.from_config(conf)
    if copy_weights:
        _copy_weights(replica, model)
    return replica
