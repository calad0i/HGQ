import numpy as np
from tensorflow import keras
import tensorflow as tf
from ..utils import warn

from .fixed_point_quantizer import FixedPointQuantizer
from ..utils import tuple_to_apf, apf_to_tuple
from ..quantizer import get_arr_bits
from keras.src.layers.convolutional.base_conv import Conv

STREAM = False

def get_input_kifs(layer:keras.layers.Layer) -> tuple[tuple[int,int,int]|np.ndarray,...]:
    """Get the input bitwidth of a layer, as a tuple of (k, i, f)."""
    parents:keras.layers.Layer|list[keras.layers.Layer] = layer._inbound_nodes[0].inbound_layers
    
    # if isinstance(parents, FixedPointQuantizer):
    #     # Terminal case
    #     return (parents.result_t_kif,)

    # As any layer that *changes* bitwidth will be followed by a quantizer, assume all layers we will meet here are "passive" layers.
    if isinstance(parents, keras.layers.Layer):
        return (get_produced_kif(parents),)
    else:
        # Passive layers with multiple inputs. Likely only Concatenate layer.
        return tuple(get_produced_kif(parent) for parent in parents)
            

def get_produced_kif(layer: keras.layers.Layer|FixedPointQuantizer) -> tuple[int, int, int]:
    """Get the produced bitwidth of a layer, as a tuple of (k, i, f)."""
    if isinstance(layer, FixedPointQuantizer):
        k,i,f = layer.result_t_kif
        return k,i,f
        
    kifs = get_input_kifs(layer)
    
    if len(kifs) == 1:
        k,i,f = kifs[0]
        if hasattr(layer, 'kernel'):
            w_k,w_i,w_f = get_arr_bits(layer.kernel.numpy())
            w_k,w_i,w_f = w_k.max(), w_i.max(), w_f.max()
            k,i,f = int(k or w_k), i+w_i, f+w_f
            if isinstance(layer, keras.layers.Dense):
                multiplicity = np.prod(layer.kernel.shape) / layer.units
            elif isinstance(layer, Conv):
                multiplicity = np.prod(layer.kernel.shape) / layer.filters
            else:
                multiplicity = np.prod(layer.kernel.shape)
                warn(f'Layer {layer.name} is not Dense or Conv. Multiplicity for accum size estimation is assumed to be maximum whole size of {multiplicity}.')
            i += int(np.ceil(np.log2(multiplicity)))
        if isinstance(layer, keras.layers.Activation):
            k,i,f, _, _ = derive_result_kifRS_from_next_quantizers(layer)
    else:
        if isinstance(layer, keras.layers.Add):
            k,i,f = np.max(kifs, axis=0)
            # being lazy here. But this will never overflow.
            i+=int(np.ceil(np.log2(len(kifs))))
        elif isinstance(layer, keras.layers.Concatenate):
            k,i,f = np.max(kifs, axis=0)
        else:
            # isinstance(layer, keras.layers.Multiply). Or, this should the most conservative assumption anyway.
            k,i,f = np.sum(kifs, axis=0)
            k = int(k!=0)
    return k,i,f


def get_requested_kif(layer:keras.layers.Layer|FixedPointQuantizer) -> tuple[int, int, int]:
    out_layers = [node.outbound_layer for node in layer._outbound_nodes]
    if len(out_layers) == 0:
        # Terminal case
        return (1,65535,65535)
    else:
        # Main case. 
        requested_kifs = [get_request_kif(out_layer) for out_layer in out_layers]
        k,i,f = np.max(requested_kifs, axis=0)
        return k,i,f


def get_request_kif(layer:keras.layers.Layer|FixedPointQuantizer) -> tuple[int, int, int]:
    """Get the requested bitwidth of a layer, as a tuple of (k, i, f)"""
    if isinstance(layer, FixedPointQuantizer):
        k,i,f = layer.result_t_kif
        rnd = layer.RND
        sat = layer.SAT
        if rnd.upper() == 'TRN':
            f+=0
        elif rnd.upper() == 'RND':
            f+=1
        else:
            f+=2
        if sat.upper() != 'WRAP':
            i = 65535
        return k, i, f
    
    if hasattr(layer, 'kernel'):
        k,i,f = 1,65535, 65535

    elif layer._outbound_nodes:
        # Main case. 
        out_layers:list[keras.layers.Layer] = [node.outbound_layer for node in layer._outbound_nodes]
        requested_kifs = [get_request_kif(out_layer) for out_layer in out_layers]
        k,i,f = np.max(requested_kifs, axis=0)
    else:
        k,i,f = 1,65535, 65535
    return k,i,f


def merge_precision(available:tuple[int,int,int], request:tuple[int,int,int]):
    """Given available precision and the maximum precision can be accepted by downstream, return the precision should be allocated for the data path."""
    k0,i0,f0 = available
    k1,i1,f1 = request
    # assert k0==k1, 'A valid proxy model should have both producer and consumer both using signed or unsigned bit.'
    k = k0
    i = min(i0,i1)
    f = min(f0,f1)
    return k,i,f


def result_kifRS_layer_with_fusible_quantizer(layer:keras.layers.Layer):
    out_nodes = layer._outbound_nodes
    if len(out_nodes) != 1:
        return
    out_layer = out_nodes[0].outbound_layer
    if not isinstance(out_layer, FixedPointQuantizer):
        return
    if not out_layer.fusible:
        return
    k,i,f = out_layer.result_t_kif
    SAT = out_layer.SAT
    RND = out_layer.RND
    return k,i,f, RND, SAT


def derive_result_kifRS_from_next_quantizers(layer:keras.layers.Layer) -> tuple[int, int, int, str, str]:
    # This is a special case for input layer and activation layer. Can also be used for layers with accum in stream setting.
    Ks, Is, Fs, RNDs, SATs = [], [], [], [], []
    out_layers:list[FixedPointQuantizer] = [node.outbound_layer for node in layer._outbound_nodes]
    kifRSs = np.array(list(map(lambda x: (*x.result_t_kif, x.RND, x.SAT), out_layers)), dtype=object)
    Ks, Is, Fs, RNDs, SATs = kifRSs.T
    assert np.all(SATs == SATs[0]), f'Input layer {layer.name} has different SATs. This is not supported.'
    SAT = SATs[0].upper()
    if SAT == 'WRAP_SM':
        warn('WRAP_SM for input layer may lead bitmismatch between proxy and hls4ml due to overflow on th first layer.')
    k = np.max(Ks)
    i = np.max(Is)
    f = np.max(Fs+2*(RNDs != 'TRN')-1*((RNDs == 'RND')))
    return k,i,f, 'TRN', SAT


def get_result_kifRS(layer:keras.layers.Layer) -> tuple[int, int, int, str, str]:
    result_t = result_kifRS_layer_with_fusible_quantizer(layer)
    if result_t is not None:
        return result_t
    if isinstance(layer, keras.layers.InputLayer):
        return derive_result_kifRS_from_next_quantizers(layer)
    if STREAM and hasattr(layer, 'kernel'):
        return derive_result_kifRS_from_next_quantizers(layer)
    
    produced_kif = get_produced_kif(layer)
    requested_kif = get_requested_kif(layer)
    k,i,f = merge_precision(produced_kif, requested_kif)
    return k,i,f, 'TRN', 'WRAP'


def get_config_wight_accum_result_bias(layer:keras.layers.Layer, bias_accum_fp:None|int=None):
    assert hasattr(layer, 'kernel'), f'Layer {layer.name} does not have kernel.'
    r_k,r_i,r_f, RND, SAT = get_result_kifRS(layer)
    k_k,k_i,k_f = get_arr_bits(layer.kernel.numpy())
    k_k,k_i,k_f = k_k.max(), k_i.max(), k_f.max()
    weight_t = tuple_to_apf((k_k,k_i,k_f))
    result_t = tuple_to_apf((r_k,r_i,r_f), RND, SAT)
    if bias_accum_fp is None:
        a_f = r_f+k_f
    else:
        a_f = r_f+bias_accum_fp
    if SAT.upper() == 'WRAP':
        accum_t = tuple_to_apf((r_k,r_i,a_f))
    else:
        k,i,f = (get_produced_kif(layer))
        if bias_accum_fp is not None:
            warn(f'Layer {layer.name} has SAT={SAT}, and has bias_accum_fp set. Proceed only if you know what you are doing.')
            f = bias_accum_fp + r_f
        accum_t = tuple_to_apf((k,i,f))
    bias_t = accum_t
    return weight_t, accum_t, result_t, bias_t


def get_config_table_tablesize_result(layer:keras.layers.Activation):
    assert isinstance(layer, keras.layers.Activation), f'Layer {layer.name} is not an activation layer.'
    k,i,f, RND, SAT = get_result_kifRS(layer)
    result_t = tuple_to_apf((k,i,f), RND, SAT)
    table_t = result_t
    if layer.activation in (keras.activations.tanh, keras.activations.sigmoid):
        result_t = tuple_to_apf((k,i,f), 'TRN', 'WRAP')  # These activations are purely LUT based, this rounding/saturating in the table only would be enough.
    i_kifs = get_input_kifs(layer)
    if len(i_kifs) > 1:
        warn(f'Activation layer {layer.name} has more than one input. Did you just make a activation with multiple inputs? table_size set in this way may make no sense. Proceed only if you know what you are doing.')
    i_k,i_i,i_f = np.max(i_kifs, axis=0)
    table_size = int(2**(i_k+i_i+i_f)) # ...the ideal case. Will be the case if we have universal LUT-based activation.
    if layer.activation is keras.activations.tanh:
        table_size = int(8 / 2.**-i_f)  # LUT Range hardcoded to -4 ~ 4, match #fractional bits
    elif layer.activation is tf.keras.activations.sigmoid:
        table_size = int(16 / 2.**-i_f)  # LUT Range hardcoded to -8 ~ 8, match #fractional bits
    return table_t, table_size, result_t


def get_config(layer:keras.layers.Layer, bias_accum_fp:None|int=None):
    if hasattr(layer, 'kernel'):
        weight_t, accum_t, result_t, bias_t = get_config_wight_accum_result_bias(layer, bias_accum_fp)
        conf = {'weight_t':weight_t, 'accum_t':accum_t, 'result_t':result_t, 'bias_t':bias_t}
    elif isinstance(layer, keras.layers.Activation):
        table_t, table_size, result_t = get_config_table_tablesize_result(layer)
        conf = {'table_t':table_t, 'table_size':table_size, 'result_t':result_t}
    else:
        k,i,f,RND,SAT = get_result_kifRS(layer)
        result_t = tuple_to_apf((k,i,f), RND, SAT)
        conf = {'result_t':result_t}
    return conf


def _get_next_quantizer(layer:keras.layers.Layer):
    out_layers = [node.outbound_layer for node in layer._outbound_nodes]
    for out_layer in out_layers:
        if isinstance(out_layer, FixedPointQuantizer):
            return out_layer
        r = _get_next_quantizer(out_layer)
        if r is not None:
            return r
    return None

def _get_last_quantizer(layer:keras.layers.Layer): 
    assert len(layer._inbound_nodes) <= 1, f'Layer {layer.name} has more than one inbound nodes. This is not supported.'
    in_layers = layer._inbound_nodes[0].inbound_layers
    if not isinstance(in_layers, list):
        in_layers = [in_layers]
    for in_layer in in_layers:
        if isinstance(in_layer, FixedPointQuantizer):
            return in_layer
        r = _get_last_quantizer(in_layer)
        if r is not None:
            return r
    return None

def get_whatever_quantizer(layer:keras.layers.Layer):
    if isinstance(layer, FixedPointQuantizer):
        return layer
    if isinstance(layer, keras.layers.InputLayer):
        pass      
    q = _get_next_quantizer(layer) or _get_last_quantizer(layer)
    assert q is not None, f'Layer {layer.name} has no quantizer before or after it. Did you create a valid proxy model?'
    return q

def register_qconf(layer:keras.layers.Layer):
    q = get_whatever_quantizer(layer)
    conf = get_config(layer)
    overrides = q.overrides or {}
    overrides['layers'].setdefault(layer.name, {}).update(conf)