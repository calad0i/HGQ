import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .layers import HLayerBase
from .utils import tuple_to_apf, apf_to_tuple


def generate_mask(layer: HLayerBase):
    """
    Generate mask function to be patched in hls4ml project for a specific HGQ layer.

    If the layer has a relu activation, the mask function will be applied after the activation. Otherwise, it will be applied before the activation.

    """

    assert hasattr(layer, 'pre_activation_quantizer')
    is_relu = hasattr(layer, 'activation') and layer.activation is tf.keras.activations.relu

    int_bits, fp_bits, kn = layer.pre_activation_quantizer.get_bits(pos_only=is_relu)  # type: ignore
    int_bits, fp_bits, kn = int_bits.numpy().ravel(), fp_bits.numpy().ravel(), kn.numpy().ravel()

    mask = (int_bits + fp_bits) <= 0

    int_bits, fp_bits, kn = int_bits.astype(np.int8), fp_bits.astype(np.int8), kn.astype(np.int8)
    int_bits[mask] = 0
    fp_bits[mask] = 0

    assert np.all(
        kn[mask] == 0), f'Bit counting error at {layer.name}. This should never happen. Please try again with cuda disabled (2^13 or above will may in error when tensorflow is run with cuda). If the error persists, please report this issue.'

    if layer.pre_activation_quantizer.rnd_strategy != 3 and not hasattr(layer, 'bias'):
        # Use AP_RND iff not round to floor and no bias can be used to compensate
        RND = 'AP_RND'
    else:
        RND = 'AP_TRN'

    container = layer.act_container
    xfr = []
    for ii, (k, i, f) in enumerate(zip(kn, int_bits, fp_bits)):
        opr = tuple_to_apf((k, i, f), rnd=RND, keep_zeros=False)
        if opr == container:
            continue
        opr = f'    out[{ii}] = ap_{opr}(inp[{ii}]);' if opr != 'nuke' else f'    out[{ii}] = 0;'
        xfr.append(opr)

    name = layer.name
    if not xfr:
        return '', name

    if is_relu:
        name += '_relu'

    body = '    \n'.join(xfr)
    mask_fn = f'''
void {name}_mask(#dtypes_in_{name}# *inp, #dtypes_out_{name}# *out) {{
    #pragma HLS INLINE
    #pragma HLS PIPELINE
        
{body}
}}
    '''
    return mask_fn, name


def generate_mask_header(model: keras.Model, project_name: str):
    """ Generate mask header for a HGQ keras model"""
    template = f'''
#ifndef MASK_H_
#define MASK_H_

#include "{project_name}.h"
#include "parameters.h"

{{body}}

#endif
'''
    body, fn_names = [], []
    for layer in model.layers:
        if not isinstance(layer, HLayerBase):
            continue
        mask_fn, name = generate_mask(layer)
        if not mask_fn:
            continue
        body.append(mask_fn)
        fn_names.append(name)
    if not body:
        return '', []
    body = '\n\n'.join(body)
    header_str = template.format(body=body)
    return header_str, fn_names


def read_purge_auto_defined_vars(entry_func_path):
    """ Read entry function and purge/merge auto defined variables (auto& a = b;)"""
    with open(entry_func_path, 'r') as f:
        entry_func = f.read()

    m = re.findall(r'\n(\s*auto&\s+([\w_]+)\s*=\s*([\w_]+)\s*;\n?)', entry_func)

    for f, a, b in m:
        entry_func = entry_func.replace(f, '')
        entry_func = entry_func.replace(a, b)
    return entry_func


def patch_hls4ml_project(hls4ml_proj_path: str | Path, model, inline_everything=False, verbose=False):
    """ Patch hls4ml project with mask functions
    Args:
        hls4ml_proj_path (str|Path): Path to hls4ml project
        model: HGQ model
        inline_everything: Whether to inline everything. May help with latency
        verbose: Whether to print out patching progress
    """
    _hls4ml_proj_path = Path(hls4ml_proj_path)
    entry_func_path = _hls4ml_proj_path.glob('firmware/*.cpp').__next__()
    prj_name = entry_func_path.stem

    entry_func = read_purge_auto_defined_vars(entry_func_path)
    header, fn_names = generate_mask_header(model, prj_name)

    if inline_everything:
        entry_func = re.sub(r'#pragma HLS ([A-Z]{8})\s*\r?\n', '#pragma HLS \\1\n    #pragma HLS INLINE recursive\n\n', entry_func)
    if not header:
        with open(entry_func_path, 'w') as f:
            f.write(entry_func)
        return
    # insert mask header include clause
    entry_func = entry_func.replace('#include "parameters.h"', f'#include "parameters.h"\n\n#include "{prj_name}_mask.h"')

    # match to begining of nn operations

    for fn_name in sorted(fn_names, key=len, reverse=True):
        if verbose:
            print(f'Patching {fn_name}')
        # insert mask function call
        if 'inp_q' in fn_name:
            # input masking
            loc_key = f'// hls-fpga-machine-learning insert layers'
            operand_name = fn_name
        else:
            # general layers
            loc_key = f'// {fn_name}'
            operand_name = re.findall(rf'\([^,]+,([^,]+)[\w,_\s]*\); {loc_key}(?=\n)', entry_func)[0].strip()

        # get dtype. Only works for io_parallel
        dtype = re.findall(rf'([\w_]+_t) {operand_name}', entry_func)[0]
        # replace dtype placeholders
        header = header.replace(f'#dtypes_in_{fn_name}#', dtype).replace(f'#dtypes_out_{fn_name}#', dtype)

        # patch entry function
        entry_func = entry_func.replace(f'{loc_key}\n', f'{loc_key}\n    {fn_name}_mask({operand_name}, {operand_name});\n')

    with open(entry_func_path, 'w') as f:
        f.write(entry_func)
    with open(entry_func_path.parent / f'{prj_name}_mask.h', 'w') as f:
        f.write(header)
