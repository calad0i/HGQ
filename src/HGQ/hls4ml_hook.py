import hls4ml
from hls4ml.model import ModelGraph
from functools import wraps
from tensorflow import keras
from typing import Any
import types

from .layers import HLayerBase, PLayerBase
from .layers import Signature
from .utils import apf_to_tuple, tuple_to_apf
from .replica import create_replica
from .hls4ml_prj_patch import patch_hls4ml_project


def hook_compile(model_hls):
    def _compile(self):
        original_write = self.write
        nothing = lambda *args, **kwargs: None
        setattr(self, 'write', nothing)
        self.compile()
        setattr(self, 'write', original_write)
    model_hls.no_write_compile = types.MethodType(_compile, model_hls)


def hook_trace(model_hls):
    def _trace(self, *args, **kwargs):
        original_write = self.write
        nothing = lambda *args, **kwargs: None
        setattr(self, 'write', nothing)
        r = self.trace(*args, **kwargs)
        setattr(self, 'write', original_write)
        return r
    model_hls.no_write_trace = types.MethodType(_trace, model_hls)


original_convert_from_keras_model = hls4ml.converters.convert_from_keras_model


@wraps(hls4ml.converters.convert_from_keras_model)
def _convert_from_keras_model(
    model: keras.Model,
    output_dir: str = 'my-hls-test',
    project_name: str = 'myproject',
    input_data_tb: Any | None = None,
    output_data_tb: Any | None = None,
    backend: str = 'Vivado',
    hls_config: Any | None = None,
    **kwargs: Any
) -> ModelGraph:
    model_hls = original_convert_from_keras_model(model, output_dir, project_name, input_data_tb, output_data_tb, backend, hls_config, **kwargs)
    hook_compile(model_hls)
    hook_trace(model_hls)
    return model_hls


def hook_converter():
    setattr(hls4ml.converters, 'convert_from_keras_model', _convert_from_keras_model)


def update_layerconf(model: keras.Model, conf: dict[str, dict], bias_accum=None):
    c = conf['LayerName']
    for layer in model.layers:
        # print(layer.name)

        if not isinstance(layer, HLayerBase) and not isinstance(layer, PLayerBase):
            continue
        cn = c[layer.name]['Precision']
        result = layer.result_container
        k, i, f = apf_to_tuple(result)
        cn['result'] = result

        if not isinstance(layer, HLayerBase):
            continue

        if hasattr(layer, 'parallel_factor'):
            c[layer.name]['ParallelizationFactor'] = int(layer.parallel_factor)

        if layer._has_kernel or layer._has_bias:
            if bias_accum is not None:
                accum_fp = f + bias_accum
            else:
                accum_fp = layer.max_accum_fp_bits
            accum = tuple_to_apf((k, i, accum_fp))
            cn['accum'] = accum

            if layer._has_kernel:
                kernel = layer.ker_container
                cn['weight'] = kernel

            if layer._has_bias:
                cn['bias'] = accum

        if not hasattr(layer, '_relu_act'):
            continue

        if not layer._relu_act:
            for name in c:
                if name.startswith(f'{layer.name}_'):
                    c[name]['Precision']['result'] = result
                    break
            continue

        c[f'{layer.name}_relu']['Precision']['result'] = layer.act_container


def convert_from_hgq_model(
    model: keras.Model,
    output_dir: str = 'my-hls-test',
    project_name: str = 'myproject',
    input_data_tb: Any | None = None,
    output_data_tb: Any | None = None,
    backend: str = 'Vivado',
    hls_config: Any | None = None,
    bias_accum=None,
    inline_everything=False,
    io_type='io_parallel',
    **kwargs: Any
) -> ModelGraph:
    """Converts a HGQ Keras model to hls4ml model.

    Args:
        model (keras.Model): HGQ Keras model. That is, all layers should be inherited from HLayerBase or PLayerBase.
        output_dir (str, optional): Output directory. Defaults to 'my-hls-test'.
        project_name (str, optional): Project name. Defaults to 'myproject'.
        input_data_tb (Any | None, optional): Input data for testbench. Defaults to None.
        output_data_tb (Any | None, optional): Output data for testbench. Defaults to None.
        backend (str, optional): Backend. Defaults to 'Vivado'. Currently, only 'Vivado' is tested.
        hls_config (Any | None, optional): hls4ml configuration. Defaults to None for auto configuration.
        bias_accum ([type], optional): Additional fp bits for accumulator. Defaults to None for bit-accurate accumulators.
        inline_everything (bool, optional): Inline everything. Helps with latency when using io_paraall.
        io_type (str, optional): IO type. Defaults to 'io_parallel'.
    """

    replica = create_replica(model)

    if hls_config is None:
        hls_config = hls4ml.utils.config_from_keras_model(replica, granularity='name')
        assert hls_config is not None
        update_layerconf(model, hls_config, bias_accum)

    model_hls = original_convert_from_keras_model(
        model=replica,
        output_dir=output_dir,
        project_name=project_name,
        input_data_tb=input_data_tb,
        output_data_tb=output_data_tb,
        backend=backend,
        hls_config=hls_config,
        io_type=io_type,
        **kwargs
    )

    write = model_hls.write

    def _write(self):
        write()
        patch_hls4ml_project(output_dir, model, inline_everything)

    model_hls.write = types.MethodType(_write, model_hls)

    return model_hls
