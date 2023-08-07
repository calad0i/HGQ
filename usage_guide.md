# Usage Guide

## Installation

`pip install HGQ`, and you are good to go. Note that HGQ requires `python3.10` and `tensorflow>=2.11`.

## Getting Started

You need at minimal four extra steps to use HGQ in your keras project:

```python

from HGQ import HDense, HQuantize
from HGQ.bops import compute_bops, ResetMinMax
from tensorflow.keras.models import Sequential
from HGQ.hls4ml_hook import convert_from_hgq_model
....

#regularization factor on MBOPs, higher for smaller bitwidth
bops_reg_factor = 1e-5 

# The first layer must be quantized, either by using HQuantize or Signature layers.
# The input quantization layer's name must contain 'inp_q' if you want to quantize the input heterogeneously.
# Use only layers provided by HGQ. You can use functional API as well.
# Please refer to the list below in this document for the full list of supported layers.
model = Sequential([
    HQuantize(bops_reg_factor=bops_reg_factor, name='inp_q', input_shape=(16)),
    HDense(10, activation='relu', bops_reg_factor=bops_reg_factor),
    HDense(10, bops_reg_factor=bops_reg_factor),
])

...

callbacks.append(ResetMinMax()) # Reset min/max every epoch, or the estimated MBOPs could be very inaccurate.

model.fit(..., callbacks=callbacks)

...

# Compute the exact MBOPs of the model.
# This step is NOT optional, as it also records the min/max pre-activation for each layer, which is necessary for determine the number of integer bits.
compute_bops(model, X_train, bsz=bsz)

# Convert the model to HLS4ML. Only vivado backend is test so far. Heterogeneous activation will NOT work with other backends. Weight heterogeneity MAY work.
model_hls = convert_from_hgq_model(
    model,
    'hls4ml_prj',
    part='xcvu9p-flga2104-2L-e',
    clock_period=5,
    bias_accum=None
)

... (standard hls4ml workflow)

```

For a complete example, please refer to this [notebook](https://github.com/calad0i/HGQ-demos/blob/master/minimal/usage_example.ipynb). Also check out the [demo repo](https://github.com/calad0i/HGQ-demos/) for more use cases.

## Configure the HG Quantizer

```python
from HGQ import set_default_kernel_quantizer_config, set_default_pre_activation_quantizer_config
from HGQ import get_default_kernel_quantizer_config, get_default_pre_activation_quantizer_config

# The default quantizers for the pre-activation and kernel are the following:

DEFAULT_KERNEL_QUANTIZER_CONFIG = \
    dict(
         # initial bitwidth for the floating part
         init_bw=2,
         # Which dimensions to quantize homogeneously. Accept a tuple of integers, or any of ['all', 'batch', 'none', 'except_last', 'except_1st'].
         skip_dims=None, 
         # How rounding is performed in training. Can choose from ['floor', 'standard_round', 'stochastic_round', 'fast_uniform_noise_injection', 'auto'].
         # In testing, 'standard_round' is used for everything except for 'floor'.
         # 'auto': 'floor' for layer without bias except HActivation layers, 'standard_round' otherwise.
         rnd_strategy='standard_round',
         # Whether round bitwidth to integers before applying the rounding. Defaults to True for weights and False for pre-activations.
         exact_q_value=True,
         dtype=None,
         # The bitwidth range for the floating part.
         bw_clip=(-23, 23),
         # Whether the bitwidth is trainable.
         trainable=True,
         # Regularization factor on the numerical bitwidth values. Useful for preventing the bitwidth from being too large for activations does not got invlolved in mul ops (e.g. final layer, layer before HActivation, etc...)
         regularizer=L1(1e-6),
         )


DEFAULT_PRE_ACTIVATION_QUANTIZER_CONFIG = \
    dict(init_bw=2,
         skip_dims=(0,), # Same to 'batch'. skipping the batch dimension, which should always be homogeneously quantized.
         rnd_strategy='standard_round',  
         exact_q_value=False,
         dtype=None,
         bw_clip=(-23, 23),
         trainable=True,
         regularizer=L1(1e-6),
         minmax_record=True
         )
```

You can set the default quantizer config for the kernel and pre-activation quantizers by calling `set_default_kernel_quantizer_config` and `set_default_pre_activation_quantizer_config`. You can also get the default quantizer config by calling `get_default_kernel_quantizer_config` and `get_default_pre_activation_quantizer_config`.

When changing the quantizer configs for a specific layer, pass the config dict to the layer with `kernel_quantizer_config` or `pre_activation_quantizer_config` keyword.

## Supported Layers

### HG Layers

Layers that (can) do HG quantization on the (pre-)activation values:

`HQuantize`: Quantize the input to the next layer. When used just after the input layer, add the `inp_q` keyword to the name of the layer. The user must use this layer or `Signature` layer directly after the input layer.

`HDense`: Dense layer with HGQ.

`HConv1D/2D`: Convolutional layers with HGQ.

`HActivation`: Similar to the `Activation` layer, but with (heterogeneous) activation

`HAdd`: Element-wise addition with HGQ.

`HBaseLayer`: Base layer for HGQ layers. Do not use this layer directly. Child layers with one input should overload `forward` and `compute_exact_bops` methods in most cases.

### Passive Layers

Layers that do not do HG quantization, but passes extra necessary information necessary for HGQ to the next layer:

`PXXXPoolND`: Pooling layers.

`PFlatten/PReshape`: Flatten/Reshape layers.

`PConcatenate`: Concatenate layers.

`PLayerBase`: Base layer for passive layers. Do not use this layer directly.

### Signature Layer

`Signature`: A special layer that does not do anything, but passes the input to the next layer. This layer is used to indicate the input data to it is already quantized to some specific bitwidth. The user must use this layer or `HQuantize` layer directly after the input layer.
