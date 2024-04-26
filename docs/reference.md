# Usage Reference

## Quantizer Config

Be default, the quantizer configs for the kernel and pre-activation are the following:

For kernel quantizer:
| option            | value              | description                                                    |
| ----------------- | ------------------ | -------------------------------------------------------------- |
| `init_bw`         | 2                  | initial bitwidth for kernel                                    |
| `skip_dims`       | None               | Which dimensions to quantize homogeneously.                    |
| `rnd_strategy`    | `"standard_round"` | How rounding is performed during training.                     |
| `exact_q_value`   | True               | Round bitwidth to integers before applying the quantization.   |
| `dtype`           | None               | dtype used for computing the quantization.                     |
| `bw_clip`         | (-23, 23)          | The bitwidth range for the floating part.                      |
| *`trainable`*     | True               | If the bitwidth is trainable.                                  |
| `regularizer`     | L1(1e-6)           | Regularization factor on the numerical bitwidth values.        |
| *`minmax_record`* | False              | Record the min/max values of the kernel if record flag is set. |

For pre-activation quantizer:
| option            | value      | description                                                            |
| ----------------- | ---------- | ---------------------------------------------------------------------- |
| `init_bw`         | 2          | initial floating bitwidth for pre-activation value                     |
| `skip_dims`       | (0,)       | Which dimensions to quantize homogeneously. (incl. batch)              |
| `rnd_strategy`    | `"auto"`   | How rounding is performed during training.                             |
| `exact_q_value`   | True       | Round bitwidth to integers before applying the quantization.           |
| `dtype`           | None       | dtype used for computing the quantization.                             |
| `bw_clip`         | (-23, 23)  | The bitwidth range for the floating part.                              |
| *`trainable`*     | True       | If the bitwidth is trainable.                                          |
| `regularizer`     | `L1(1e-6)` | Regularization factor on the numerical bitwidth values.                |
| *`minmax_record`* | True       | Record the min/max values of the pre-activation if record flag is set. |

You can get/set the default quantizer configs by calling `get_default_kq_conf`/`get_default_paq_conf` and `set_default_kq_conf`/`set_default_paq_conf`.

When changing the quantizer configs for a specific layer, pass the config dict to the layer with `kq_conf` or `paq_conf` keyword.

## Supported layers

To define a HGQ model, the following layers are available:

Heterogenerous layers (`H-` prefix):

- `HQuantize`: Heterogeneous quantization layer.
- `HDense`: Dense layer.
- `HConv*D`: Convolutional layers. Only 1D and 2D convolutional layers are exposed, as 3D conv layer is not supported by hls4ml.
  - Param `parallel_factor`: how many kernel operations are performed in parallel. Defaults to 1. This parameter will be passed to hls4ml.
- `HActivation`: Similar to the `Activation` layer, but with (heterogeneous) activations.
  - Supports any built-in keras activations, but may or may not be supported by hls4ml, or bit-accurate in general.
  - The tested activations are: `linear`, `relu`, `sigmoid`, `tanh`, `softmax`. `softmax` is never bit-accurate, and `tanh` and `sigmoid` are only bit-accurate when certain conditions are met.
- `HAdd`: Element-wise addition.
- `HDenseBatchNorm`: `HDense` with fused batch normalization. No resource overhead when converting to hls4ml.
- `HConv*DBatchNorm`: `HConv*D` with fused batch normalization. No resource overhead when converting to hls4ml.
- (New in 0.2) `HActivation` with **arbitrary unary function**. (See note below.)

```{note}
`HActivation` will be converted to a general `unaryLUT` in `to_proxy_model` when
 - the required table size is smaller or equal to `unary_lut_max_table_size`.
 - the corresponding function is not `relu`.

Here, table size is determined by $2^{bw_{in}}$, where $bw_{in}$ is the bitwidth of the input.

If the condition is not met, already supported activations like `tanh` or `sigmoid` will be done in the traditional way. However, if a arbitrary unary function is used, the conversion will fail. Thus, when using arbitrary unary functions, make sure that the table size is small enough.
```

```{note}
`H*BatchNorm` layers require both scaling and shifting parameters to be fused into the layer. Thus, when bias is set to `False`, shifting will not be available.
```

Passive layers (`P-` prefix):

- `PMaxPooling*D`: Max pooling layers.
- `PAveragePooling*D`: Average pooling layers.
- `PConcatenate`: Concatenate layer.
- `PReshape`: Reshape layer.
- `PFlatten`: Flatten layer.
- `Signature`: Does nothing, but marks the input to the next layer as already quantized to specified bitwidth.

```{note}
Average pooling layers are now bit-accurate, with the requirement that **all** individual pool size is a power of 2. This include all padded pools, with are with smaller sizes, if any.
```

```{warning}
As of hls4ml v0.9.1, padding in pooling layers with `io_stream` is not supported. If you are using `io_stream`, please make sure that the padding is set to `valid`. For more details, merely setting `padding='same'` is fine, but no actual padding may be performed, or the generated firmware will fail at an assertion.
```

## Commonly used functions

- `trace_minmax`: Trace the min/max values of the model against a dataset, print computed `BOPs` per-layer, and return the accumulated `BOPs` of the model.
- `to_proxy_model`: Convert a HGQ model to a hls4ml-compatible proxy model. The proxy model will contain all necessary information for HLS synthesis.
  - Param `aggressive`: If `True`, the proxy model will use `WRAP` for as overflow mode for all layers in seek of latency. If `False`, the overflow mode will be set to `SAT`.

## Callbacks

- `ResetMinMax`: Reset the min/max values of the model after each epoch. This is useful when the model is trained for multiple epochs.
- `FreeBOPs`: Add the accumulated `BOPs` of the model computed during training after each epoch to the model as a metric `bops`. As min/max registered during training will usually have a larger range than actual, this `BOPs` will usually be an overestimate.
- `CalibratedBOPs`: Add the accumulated `BOPs` of the model computed during training after each epoch to the model as a metric `bops`. The `BOPs` will be computed against a calibration dataset.

## Proxy model

The proxy model is a bridge between the HGQ model and hls4ml. It contains all necessary information for HLS synthesis, and can be converted to a hls4ml model by calling `convert_from_keras_model`. The proxy model is also bit-accurate with the hls4ml **with or without overflow**.

Before converting a HGQ model to proxy model, you must call the `trace_minmax` first, or the conversion will likely to fail.

```{tip}
If there is overflow, the proxy model will have different outputs to the HGQ model. This can be used as a fast check before hls4ml inference test. If there is a discrepancy, consider increase the `cover_factor` when performing `trace_minmax` against a calibration dataset.
```

```{note}
Though the proxy model is bit-accurate with hls4ml in general, exceptions exist:

1. Some intermediate values cannot be represented by the floating point format used by tensorflow, which is usually `float32` (23 bits mantissa) or `TF32` (10 bits mantissa).
2. For activations, bit-accuracy cannot be guaranteed. A great example of this is `softmax`. Also, unary nonlinear activations may or may not be bit-accurate with the current hls4ml implementation. Currently, if the bitwidth is very high and the input value's range is greater than a certain value, bit-accuracy will be lost due to some hardcoded LUT size in hls4ml.
```

```{tip}
The proxy model can also be used to convert a `QKeras` model to a bit-accurate hls4ml-ready proxy model. See more details in the [Regarding QKeras](qkeras.md) section.
```

```{warning}
Experimental: Nested layer structure is now supported by `to_keras_model` in v0.2.0. If you pass a model with nested layers, the function will flatten the model. However, be careful that some information in the inner models (e.g., `parallelization_factor`) may be lost during the conversion.
```
