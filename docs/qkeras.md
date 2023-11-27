# Regarding QKeras

This framework is designed as a alternative to `QKeras` for ultra-low latency applications, mainly L1 triggers at colliders:

While QKeras is designed as a general-purpose QAT framework, this framework is designed for L1 trigger-like applications only, which are highly, if not fully, unrolled and pipelined. This allows the framework to perform more aggressive optimizations that are not possible in `QKeras`.

## Compatibility

In general, this framework cannot be used in combination with `QKeras` layers or quantizers.

However, by reusing a part of precision-derivation logics in this framework, it is possible to convert a `QKeras` model to a hls4ml-ready proxy model that will generate bit-accurate after hls4ml workflow:

```python
from HGQ import to_proxy_model

proxy = to_proxy_model(qkeras_model, aggressive=False)
model_hls = convert_from_keras_model(proxy, ...)
```

Due to the default behavior of `QKeras` quantizer, it is strongly recommended to use `Aggressive=False` when performing the conversion. Otherwise, there will likely be a large discrepancy between the proxy model and the original `QKeras` model due to the different overflow mode.

```{note}
Not all QKeras layers are not supported, such as `QConv*DBatchNorm`. If the model contains such layers, the conversion will fail.
```


```{warning}
The pre-requisite of this conversion is that the `QKeras` model must be a model that may be converted to a hls4ml model in a bit-accurate manner. This has two major implications:

1. All inputs must be followed immediately by a `QActivation` layer with a `quantized_bits` activation to mark the input precision.
2. All quantizers, when applicable, must have `alpha=1`. The default arbitrary power-of-two scaling is NOT achievable in hls4ml.

If these conditions are not met, the conversion will fail.

Also, if the quantizer for any parameter is missing, the framework will try to derive a bitwidth that will produce a bit-accurate model. However, this may result in a huge bitwidth **silently without warning**. Hence, before running synthesis, it is strongly recommended to check the bitwidth manually.
```

You can find an example of this conversion in [this notebook](https://github.com/calad0i/HGQ/tree/master/examples/qkeras.ipynb).
