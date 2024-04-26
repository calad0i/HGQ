# Quick Start

```{note}
This guide is only for models with fully heterogeneous quantized weights (per-weight bitwidth).
```

## Model definition & training

Let's consider the following model for MNIST classification:

```python
import keras

model = keras.models.Sequential([
    keras.layers.Reshape((28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(2, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(2, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

opt = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

```

To quantize the model with HGQ, the following steps are required:

1. Replace all layers by corresponding HGQ layers. For example, replace `Dense` by `HDense`, `Flatten` by `PFlatten`, etc...
2. The first layer must be a `HQuantize` or `Signature` layer.
3. Add `ResetMinMax()` callback to the last of callbacks.

``` python
from HGQ.layers import HDense, HConv2D, PMaxPooling2D, PFlatten, PReshape, HQuantize
from HGQ import ResetMinMax, FreeBOPs

model = keras.models.Sequential([
    HQuantize(beta=3e-5),
    PReshape((28, 28, 1)),
    PMaxPooling2D((2, 2)),
    HConv2D(2, (3, 3), activation='relu', beta=3e-5, parallel_factor=144),
    PMaxPooling2D((2, 2)),
    HConv2D(2, (3, 3), activation='relu', beta=3e-5, parallel_factor=16),
    PMaxPooling2D((2, 2)),
    PFlatten(),
    HDense(10, beta=3e-5)
])

opt = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
callbacks = [ResetMinMax(), FreeBOPs()]
```

The `beta` factor is a regularization factor on the number of BOPs. Higher `beta` means smaller bitwidth. The `beta` factor can be set to different values for different layers.

The `parallel_factor` is the number controls how many output elements are computed in parallel. In this example, the `parallel_factor` is set to match the output size, thus everything will be computed in parallel. In this case, we would have minimum latency & maximum throughput at the cost of larger logic.

```{tip}
Though not strictly necessary, it is recommended to add `FreeBOPs()` to callbacks. This callback will add a new metric `bops` to the model, and you will be able to estimate the size of the model on the fly during training.
```

Then, by calling `model.fit(..., callbacks=callbacks)`, the model will be trained with HGQ. To pass the model to hls4ml for HLS synthesis, you would need

## Conversion to hls4ml

1. Call `trace_minmax(model, data, bsz=..., cover_factor=...)` to compute the necessary bitwidth for each layer, against a calibration dataset.
2. Convert the HGQ model to a hls4ml-compatible proxy model by calling `to_proxy_model`.
3. Call `convert_from_keras_model` as usual, and enjoy **almost bit-accurate** HLS synthesis.

```python
from HGQ import trace_minmax, to_proxy_model
from hls4ml.converters import convert_from_keras_model

trace_minmax(model, x_train, cover_factor=1.0)
proxy = to_proxy_model(model, aggressive=True)

model_hls = convert_from_keras_model(proxy, backend='vivado',output_dir=... ,part=...)
```

```{tip}
The proxy model is meant to contain all necessary information for HLS synthesis. The user will not need to modify the `hls_config` for conversion in general.
```

```{note}
By almost bit-accurate, the model will be bit-accurate except for the following cases:

1. Some intermediate values cannot be represented by the floating point format used by tensorflow, which is usually `float32` (23 bits mantissa) or `TF32` (10 bits mantissa).
2. Overflow happens. As the bitwidth is determined only against the calibration dataset, overflow may happen during inference on the test dataset. This can usually be mitigated by increasing the `cover_factor` in the `trace_minmax` function at the cost of larger synthesized logic.
3. For activations, bit-accuracy cannot be guaranteed. A great example of this is `softmax`. Also, unary nonlinear activations may or may not be bit-accurate with the current hls4ml implementation. Currently, if the bitwidth is very high and the input value's range is greater than a certain value, bit-accuracy will be lost due to some hardcoded LUT size in hls4ml.
```

For a complete example, please refer to this [notebook](https://github.com/calad0i/HGQ/tree/master/examples/mnist.ipynb).
