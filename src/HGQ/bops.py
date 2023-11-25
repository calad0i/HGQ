import numpy as np
import tensorflow as tf
from tensorflow import keras

from .layers import HLayerBase
from .utils import warn


class FreeEBOPs(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        assert self.model is not None
        ebops = 0
        for layer in self.model.layers:
            if hasattr(layer, 'ebops'):
                ebops += layer.ebops.numpy()
        logs['ebops'] = ebops  # type: ignore


class ResetMinMax(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        assert self.model is not None
        for layer in self.model.layers:
            if isinstance(layer, HLayerBase):
                layer.reset_minmax()


class CalibratedEBOPs(tf.keras.callbacks.Callback):
    def __init__(self, calibration_data, bsz=None):
        self.calibration_data = calibration_data
        self.bsz = bsz

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(logs, dict)
        assert isinstance(self.model, keras.models.Model)

        data = self.calibration_data
        bsz = self.bsz or len(data)
        ebops = trace_minmax(self.model, data, bsz=bsz, verbose=False)
        logs['multi'] = ebops


def trace_minmax(model, dataset, bsz=16384, verbose=True, return_predictions=False, no_ebops_computation=False, rst=True, cover_factor=1.0):
    if rst:
        for layer in model.layers:
            if isinstance(layer, HLayerBase):
                layer.reset_minmax()
                layer.record_minmax = True

    r = []
    if isinstance(dataset, list):
        length = len(dataset[0])
        for i in range(0, length, bsz):
            r.append(model([d[i:i + bsz] for d in dataset], training=False))
    else:
        for i in range(0, dataset.shape[0], bsz):
            r.append(model(dataset[i:i + bsz], training=False))

    if cover_factor != 1.0:
        assert cover_factor > 0.
        if cover_factor < 1:
            warn(f'cover_factor<1.0 will likely to result in overflows.')
        for layer in model.layers:
            if not isinstance(layer, HLayerBase):
                continue
            aq = layer.paq
            aq._min.assign(aq(aq._min * cover_factor))  # type: ignore
            aq._max.assign(aq(aq._max * cover_factor))  # type: ignore

    for layer in model.layers:
        if not hasattr(layer, 'activation'):
            continue
        aq = layer.paq
        if layer.activation is tf.keras.activations.softmax:
            # softmax_lut behaves slightly differently from softmax, which is more likely to generate exact 1s. Overflows may occur with traced min-max.
            aq._min.assign(tf.zeros_like(aq._min))  # type: ignore
            aq._max.assign(tf.ones_like(aq._max))  # type: ignore

        if layer.activation is tf.keras.activations.sigmoid:
            # <0 or >1 for sigmoid does not make sense. Undo the effect of cover_factor.
            aq._min.assign(tf.maximum(aq._min, tf.zeros_like(aq._min)))  # type: ignore
            aq._max.assign(tf.minimum(aq._max, tf.ones_like(aq._max)))  # type: ignore

        if layer.activation is tf.keras.activations.tanh:
            # <-1 or >1 for tanh does not make sense. Undo the effect of cover_factor.
            aq._min.assign(tf.maximum(aq._min, -tf.ones_like(aq._min)))  # type: ignore
            aq._max.assign(tf.minimum(aq._max, tf.ones_like(aq._max)))  # type: ignore

    if no_ebops_computation:
        return -1

    ebops = 0
    for layer in model.layers:
        if isinstance(layer, HLayerBase):
            layer.record_minmax = False
            debops = layer.compute_exact_bops  # type: ignore
            ebops += debops
            if verbose:
                print(f'{layer.name}: {debops}')

    if return_predictions:
        return ebops, np.concatenate(r, axis=0)

    return ebops
