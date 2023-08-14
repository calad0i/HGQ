import numpy as np
import tensorflow as tf
from tensorflow import keras

from .layers import HLayerBase
from .utils import warn


class FreeBOPs(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        assert self.model is not None
        bops = 0
        for layer in self.model.layers:
            if hasattr(layer, 'bops'):
                bops += layer.bops.numpy()
        logs['multi'] = bops  # type: ignore


class ResetMinMax(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        assert self.model is not None
        for layer in self.model.layers:
            if isinstance(layer, HLayerBase):
                layer.reset_minmax()


class CalibratedBOPs(tf.keras.callbacks.Callback):
    def __init__(self, calibration_data, bsz=None):
        self.calibration_data = calibration_data
        self.bsz = bsz

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(logs, dict)
        assert isinstance(self.model, keras.models.Model)

        data = self.calibration_data
        bsz = self.bsz or len(data)
        bops = compute_bops(self.model, data, bsz=bsz, verbose=False)
        logs['multi'] = bops


def compute_bops(model, dataset, bsz=16384, verbose=True, return_results=False, rst=True, cover_factor=1.0):
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
            aq = layer.pre_activation_quantizer
            aq._min.assign(aq(aq._min * cover_factor))  # type: ignore
            aq._max.assign(aq(aq._max * cover_factor))  # type: ignore

    bops = 0
    for layer in model.layers:
        if isinstance(layer, HLayerBase):
            layer.record_minmax = False
            dbops = layer.compute_exact_bops  # type: ignore
            bops += dbops
            if verbose:
                print(f'{layer.name}: {dbops}')
    if not return_results:
        return bops

    return bops, np.concatenate(r, axis=0)
