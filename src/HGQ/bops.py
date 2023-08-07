import numpy as np
import tensorflow as tf
from tensorflow import keras

from .layers import HLayerBase


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


class ExactBOPs(tf.keras.callbacks.Callback):
    def __init__(self, calibration_data, bsz=None):
        self.calibration_data = calibration_data
        self.bsz = bsz

    def on_epoch_end(self, epoch, logs=None):
        assert isinstance(logs, dict)
        assert isinstance(self.model, keras.models.Model)

        bops = 0
        data = self.calibration_data
        for l in self.model.layers:
            if isinstance(l, HLayerBase):
                l.reset_minmax()
        bsz = self.bsz or self.params['batch_size']
        _ = model.predict(data, batch_size=bsz, verbose=0)  # type: ignore
        for l in self.model.layers:
            if isinstance(l, HLayerBase):
                dbops = l.compute_exact_bops
                bops += dbops
        for l in self.model.layers:
            if isinstance(l, HLayerBase):
                l.reset_minmax()
        logs['multi'] = bops


def compute_bops(model, dataset, bsz=16384, verbose=True, return_results=False, rst=True):
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
