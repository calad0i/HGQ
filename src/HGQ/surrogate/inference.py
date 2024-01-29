from dataclasses import dataclass, field
from functools import singledispatch
from pathlib import Path

import h5py as h5
import keras
import numpy as np
import pandas as pd

from HGQ.proxy import FixedPointQuantizer, UnaryLUT
from HGQ.quantizer.quantizer import get_arr_bits
from HGQ.utils import apf_to_tuple, tuple_to_apf, warn

from .feature import LayerSurrogateFeature, get_hls_config, get_layer_feature
from .resource_map import to_bram18, to_DSP, to_LUT, to_REG


class ModelSurrogate:
    def __init__(self, features: dict[str, LayerSurrogateFeature]):
        self.features = features

    @classmethod
    def from_proxy(cls, model: keras.Model):
        "Get model surrogate feature from HGQ proxy model"
        features = {}
        hls_conf = get_hls_config(model)
        for layer in model.layers:
            feature = get_layer_feature(layer, hls_conf)
            if feature is None:
                continue
            features[layer.name] = feature
        return cls(features)

    def _save(self, f: h5.Group):
        for name, feature in self.features.items():
            feature.save(f)

    def save(self, f: h5.Group | str | Path):
        "Save to h5 file"
        if isinstance(f, (str, Path)):
            with h5.File(f, 'w') as f:
                self._save(f)
        else:
            self._save(f)

    @classmethod
    def _load(cls, f: h5.Group):
        features = {}
        for name in f.keys():
            features[name] = LayerSurrogateFeature.load(f, name)
        return cls(features)

    @classmethod
    def load(cls, f: h5.Group | str | Path):
        "Load from h5 file"
        if isinstance(f, (str, Path)):
            with h5.File(f, 'r') as f:
                return cls._load(f)
        else:
            return cls._load(f)

    def __repr__(self) -> str:
        return f"ModelSurrogateFeature({self.features})"

    @property
    def bops(self):
        return sum(v.bops for v in self.features.values())

    def predict(self) -> pd.DataFrame:
        "Predict the latency of each layer"
        D = {}
        for name, feature in self.features.items():
            lut = to_LUT(feature)
            dsp = to_DSP(feature)
            bram = to_bram18(feature)
            reg = to_REG(feature)
            d = {'LUT': lut, 'DSP48': dsp, 'BRAM18': bram, 'REG': reg}
            D[name] = d
        df = pd.DataFrame.from_dict(D, orient='index')
        df.index.name = 'layer'
        df.loc['TOTAL'] = df.sum(axis=0)
        return df

    def flatten(self):
        return np.sum([f.flatten() for f in self.features.values()], axis=0)


def proxy_to_surrogate(proxy_model: keras.Model):
    "Wrapper for ModelSurrogateFeature.from_proxy"
    return ModelSurrogate.from_proxy(proxy_model)


def load_surrogate(path: str | Path):
    "Wrapper for ModelSurrogateFeature.load"
    return ModelSurrogate.load(path)
