import numpy as np

from .feature import GRID_SIZE, LayerSurrogateFeature

#############################################
# multiplication to *


def to_LUT(feature: LayerSurrogateFeature) -> float:
    'Estimate the LUT size of a layer'
    mul_oprs = feature.mul_oprs
    x, y = GRID_SIZE
    a, b = np.arange(x) + 1, np.arange(y) + 1
    v = int(np.sum(mul_oprs * a[:, None] * b[None, :]))
    return round(v * 1.02)


def to_DSP(features: LayerSurrogateFeature) -> float:
    return 0


def to_bram18(features: LayerSurrogateFeature) -> float:
    a = np.arange(GRID_SIZE[1]) + 1
    return np.prod(features.lut_oprs * a).astype(float) / 328.


def to_REG(features: LayerSurrogateFeature) -> float:
    return 0
