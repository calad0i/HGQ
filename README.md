<img src="docs/_static/logo.svg" alt="HGQ-logo" width="230"/>


# High Granularity Quantization

HGQ is a method for quantization aware training of neural works to be deployed on FPGAs, which allows for per-weight and per-activation bitwidth optimization.

Depending on the specific [application](https://arxiv.org/abs/2006.10159), HGQ could achieve up to 10x resource reduction compared to the traditional `AutoQkeras` approach, while maintaining the same accuracy. For some more challenging [tasks](https://arxiv.org/abs/2202.04976), where the model is already under-fitted, HGQ could still improve the performance under the same on-board resource consumption. For more details, please refer to our paper (link coming not too soon).

This repository implements HGQ for `tensorflow.keras` models. It is independent of the [QKeras project](https://github.com/google/qkeras).

Notice: this repository is still under development, and the API might change in the future.


## This package is still under development. Any API might change without notice at any time!
