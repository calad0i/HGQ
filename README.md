# High Granularity Quantization for Ultra-Fast Inference on FPGAs

HGQ is a method for quantization aware training of neural works to be deployed on FPGAs, which allows for per-weight and per-activation bitwidth optimization.

Depending on the specific [application](https://arxiv.org/abs/2006.10159), HGQ could achieve up to 10x resource reduction compared to the traditional `AutoQkeras` approach, while maintaining the same accuracy. For some more challenging [tasks](https://arxiv.org/abs/2202.04976), where the model is already under-fitted, HGQ could still improve the performance under the same on-board resource consumption. For more details, please refer to our paper (link coming not too soon).

This repository implements HGQ for `tensorflow.keras` models. It is independent of the [QKeras project](https://github.com/google/qkeras).

Notice: this repository is still under development, and the API might change in the future.

## This package is still under development. Any API might change without notice at any time.

## Installation

`pip install HGQ`, and you are good to go. Note that HGQ requires `python3.10` and `tensorflow>=2.11`.

## Usage Guide

Please refer to the [usage guide](./usage_guide.md) for more details.
This [repo](https://github.com/calad0i/HGQ-demos) contains some use cases for HGQ.

## FAQ

Please refer to the [FAQ](./faq.md) for more details.

## Citation

The paper is not ready. Please check back later.
