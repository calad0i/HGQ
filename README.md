<img src="docs/_static/logo.svg" alt="HGQ-logo" width="230"/>


# High Granularity Quantization

[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Documentation](https://github.com/calad0i/HGQ/actions/workflows/sphinx-build.yml/badge.svg)](https://calad0i.github.io/HGQ/)
[![PyPI version](https://badge.fury.io/py/hgq.svg)](https://badge.fury.io/py/hgq)


HGQ is a framework for quantization aware training of neural networks to be deployed on FPGAs, which allows for per-weight and per-activation bitwidth optimization.

Depending on the specific [application](https://arxiv.org/abs/2006.10159), HGQ could achieve up to 10x resource reduction compared to the traditional `AutoQkeras` approach, while maintaining the same accuracy. For some more challenging [tasks](https://arxiv.org/abs/2202.04976), where the model is already under-fitted, HGQ could still improve the performance under the same on-board resource consumption. For more details, please refer to our paper (link coming not too soon).

This repository implements HGQ for `tensorflow.keras` models. It is independent of the [QKeras project](https://github.com/google/qkeras).

## Warning:

This framework requires an **unmerged** [PR](https://github.com/fastmachinelearning/hls4ml/pull/914) of hls4ml. Please install it by running `pip install "git+https://github.com/calad0i/hls4ml@HGQ-integration"`. Or, conversion will fail with unsupported layer error.

## This package is still under development. Any API might change without notice at any time!
