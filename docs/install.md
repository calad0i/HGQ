# Installation

Use `pip install HGQ` to install the latest version from PyPI. You will need a environment with `python>=3.10` installed. Currently, only `python3.10 and 3.11` are tested.

```{warning}
This framework requires an **unmerged** [PR](https://github.com/fastmachinelearning/hls4ml/pull/914) of hls4ml. Please install it by running `pip install "git+https://github.com/calad0i/hls4ml@HGQ-integration"`. Or, conversion will fail with unsupported layer error.
```

```{warning}
HGQ v0.2 requires `tensorflow>=2.13,<2.16` (tested on 2.13 and 2.15; 2.16 untested but may work) and `python>=3.10`. Please make sure that you have the correct version of python and tensorflow installed.
```
