# Installation

Use `pip install --pre HGQ` to install the latest version from PyPI. You will need a environment with `python==3.11` installed.

```{warning}
This framework requires an **unmerged** [PR](https://github.com/fastmachinelearning/hls4ml/pull/914) of hls4ml. Please install it by running `pip install "git+https://github.com/calad0i/hls4ml@HGQ-integration"`. Or, conversion will fail with unsupported layer error.
```

```{note}
The current varsion requires an **unmerged** version of hls4ml. Please install it by running `pip install git+https://github.com/calad0i/hls4ml`.
```

```{warning}
HGQ v0.2 requires `python3.10/3.11` and `tensorflow==2.13`. Please make sure that you have the correct version of python and tensorflow installed.
```

If you want to use the legency v0.1 version, please run `pip install HGQ "tensorflow==2.11" "numpy>=1.23"` to install it from PyPI. You will need a environment with `python==3.10` installed.

```{warning}
Due to broken dependency declaration, you will need to specify the version of tensorflow manually. Otherwise, there will likely to be version conflicts.
```
