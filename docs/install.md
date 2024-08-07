# Installation

Use `pip install HGQ` to install the latest version from PyPI. You will need an environment with `python>=3.10` installed. Currently, only `python3.10 and 3.11` are tested.

```{warning}
This package requires `hls4ml>=1.0` to handle model conversion to HLS projects. As of now, the PyPI release of hls4ml (0.9) is still not compatible with HGQ's models, and one will still need to install the latest version of hls4ml from github: `pip install "git+https://github.com/fastmachinelearning/hls4ml"`. Future PyPI releases of hls4ml (>=1.0) will support HGQ models.
```

```{warning}
HGQ v0.2 requires `tensorflow>=2.13,<2.16` (tested on 2.13 and 2.15; 2.16 untested but may work) and `python>=3.10`. Please make sure that you have the correct version of python and tensorflow installed.
```
