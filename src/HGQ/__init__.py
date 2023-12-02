import sys

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

from . import _patch


class Shutup:
    def write(self, s):
        pass

    def flush(self):
        pass

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = self
        self._original_stderr = sys.stderr
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

    def close(self):
        pass


shutup = Shutup()

with shutup:

    from .bops import CalibratedBOPs, FreeBOPs, ResetMinMax, trace_minmax
    from .layers import HConv1D, HConv1DBatchNorm, HConv2D, HConv2DBatchNorm, HDense, HDenseBatchNorm, HQuantize, PAvgPool1D, PAvgPool2D, PConcatenate, PFlatten, PMaxPool1D, PMaxPool2D, PReshape, Signature
    from .proxy import to_proxy_model
    from .quantizer import HGQ
    from .utils import get_default_kq_conf, get_default_paq_conf, set_default_kq_conf, set_default_paq_conf
