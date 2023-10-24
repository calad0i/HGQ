import sys


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


shutup = Shutup()

with shutup:

    from .utils import get_default_kernel_quantizer_config, get_default_pre_activation_quantizer_config
    from .utils import set_default_kernel_quantizer_config, set_default_pre_activation_quantizer_config
    from .quantizer import HGQ
    from .layers import HDense, Signature, HQuantize, HConv1D, HConv2D
    from .layers import PReshape, PFlatten, PConcatenate
    from .layers import PMaxPool1D, PMaxPool2D, PAvgPool1D, PAvgPool2D
    from .layers import Signature
    from .layers import PFlatten, PReshape
    from .bops import trace_minmax, FreeBOPs, ResetMinMax, CalibratedBOPs