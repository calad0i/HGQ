def patch_singledispatch():
    "Patch functools.singledispatch to support Union types. Invoked only for python<=3.10"
    import sys
    if sys.version_info >= (3, 11):
        return
    import functools
    from functools import singledispatch as _singledispatch
    from types import UnionType

    def singledispatch(func):
        func = _singledispatch(func)
        _register = func.register

        def register(cls, func=None):
            from typing import get_type_hints
            if isinstance(cls, type):
                return _register(cls, func)
            if isinstance(cls, UnionType):
                def register_seq(func):
                    for t in cls.__args__:
                        _register(t, func)
                    return func
                return register_seq

            func = cls
            argname, cls = next(iter(get_type_hints(func).items()))
            if isinstance(cls, type):
                return _register(cls, func)
            if isinstance(cls, UnionType):
                for t in cls.__args__:
                    func = _register(t, func)
                return func

        func.register = register  # type: ignore
        return func
    setattr(functools, 'singledispatch', singledispatch)


patch_singledispatch()
