def init_all():
    try:
        from .qkeras import init
        init()
    except ImportError:
        pass
