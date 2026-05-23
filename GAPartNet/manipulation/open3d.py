"""Minimal Open3D stub for headless manipulation eval.
The parallel-jaw baseline does not use Open3D unless visual point-cloud output is requested.
"""
class _MissingOpen3D:
    def __getattr__(self, name):
        raise RuntimeError("open3d is not installed; this headless eval stub only supports non-visual code paths")
geometry = _MissingOpen3D()
utility = _MissingOpen3D()
visualization = _MissingOpen3D()
io = _MissingOpen3D()
camera = _MissingOpen3D()
