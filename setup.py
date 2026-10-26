"""Setuptools shim: marks the wheel as platform-specific because it ships the compiled C++ core
(psim/numerics/_psim_native*). Build that first with `python build_native.py`; all metadata is in
pyproject.toml."""
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


setup(distclass=BinaryDistribution)
