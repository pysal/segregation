"""
:mod:`segregation` --- Spatial/Aspatial Segregation Analysis
=================================================

"""
import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import batch, decomposition, dynamics, inference, local, multigroup, singlegroup, util

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("segregation")