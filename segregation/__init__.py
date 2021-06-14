__version__ = "2.0.0"
"""
:mod:`segregation` --- Spatial/Aspatial Segregation Analysis
=================================================

"""
from . import batch, decomposition, dynamics, inference, local, multigroup, singlegroup, util


# below handles deprecation warnings. Remove in 2.2.0
from . import  aspatial, spatial
