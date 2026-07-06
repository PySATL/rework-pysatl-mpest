"""
The `distribution` package provides an abstract class of continuous distributions and concrete implementations.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_core.families import configure_families_register

from .continuous_dist import ContinuousDistribution
from .exponential import Exponential
from .normal import Normal
from .uniform import Uniform

configure_families_register()

__all__ = ["ContinuousDistribution", "Exponential", "Normal", "Uniform"]
