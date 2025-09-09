"""
The `distribution` package provides an abstract class of continuous distributions and concrete implementations.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.distributions.exponential import Exponential
from rework_pysatl_mpest.distributions.uniform import Uniform

__all__ = [
    "ContinuousDistribution",
    "Exponential",
    "Uniform"
]
