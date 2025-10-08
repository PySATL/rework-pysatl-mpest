"""
Pysatl-mpest: A Python package for statistical analysis using mixture models.

This package provides tools for creating, manipulating, and estimating parameters
of finite mixture models of continuous probability distributions.

Key modules include:
- `core`: Provides the main `MixtureModel` class and the `Parameter` descriptor.
- `distributions`: Contains an abstract base class for continuous distributions
  and concrete implementations like `Exponential`, `Uniform`.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from rework_pysatl_mpest.core import MixtureModel, Parameter
from rework_pysatl_mpest.distributions import (
    Cauchy,
    ContinuousDistribution,
    Exponential,
    Normal,
    Pareto,
    Uniform,
    Weibull,
)

__all__ = [
    "Cauchy",
    "ContinuousDistribution",
    "Exponential",
    "MixtureModel",
    "Normal",
    "Parameter",
    "Pareto",
    "Uniform",
    "Weibull",
]
