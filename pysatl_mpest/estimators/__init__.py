"""Provides tools for estimating the parameters of mixture models.

This package brings together various algorithms and tools for estimating the
parameters of `MixtureModel` objects from a data sample. All estimation
algorithms inherit from the abstract base class `BaseEstimator`, which defines a
unified interface—the `fit` method.

The package offers two main approaches to estimation:

1.  **Iterative Algorithms** (in the :mod:`pysatl_mpest.estimators.iterative` submodule):
    A flexible module for building complex, multi-step algorithms like
    Expectation-Maximization (EM). It allows for fine-tuning each step,
    stopping conditions, and additional strategies.

2.  **Direct Algorithms** (in the :mod:`pysatl_mpest.estimators.direct` submodule):
    Simpler methods that compute parameters in a single step, for example, by
    directly solving Maximum Likelihood Estimation (MLE) equations or using the
    Method of Moments (MoM).

For ease of use, **Facades** are also provided. These are high-level classes
that hide the complexity of configuration and offer ready-to-use, popular
algorithms.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .base_estimator import BaseEstimator
from .ecm import ECM

__all__ = ["ECM", "BaseEstimator"]
