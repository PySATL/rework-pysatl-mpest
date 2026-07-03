"""Mocks for the distributions module.

Provides dummy continuous distributions with predictable behavior to test
estimators and initializers without complex likelihood evaluations.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .continuous_dist import MockContinuousDistribution, MockInfLpdfContinuousDistribution

__all__ = ["MockContinuousDistribution", "MockInfLpdfContinuousDistribution"]
