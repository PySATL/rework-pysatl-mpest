"""Helper utilities for testing PySATL algorithms and mathematical assertions.

This package provides mathematical assertions, golden data comparisons, and other
utilities that are shared across various test suites.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .golden import GoldenDataComparator
from .math_assertions import assert_computational_stability, assert_no_nan_inf, assert_probabilities_sum_to_one

__all__ = [
    "GoldenDataComparator",
    "assert_computational_stability",
    "assert_no_nan_inf",
    "assert_probabilities_sum_to_one",
]
