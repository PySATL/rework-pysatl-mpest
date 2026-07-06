"""Mathematical assertions utilities for testing."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Any

import numpy as np
from pysatl_mpest.typings import FloatArray


def assert_probabilities_sum_to_one(weights: FloatArray, rtol: float = 1e-5, atol: float = 1e-8) -> None:
    """
    Asserts that the elements of the probability (or weights) array sum to one.
    If the array is multi-dimensional, the sum is checked along the last axis (axis=-1).
    """

    np.testing.assert_allclose(np.sum(weights, axis=-1), 1.0, rtol=rtol, atol=atol)


def assert_no_nan_inf(tensor: FloatArray) -> None:
    """
    Asserts that the tensor does not contain any NaN or Inf values.
    """

    assert not np.isnan(tensor).any(), "Array contains NaN"
    assert not np.isinf(tensor).any(), "Array contains Inf"


def assert_computational_stability(log_probs: FloatArray) -> None:
    """
    Asserts the computational stability of a log-probabilities (lpdf) tensor.
    Ensures no NaN values are present. For Inf, only -Inf is allowed
    (in case of log(0)), but +Inf is strictly forbidden.
    """

    assert not np.isnan(log_probs).any(), "Log-probabilities array contains NaN"
    assert not np.isposinf(log_probs).any(), "Log-probabilities array contains +Inf"


def assert_is_scalar_type(val: Any) -> None:
    """Asserts that the provided value is a numpy scalar of type float64."""

    assert np.isscalar(val), f"Expected scalar, got {type(val)}"
    assert isinstance(val, np.float64), f"Expected np.float64, got {type(val)}"


def assert_is_array_type(val: Any, expected_shape: tuple[int, ...]) -> None:
    """Asserts that the provided value is a numpy array of type float64 with the expected shape."""

    assert isinstance(val, np.ndarray), f"Expected np.ndarray, got {type(val)}"
    assert val.dtype == np.float64, f"Expected float64 dtype, got {val.dtype}"
    assert val.shape == expected_shape, f"Expected shape {expected_shape}, got {val.shape}"
