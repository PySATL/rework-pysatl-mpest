"""Tests for the base ContinuousDistribution class via a mock."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import copy

import numpy as np
import pytest
from tests.helpers.math_assertions import assert_dtype
from tests.mocks.distributions.continuous_dist import (
    MockContinuousDistribution,
    MockInfLpdfContinuousDistribution,
)


def test_continuous_dist_fix_param(floating_dtype: type[np.floating]) -> None:
    """Test fixing valid and invalid parameters."""

    dist = MockContinuousDistribution(dtype=floating_dtype)
    assert dist.params_to_optimize == {"param1", "param2"}

    dist.fix_param("param1")
    assert dist.params_to_optimize == {"param2"}
    assert "param1" in dist._fixed_params

    with pytest.raises(ValueError, match="does not exist in this distribution"):
        dist.fix_param("invalid_param")


def test_continuous_dist_unfix_param(floating_dtype: type[np.floating]) -> None:
    """Test unfixing parameters and ignoring unfixed ones."""

    dist = MockContinuousDistribution(dtype=floating_dtype)
    dist.fix_param("param1")
    assert dist.params_to_optimize == {"param2"}

    dist.unfix_param("param1")
    assert dist.params_to_optimize == {"param1", "param2"}

    # Should not raise any errors
    dist.unfix_param("param1")
    dist.unfix_param("invalid_param")
    assert dist.params_to_optimize == {"param1", "param2"}


def test_continuous_dist_get_params_vector(floating_dtype: type[np.floating]) -> None:
    """Test retrieving parameter values as a list in specified order."""

    dist = MockContinuousDistribution(param1=10.0, param2=20.0, dtype=floating_dtype)

    vec = dist.get_params_vector(["param2", "param1"])
    np.testing.assert_allclose(vec, [20.0, 10.0], rtol=1e-5, atol=1e-8)
    assert_dtype(vec, floating_dtype)

    with pytest.raises(ValueError, match="Invalid parameter names provided"):
        dist.get_params_vector(["param1", "invalid"])


def test_continuous_dist_set_params_from_vector(floating_dtype: type[np.floating]) -> None:
    """Test updating parameters from a vector with dtype casting."""

    dist = MockContinuousDistribution(dtype=floating_dtype)

    expected_param2 = 100.5
    expected_param1 = 50.0

    dist.set_params_from_vector(["param2", "param1"], [expected_param2, expected_param1])

    np.testing.assert_allclose(dist.param2, expected_param2, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(dist.param1, expected_param1, rtol=1e-5, atol=1e-8)
    assert type(dist.param2) is floating_dtype
    assert type(dist.param1) is floating_dtype

    with pytest.raises(ValueError, match="number of parameter names must match"):
        dist.set_params_from_vector(["param1", "param2"], [1.0])

    with pytest.raises(ValueError, match="Invalid parameter names provided"):
        dist.set_params_from_vector(["param1", "invalid"], [1.0, 2.0])


def test_continuous_dist_astype(floating_dtype: type[np.floating]) -> None:
    """Test casting the distribution to a new floating precision."""

    dist = MockContinuousDistribution(param1=10.0, param2=20.0, dtype=floating_dtype)
    dist.fix_param("param1")

    # Pick a different dtype to cast to
    new_dtype = np.float32 if floating_dtype is np.float64 else np.float64

    new_dist = dist.astype(new_dtype)

    assert new_dist is not dist
    assert new_dist.dtype is new_dtype
    assert type(new_dist.param1) is new_dtype
    assert type(new_dist.param2) is new_dtype
    assert "param1" in new_dist._fixed_params

    # Test identity return if casting to the same dtype
    same_dist = dist.astype(floating_dtype)
    assert same_dist is dist


def test_continuous_dist_copy(floating_dtype: type[np.floating]) -> None:
    """Test creating a copy of the distribution."""

    dist = MockContinuousDistribution(param1=10.0, param2=20.0, dtype=floating_dtype)
    dist.fix_param("param1")

    dist_copy = copy.copy(dist)

    assert dist_copy is not dist
    np.testing.assert_allclose(dist_copy.param1, dist.param1, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(dist_copy.param2, dist.param2, rtol=1e-5, atol=1e-8)
    assert "param1" in dist_copy._fixed_params
    assert dist_copy.dtype is floating_dtype


def test_continuous_dist_eq(floating_dtype: type[np.floating]) -> None:
    """Test equality comparison between distributions."""

    dist1 = MockContinuousDistribution(param1=10.0, param2=20.0, dtype=floating_dtype)
    dist2 = MockContinuousDistribution(param1=10.0, param2=20.0, dtype=floating_dtype)

    assert dist1 == dist2

    # Different parameter values
    dist3 = MockContinuousDistribution(param1=15.0, param2=20.0, dtype=floating_dtype)
    assert dist1 != dist3

    # Different dtype
    other_dtype = np.float32 if floating_dtype is np.float64 else np.float64
    dist4 = MockContinuousDistribution(param1=10.0, param2=20.0, dtype=other_dtype)
    assert dist1 != dist4

    # Different type entirely
    assert dist1 != "not_a_distribution"


def test_continuous_dist_hash(floating_dtype: type[np.floating]) -> None:
    """Test hashing of distributions."""

    dist1 = MockContinuousDistribution(param1=10.0, param2=20.0, dtype=floating_dtype)
    dist2 = MockContinuousDistribution(param1=10.0, param2=20.0, dtype=floating_dtype)
    dist3 = MockContinuousDistribution(param1=15.0, param2=20.0, dtype=floating_dtype)

    assert hash(dist1) == hash(dist2)
    assert hash(dist1) != hash(dist3)


def test_continuous_dist_eq_different_subclasses(floating_dtype: type[np.floating]) -> None:
    """Test equality comparison between different subclasses of ContinuousDistribution."""

    dist1 = MockContinuousDistribution(param1=10.0, param2=20.0, dtype=floating_dtype)
    dist2 = MockInfLpdfContinuousDistribution(param1=10.0, param2=20.0, dtype=floating_dtype)

    assert dist1 != dist2
