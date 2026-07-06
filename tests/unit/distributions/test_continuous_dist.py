"""Tests for the base ContinuousDistribution class via a mock."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import copy

import numpy as np
import pytest
from tests.mocks.distributions.continuous_dist import (
    MockContinuousDistribution,
    MockInfLpdfContinuousDistribution,
)


def test_continuous_dist_fix_param() -> None:
    """Test fixing valid and invalid parameters."""

    dist = MockContinuousDistribution()
    assert dist.params_to_optimize == {"param1", "param2"}

    dist.fix_param("param1")
    assert dist.params_to_optimize == {"param2"}
    assert "param1" in dist._fixed_params

    with pytest.raises(ValueError, match="does not exist in this distribution"):
        dist.fix_param("invalid_param")


def test_continuous_dist_unfix_param() -> None:
    """Test unfixing parameters and ignoring unfixed ones."""

    dist = MockContinuousDistribution()
    dist.fix_param("param1")
    assert dist.params_to_optimize == {"param2"}

    dist.unfix_param("param1")
    assert dist.params_to_optimize == {"param1", "param2"}

    # Should not raise any errors
    dist.unfix_param("param1")
    dist.unfix_param("invalid_param")
    assert dist.params_to_optimize == {"param1", "param2"}


def test_continuous_dist_get_params_vector() -> None:
    """Test retrieving parameter values as a list in specified order."""

    dist = MockContinuousDistribution(param1=10.0, param2=20.0)

    vec = dist.get_params_vector(["param2", "param1"])
    np.testing.assert_allclose(vec, [20.0, 10.0], rtol=1e-5, atol=1e-8)

    with pytest.raises(ValueError, match="Invalid parameter names provided"):
        dist.get_params_vector(["param1", "invalid"])


def test_continuous_dist_copy() -> None:
    """Test creating a copy of the distribution."""

    dist = MockContinuousDistribution(param1=10.0, param2=20.0)
    dist.fix_param("param1")

    dist_copy = copy.copy(dist)

    assert dist_copy is not dist
    np.testing.assert_allclose(dist_copy.param1, dist.param1, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(dist_copy.param2, dist.param2, rtol=1e-5, atol=1e-8)
    assert "param1" in dist_copy._fixed_params


def test_continuous_dist_eq() -> None:
    """Test equality comparison between distributions."""

    dist1 = MockContinuousDistribution(param1=10.0, param2=20.0)
    dist2 = MockContinuousDistribution(param1=10.0, param2=20.0)

    assert dist1 == dist2

    # Different parameter values
    dist3 = MockContinuousDistribution(param1=15.0, param2=20.0)
    assert dist1 != dist3

    # Different type entirely
    assert dist1 != "not_a_distribution"


def test_continuous_dist_hash() -> None:
    """Test hashing of distributions."""

    dist1 = MockContinuousDistribution(param1=10.0, param2=20.0)
    dist2 = MockContinuousDistribution(param1=10.0, param2=20.0)
    dist3 = MockContinuousDistribution(param1=15.0, param2=20.0)

    assert hash(dist1) == hash(dist2)
    assert hash(dist1) != hash(dist3)


def test_continuous_dist_eq_different_subclasses() -> None:
    """Test equality comparison between different subclasses of ContinuousDistribution."""

    dist1 = MockContinuousDistribution(param1=10.0, param2=20.0)
    dist2 = MockInfLpdfContinuousDistribution(param1=10.0, param2=20.0)

    assert dist1 != dist2
