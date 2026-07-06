"""Unit tests for MixtureModel core class."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from copy import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_mpest.core.mixture import MixtureModel
from tests.helpers.math_assertions import (
    assert_computational_stability,
    assert_probabilities_sum_to_one,
)
from tests.mocks.distributions.continuous_dist import (
    MockContinuousDistribution,
    MockInfLpdfContinuousDistribution,
)


def test_init_default_weights() -> None:
    """Test initialization with default equal weights."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()

    mixture = MixtureModel([comp1, comp2])
    rtol, atol = 1e-5, 1e-8

    expected_n_components = 2
    assert mixture.n_components == expected_n_components
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.5, 0.5], rtol=rtol, atol=atol)


def test_init_custom_weights() -> None:
    """Test initialization with custom valid weights."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()

    mixture = MixtureModel([comp1, comp2], weights=[0.2, 0.8])
    rtol, atol = 1e-5, 1e-8

    expected_n_components = 2
    assert mixture.n_components == expected_n_components
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.2, 0.8], rtol=rtol, atol=atol)


def test_init_zero_weight() -> None:
    """Test that exactly 0.0 weight does not crash logarithm."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()

    mixture = MixtureModel([comp1, comp2], weights=[0.0, 1.0])
    rtol, atol = 1e-5, 1e-8

    # 0.0 weight will become very small via np.finfo.tiny, but shouldn't crash
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.0, 1.0], rtol=rtol, atol=atol)


def test_log_weights_setter() -> None:
    """Test that assigning new log-weights recalculates weights."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()

    mixture = MixtureModel([comp1, comp2])

    # Set new log-weights that correspond to [0.2, 0.8]
    mixture.log_weights = np.array(np.log([0.2, 0.8]))
    rtol, atol = 1e-5, 1e-8

    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.2, 0.8], rtol=rtol, atol=atol)


def test_init_invalid_components() -> None:
    """Test that empty component list raises ValueError."""
    with pytest.raises(ValueError, match="List of components cannot be empty"):
        MixtureModel([])


def test_init_invalid_weights() -> None:
    """Test that invalid weights raise ValueError."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()
    components = [comp1, comp2]

    # Negative weights
    with pytest.raises(ValueError, match="Weights must be positive"):
        MixtureModel(components, weights=[-0.1, 1.1])

    # Weights sum not equal to 1
    with pytest.raises(ValueError, match="Sum of the weights must be equal 1"):
        MixtureModel(components, weights=[0.5, 0.6])

    # Weights length mismatch
    with pytest.raises(ValueError, match="must be equal to weights number"):
        MixtureModel(components, weights=[1.0])


def test_log_weights_setter_invalid_length() -> None:
    """Test that setting log_weights with incorrect length raises ValueError."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()
    mixture = MixtureModel([comp1, comp2])

    with pytest.raises(ValueError, match="length of the new logit vector does not match"):
        mixture.log_weights = np.array([0.0])


def test_add_component() -> None:
    """Test adding a valid component correctly updates components and weights."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()
    mixture = MixtureModel([comp1])

    mixture.add_component(comp2, weight=0.5)
    rtol, atol = 1e-5, 1e-8

    expected_n_components = 2
    assert mixture.n_components == expected_n_components
    assert mixture.components[1] == comp2

    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)

    # Originally [1.0], adding weight 0.5 renormalizes to [0.5, 0.5]
    assert_allclose(mixture.weights, [0.5, 0.5], rtol=rtol, atol=atol)


def test_set_component() -> None:
    """Test replacing a component at a specific index."""
    comp1 = MockContinuousDistribution(param1=1.0)
    comp2 = MockContinuousDistribution(param1=2.0)
    comp3 = MockContinuousDistribution(param1=3.0)

    expected_n_components = 2

    mixture = MixtureModel([comp1, comp2], weights=[0.4, 0.6])

    # We first access __eq__ or hash to populate the cache
    _ = mixture == mixture  # noqa: PLR0124
    assert mixture._sorted_pairs_cache is not None

    mixture.set_component(comp3, 1)

    assert mixture.n_components == expected_n_components
    assert mixture.components[0] == comp1
    assert mixture.components[1] == comp3
    assert mixture.components[1] is not comp3  # Since copy() should be used

    # Verify that the sorted pairs cache was invalidated
    assert mixture._sorted_pairs_cache is None


def test_add_component_invalid_weight() -> None:
    """Test adding component with invalid weight bounds."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()
    mixture = MixtureModel([comp1])

    with pytest.raises(ValueError, match="must be in the range"):
        mixture.add_component(comp2, weight=-0.1)

    with pytest.raises(ValueError, match="must be in the range"):
        mixture.add_component(comp2, weight=1.1)


def test_remove_component() -> None:
    """Test removing a component renormalizes remaining weights."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()
    comp3 = MockContinuousDistribution()

    mixture = MixtureModel([comp1, comp2, comp3], weights=[0.2, 0.3, 0.5])
    mixture.remove_component(1)
    rtol, atol = 1e-5, 1e-8

    expected_n_components = 2
    assert mixture.n_components == expected_n_components
    assert mixture.components[0] is comp1
    assert mixture.components[1] is comp3

    # Remaining weights should be [0.2 / 0.7, 0.5 / 0.7] = [0.2857..., 0.7142...]
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.2 / 0.7, 0.5 / 0.7], rtol=rtol, atol=atol)


def test_remove_last_component() -> None:
    """Test removing the last component raises ValueError."""
    comp1 = MockContinuousDistribution()
    mixture = MixtureModel([comp1])

    with pytest.raises(ValueError, match="last component cannot be removed"):
        mixture.remove_component(0)


def test_remove_component_out_of_bounds() -> None:
    """Test removing a component with invalid index raises IndexError."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()
    mixture = MixtureModel([comp1, comp2])

    with pytest.raises(IndexError, match="out of range"):
        mixture.remove_component(2)


def test_pdf_and_lpdf() -> None:
    """Test PDF and LPDF computations with scalar and array inputs."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()
    mixture = MixtureModel([comp1, comp2], weights=[0.4, 0.6])
    rtol, atol = 1e-5, 1e-8

    # Scalar input
    x_scalar = 1.0
    lpdf_scalar = mixture.lpdf(x_scalar)
    pdf_scalar = mixture.pdf(x_scalar)

    assert_allclose(pdf_scalar, np.exp(lpdf_scalar), rtol=rtol, atol=atol)

    # Array input
    x_array = np.array([1.0, 2.0, 3.0])
    lpdf_array = mixture.lpdf(x_array)
    pdf_array = mixture.pdf(x_array)

    assert lpdf_array.shape == (3,)
    assert pdf_array.shape == (3,)
    assert_allclose(pdf_array, np.exp(lpdf_array), rtol=rtol, atol=atol)


def test_lpdf_stability() -> None:
    """Test LPDF stability and assert no NaN or positive inf."""
    comp1 = MockInfLpdfContinuousDistribution()
    comp2 = MockContinuousDistribution()
    mixture = MixtureModel([comp1, comp2], weights=[0.5, 0.5])

    X = np.array([-1.0, 0.0, 1.0])
    lpdf_vals = mixture.lpdf(X)

    # Assert there are no NaNs and no +inf
    assert_computational_stability(lpdf_vals)


def test_loglikelihood() -> None:
    """Test log-likelihood computes the sum of lpdf correctly."""
    comp1 = MockContinuousDistribution()
    comp2 = MockContinuousDistribution()
    mixture = MixtureModel([comp1, comp2], weights=[0.5, 0.5])

    X = np.array([1.0, 2.5, 3.1])

    ll = mixture.loglikelihood(X)
    expected_ll = np.sum(mixture.lpdf(X))

    rtol, atol = 1e-5, 1e-8
    assert_allclose(ll, expected_ll, rtol=rtol, atol=atol)


def test_generate() -> None:
    """Test random sample generation."""
    comp1 = MockContinuousDistribution()
    mixture = MixtureModel([comp1])

    # size=None
    sample_scalar = mixture.generate(size=None)
    assert np.isscalar(sample_scalar)

    # size=int
    sample_1d = mixture.generate(size=5)
    assert sample_1d.shape == (5,)

    # size=tuple
    sample_2d = mixture.generate(size=(3, 2))
    assert sample_2d.shape == (3, 2)


def test_generate_zero_size() -> None:
    """Test generation with zero sizes."""
    comp1 = MockContinuousDistribution()
    mixture = MixtureModel([comp1])

    res_int = mixture.generate(size=0)
    assert res_int.shape == (0,)

    res_tuple = mixture.generate(size=(0, 2))
    assert res_tuple.shape == (0, 2)


def test_dunder_methods() -> None:
    """Test __getitem__, __iter__, __copy__, __eq__ and __hash__."""
    comp1 = MockContinuousDistribution(param1=1.0)
    comp2 = MockContinuousDistribution(param1=2.0)
    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])

    # __getitem__
    assert mixture1[0] is comp1
    assert mixture1[1] is comp2

    # __iter__
    comps = list(mixture1)
    assert comps == [comp1, comp2]

    # __copy__
    mixture2 = copy(mixture1)
    assert mixture2 is not mixture1
    assert mixture2.components[0] is not comp1
    assert mixture2.weights is not mixture1.weights

    # __eq__ and __hash__
    assert mixture1 == mixture2
    assert hash(mixture1) == hash(mixture2)


def test_eq_unrelated_type() -> None:
    """Test equality with an unrelated type."""
    comp1 = MockContinuousDistribution()
    mixture = MixtureModel([comp1])

    assert mixture != "a string"


def test_eq_different_length() -> None:
    """Test equality returns False when number of components differs."""
    comp1 = MockContinuousDistribution(param1=1.0)
    comp2 = MockContinuousDistribution(param1=2.0)
    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
    mixture2 = MixtureModel([comp1])

    assert mixture1 != mixture2


def test_eq_different_components() -> None:
    """Test equality returns False when components differ."""
    comp1 = MockContinuousDistribution(param1=1.0)
    comp2 = MockContinuousDistribution(param1=2.0)
    comp3 = MockContinuousDistribution(param1=3.0)

    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
    mixture2 = MixtureModel([comp1, comp3], weights=[0.5, 0.5])

    assert mixture1 != mixture2


def test_eq_different_weights() -> None:
    """Test equality returns False when weights differ."""
    comp1 = MockContinuousDistribution(param1=1.0)
    comp2 = MockContinuousDistribution(param1=2.0)

    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
    mixture2 = MixtureModel([comp1, comp2], weights=[0.2, 0.8])

    assert mixture1 != mixture2


def test_eq_cache_usage() -> None:
    """Test that multiple equality checks utilize the sorted pairs cache correctly."""
    comp1 = MockContinuousDistribution()
    mixture1 = MixtureModel([comp1])
    mixture2 = MixtureModel([comp1])

    # First check builds the cache
    assert mixture1 == mixture2
    # Second check uses the _sorted_pairs_cache logic flow
    assert mixture1 == mixture2


def test_hash_precision() -> None:
    """Test hashing precision works without crashing."""
    comp1 = MockContinuousDistribution()
    mixture = MixtureModel([comp1])

    # Basic check that hash executes
    assert isinstance(hash(mixture), int)
