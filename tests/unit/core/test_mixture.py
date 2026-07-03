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
    assert_dtype,
    assert_probabilities_sum_to_one,
)
from tests.mocks.distributions.continuous_dist import (
    MockContinuousDistribution,
    MockInfLpdfContinuousDistribution,
)


def get_tol(dtype: type[np.floating]) -> tuple[float, float]:
    """Return appropriate rtol, atol based on floating precision."""
    if dtype == np.float16:
        return 1e-3, 1e-4
    return 1e-5, 1e-8


def test_init_default_weights(floating_dtype: type[np.floating]) -> None:
    """Test initialization with default equal weights."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)

    mixture = MixtureModel([comp1, comp2], dtype=floating_dtype)
    rtol, atol = get_tol(floating_dtype)

    expected_n_components = 2
    assert mixture.n_components == expected_n_components
    assert_dtype(mixture.weights, floating_dtype)
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.5, 0.5], rtol=rtol, atol=atol)


def test_init_custom_weights(floating_dtype: type[np.floating]) -> None:
    """Test initialization with custom valid weights."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)

    mixture = MixtureModel([comp1, comp2], weights=[0.2, 0.8], dtype=floating_dtype)
    rtol, atol = get_tol(floating_dtype)

    expected_n_components = 2
    assert mixture.n_components == expected_n_components
    assert_dtype(mixture.weights, floating_dtype)
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.2, 0.8], rtol=rtol, atol=atol)


def test_init_casts_components(floating_dtype: type[np.floating]) -> None:
    """Test that components passed with a different dtype are cast to mixture's dtype."""
    # Use the opposite dtype (float64 if float32, float32 if float64)
    other_dtype = np.float32 if floating_dtype == np.float64 else np.float64

    comp1 = MockContinuousDistribution(dtype=other_dtype)
    comp2 = MockContinuousDistribution(dtype=other_dtype)

    mixture = MixtureModel([comp1, comp2], dtype=floating_dtype)

    assert mixture.components[0].dtype == floating_dtype
    assert mixture.components[1].dtype == floating_dtype


def test_init_zero_weight(floating_dtype: type[np.floating]) -> None:
    """Test that exactly 0.0 weight does not crash logarithm."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)

    mixture = MixtureModel([comp1, comp2], weights=[0.0, 1.0], dtype=floating_dtype)
    rtol, atol = get_tol(floating_dtype)

    # 0.0 weight will become very small via np.finfo.tiny, but shouldn't crash
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.0, 1.0], rtol=rtol, atol=atol)


def test_log_weights_setter(floating_dtype: type[np.floating]) -> None:
    """Test that assigning new log-weights recalculates weights."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)

    mixture = MixtureModel([comp1, comp2], dtype=floating_dtype)

    # Set new log-weights that correspond to [0.2, 0.8]
    mixture.log_weights = np.array(np.log([0.2, 0.8]), dtype=floating_dtype)
    rtol, atol = get_tol(floating_dtype)

    assert_dtype(mixture.weights, floating_dtype)
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.2, 0.8], rtol=rtol, atol=atol)


def test_init_invalid_components(floating_dtype: type[np.floating]) -> None:
    """Test that empty component list raises ValueError."""
    with pytest.raises(ValueError, match="List of components cannot be empty"):
        MixtureModel([], dtype=floating_dtype)


def test_init_invalid_weights(floating_dtype: type[np.floating]) -> None:
    """Test that invalid weights raise ValueError."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)
    components = [comp1, comp2]

    # Negative weights
    with pytest.raises(ValueError, match="Weights must be positive"):
        MixtureModel(components, weights=[-0.1, 1.1], dtype=floating_dtype)

    # Weights sum not equal to 1
    with pytest.raises(ValueError, match="Sum of the weights must be equal 1"):
        MixtureModel(components, weights=[0.5, 0.6], dtype=floating_dtype)

    # Weights length mismatch
    with pytest.raises(ValueError, match="must be equal to weights number"):
        MixtureModel(components, weights=[1.0], dtype=floating_dtype)


def test_log_weights_setter_invalid_length(floating_dtype: type[np.floating]) -> None:
    """Test that setting log_weights with incorrect length raises ValueError."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1, comp2], dtype=floating_dtype)

    with pytest.raises(ValueError, match="length of the new logit vector does not match"):
        mixture.log_weights = np.array([0.0], dtype=floating_dtype)


def test_add_component(floating_dtype: type[np.floating]) -> None:
    """Test adding a valid component correctly updates components and weights."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1], dtype=floating_dtype)

    mixture.add_component(comp2, weight=0.5)
    rtol, atol = get_tol(floating_dtype)

    expected_n_components = 2
    assert mixture.n_components == expected_n_components
    assert mixture.components[1] is comp2

    assert_dtype(mixture.weights, floating_dtype)
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)

    # Originally [1.0], adding weight 0.5 renormalizes to [0.5, 0.5]
    assert_allclose(mixture.weights, [0.5, 0.5], rtol=rtol, atol=atol)


def test_add_component_invalid_weight(floating_dtype: type[np.floating]) -> None:
    """Test adding component with invalid weight bounds."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1], dtype=floating_dtype)

    with pytest.raises(ValueError, match="must be in the range"):
        mixture.add_component(comp2, weight=-0.1)

    with pytest.raises(ValueError, match="must be in the range"):
        mixture.add_component(comp2, weight=1.1)


def test_add_component_casts_type(floating_dtype: type[np.floating]) -> None:
    """Test adding a component casts its dtype if needed."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1], dtype=floating_dtype)

    other_dtype = np.float32 if floating_dtype == np.float64 else np.float64
    comp2 = MockContinuousDistribution(dtype=other_dtype)

    mixture.add_component(comp2, weight=0.5)

    assert mixture.components[1].dtype == floating_dtype


def test_remove_component(floating_dtype: type[np.floating]) -> None:
    """Test removing a component renormalizes remaining weights."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)
    comp3 = MockContinuousDistribution(dtype=floating_dtype)

    mixture = MixtureModel([comp1, comp2, comp3], weights=[0.2, 0.3, 0.5], dtype=floating_dtype)
    mixture.remove_component(1)
    rtol, atol = get_tol(floating_dtype)

    expected_n_components = 2
    assert mixture.n_components == expected_n_components
    assert mixture.components[0] is comp1
    assert mixture.components[1] is comp3

    # Remaining weights should be [0.2 / 0.7, 0.5 / 0.7] = [0.2857..., 0.7142...]
    assert_dtype(mixture.weights, floating_dtype)
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.2 / 0.7, 0.5 / 0.7], rtol=rtol, atol=atol)


def test_remove_last_component(floating_dtype: type[np.floating]) -> None:
    """Test removing the last component raises ValueError."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1], dtype=floating_dtype)

    with pytest.raises(ValueError, match="last component cannot be removed"):
        mixture.remove_component(0)


def test_remove_component_out_of_bounds(floating_dtype: type[np.floating]) -> None:
    """Test removing a component with invalid index raises IndexError."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1, comp2], dtype=floating_dtype)

    with pytest.raises(IndexError, match="out of range"):
        mixture.remove_component(2)


def test_pdf_and_lpdf(floating_dtype: type[np.floating]) -> None:
    """Test PDF and LPDF computations with scalar and array inputs."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1, comp2], weights=[0.4, 0.6], dtype=floating_dtype)
    rtol, atol = get_tol(floating_dtype)

    # Scalar input
    x_scalar = 1.0
    lpdf_scalar = mixture.lpdf(x_scalar)
    pdf_scalar = mixture.pdf(x_scalar)

    assert_allclose(pdf_scalar, np.exp(lpdf_scalar), rtol=rtol, atol=atol)
    assert isinstance(lpdf_scalar, floating_dtype) or np.isscalar(lpdf_scalar)

    # Array input
    x_array = np.array([1.0, 2.0, 3.0], dtype=floating_dtype)
    lpdf_array = mixture.lpdf(x_array)
    pdf_array = mixture.pdf(x_array)

    assert lpdf_array.shape == (3,)
    assert pdf_array.shape == (3,)
    assert_dtype(lpdf_array, floating_dtype)
    assert_allclose(pdf_array, np.exp(lpdf_array), rtol=rtol, atol=atol)


def test_lpdf_stability(floating_dtype: type[np.floating]) -> None:
    """Test LPDF stability and assert no NaN or positive inf."""
    comp1 = MockInfLpdfContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1, comp2], weights=[0.5, 0.5], dtype=floating_dtype)

    X = np.array([-1.0, 0.0, 1.0], dtype=floating_dtype)
    lpdf_vals = mixture.lpdf(X)

    # Assert there are no NaNs and no +inf
    assert_computational_stability(lpdf_vals)


def test_loglikelihood(floating_dtype: type[np.floating]) -> None:
    """Test log-likelihood computes the sum of lpdf correctly."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    comp2 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1, comp2], weights=[0.5, 0.5], dtype=floating_dtype)

    X = np.array([1.0, 2.5, 3.1], dtype=floating_dtype)

    ll = mixture.loglikelihood(X)
    expected_ll = np.sum(mixture.lpdf(X))

    rtol, atol = get_tol(floating_dtype)
    assert_allclose(ll, expected_ll, rtol=rtol, atol=atol)


def test_generate(floating_dtype: type[np.floating]) -> None:
    """Test random sample generation."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1], dtype=floating_dtype)

    # size=None
    sample_scalar = mixture.generate(size=None)
    assert isinstance(sample_scalar, floating_dtype) or np.isscalar(sample_scalar)

    # size=int
    sample_1d = mixture.generate(size=5)
    assert sample_1d.shape == (5,)
    assert_dtype(sample_1d, floating_dtype)

    # size=tuple
    sample_2d = mixture.generate(size=(3, 2))
    assert sample_2d.shape == (3, 2)
    assert_dtype(sample_2d, floating_dtype)


def test_generate_zero_size(floating_dtype: type[np.floating]) -> None:
    """Test generation with zero sizes."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1], dtype=floating_dtype)

    res_int = mixture.generate(size=0)
    assert res_int.shape == (0,)
    assert_dtype(res_int, floating_dtype)

    res_tuple = mixture.generate(size=(0, 2))
    assert res_tuple.shape == (0, 2)
    assert_dtype(res_tuple, floating_dtype)


def test_astype(floating_dtype: type[np.floating]) -> None:
    """Test astype method converts dtype of mixture and components."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1], dtype=floating_dtype)

    target_dtype = np.float32 if floating_dtype != np.float32 else np.float64

    new_mixture = mixture.astype(target_dtype)
    assert new_mixture is not mixture
    assert new_mixture.dtype == target_dtype
    assert new_mixture.components[0].dtype == target_dtype

    same_mixture = mixture.astype(floating_dtype)
    assert same_mixture is mixture


def test_dunder_methods(floating_dtype: type[np.floating]) -> None:
    """Test __getitem__, __iter__, __copy__, __eq__ and __hash__."""
    comp1 = MockContinuousDistribution(param1=1.0, dtype=floating_dtype)
    comp2 = MockContinuousDistribution(param1=2.0, dtype=floating_dtype)
    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5], dtype=floating_dtype)

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


def test_eq_unrelated_type(floating_dtype: type[np.floating]) -> None:
    """Test equality with an unrelated type."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1], dtype=floating_dtype)

    assert mixture != "a string"


def test_eq_different_length(floating_dtype: type[np.floating]) -> None:
    """Test equality returns False when number of components differs."""
    comp1 = MockContinuousDistribution(param1=1.0, dtype=floating_dtype)
    comp2 = MockContinuousDistribution(param1=2.0, dtype=floating_dtype)
    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5], dtype=floating_dtype)
    mixture2 = MixtureModel([comp1], dtype=floating_dtype)

    assert mixture1 != mixture2


def test_eq_different_dtype(floating_dtype: type[np.floating]) -> None:
    """Test equality returns False when mixture dtypes differ."""
    other_dtype = np.float32 if floating_dtype == np.float64 else np.float64

    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    mixture1 = MixtureModel([comp1], dtype=floating_dtype)

    comp1_other = MockContinuousDistribution(dtype=other_dtype)
    mixture2 = MixtureModel([comp1_other], dtype=other_dtype)

    assert mixture1 != mixture2


def test_eq_different_components(floating_dtype: type[np.floating]) -> None:
    """Test equality returns False when components differ."""
    comp1 = MockContinuousDistribution(param1=1.0, dtype=floating_dtype)
    comp2 = MockContinuousDistribution(param1=2.0, dtype=floating_dtype)
    comp3 = MockContinuousDistribution(param1=3.0, dtype=floating_dtype)

    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5], dtype=floating_dtype)
    mixture2 = MixtureModel([comp1, comp3], weights=[0.5, 0.5], dtype=floating_dtype)

    assert mixture1 != mixture2


def test_eq_different_weights(floating_dtype: type[np.floating]) -> None:
    """Test equality returns False when weights differ."""
    comp1 = MockContinuousDistribution(param1=1.0, dtype=floating_dtype)
    comp2 = MockContinuousDistribution(param1=2.0, dtype=floating_dtype)

    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5], dtype=floating_dtype)
    mixture2 = MixtureModel([comp1, comp2], weights=[0.2, 0.8], dtype=floating_dtype)

    assert mixture1 != mixture2


def test_eq_cache_usage(floating_dtype: type[np.floating]) -> None:
    """Test that multiple equality checks utilize the sorted pairs cache correctly."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    mixture1 = MixtureModel([comp1], dtype=floating_dtype)
    mixture2 = MixtureModel([comp1], dtype=floating_dtype)

    # First check builds the cache
    assert mixture1 == mixture2
    # Second check uses the _sorted_pairs_cache logic flow
    assert mixture1 == mixture2


def test_hash_precision(floating_dtype: type[np.floating]) -> None:
    """Test hashing precision works without crashing."""
    comp1 = MockContinuousDistribution(dtype=floating_dtype)
    mixture = MixtureModel([comp1], dtype=floating_dtype)

    # Basic check that hash executes
    assert isinstance(hash(mixture), int)
