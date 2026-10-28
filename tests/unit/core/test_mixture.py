"""Unit tests for MixtureModel core class."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterator
from copy import copy
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.transformations.operations.distributions.mixture import FiniteMixtureDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core.mixture import MixtureModel
from tests.helpers.families import family
from tests.helpers.math_assertions import (
    assert_computational_stability,
    assert_is_array_type,
    assert_is_scalar_type,
    assert_probabilities_sum_to_one,
)
from tests.mocks.distributions import (
    MockConstantLpdfContinuousDistribution,
    MockContinuousDistribution,
    MockInfLpdfContinuousDistribution,
)


@pytest.fixture
def mock_comp() -> ParametricFamilyDistribution:
    """Fixture for a single mock distribution."""
    return cast(ParametricFamilyDistribution, MockContinuousDistribution())


@pytest.fixture
def mock_comps() -> tuple[ParametricFamilyDistribution, ParametricFamilyDistribution]:
    """Fixture for two mock distributions."""
    return (
        cast(ParametricFamilyDistribution, MockContinuousDistribution()),
        cast(ParametricFamilyDistribution, MockContinuousDistribution()),
    )


@pytest.fixture
def param_comps() -> tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution]:
    """Fixture for three differently parametrized mock distributions."""
    return (
        cast(ParametricFamilyDistribution, MockContinuousDistribution(param1=1.0)),
        cast(ParametricFamilyDistribution, MockContinuousDistribution(param1=2.0)),
        cast(ParametricFamilyDistribution, MockContinuousDistribution(param1=3.0)),
    )


@pytest.fixture
def mock_finite_mixture() -> Iterator[MagicMock]:
    """Observe construction of core mixtures without constructing one."""
    with patch("pysatl_mpest.core.mixture.finite_mixture") as mock_fm:
        yield mock_fm


def test_init_default_weights(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test initialization with default equal weights."""
    mixture = MixtureModel(mock_comps)
    rtol, atol = 1e-5, 1e-8

    expected_components = 2
    assert mixture.n_components == expected_components
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.5, 0.5], rtol=rtol, atol=atol)


def test_init_custom_weights(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test initialization with custom valid weights."""
    mixture = MixtureModel(mock_comps, weights=[0.2, 0.8])
    rtol, atol = 1e-5, 1e-8

    expected_components = 2
    assert mixture.n_components == expected_components
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.2, 0.8], rtol=rtol, atol=atol)


def test_init_zero_weight(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test that exactly 0.0 weight does not crash logarithm."""
    mixture = MixtureModel(mock_comps, weights=[0.0, 1.0])
    rtol, atol = 1e-5, 1e-8

    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.0, 1.0], rtol=rtol, atol=atol)


def test_log_weights_setter(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test that assigning new log-weights recalculates weights."""
    mixture = MixtureModel(mock_comps)
    mixture.log_weights = np.array(np.log([0.2, 0.8]))
    rtol, atol = 1e-5, 1e-8

    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.2, 0.8], rtol=rtol, atol=atol)


def test_init_invalid_components() -> None:
    """Test that empty component list raises ValueError."""
    with pytest.raises(ValueError, match=r"^List of components cannot be empty$"):
        MixtureModel([])


def test_init_invalid_weights(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test that invalid weights raise ValueError."""
    with pytest.raises(ValueError, match=r"^Weights must be positive\.$"):
        MixtureModel(mock_comps, weights=[-0.1, 1.1])

    with pytest.raises(ValueError, match=r"^Sum of the weights must be equal 1, but it equal 1\.1\.$"):
        MixtureModel(mock_comps, weights=[0.5, 0.6])

    with pytest.raises(ValueError, match="must be equal to weights number"):
        MixtureModel(mock_comps, weights=[1.0])


def test_log_weights_setter_invalid_length(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test that setting log_weights with incorrect length raises ValueError."""
    mixture = MixtureModel(mock_comps)
    with pytest.raises(ValueError, match="length of the new log-weights vector"):
        mixture.log_weights = np.array([0.0])


def test_add_component(mock_comp: ParametricFamilyDistribution) -> None:
    """Test adding a valid component correctly updates components and weights."""
    mixture = MixtureModel([mock_comp])
    comp2 = cast(ParametricFamilyDistribution, MockContinuousDistribution())
    mixture.add_component(comp2, weight=0.5)
    rtol, atol = 1e-5, 1e-8

    expected_components = 2
    assert mixture.n_components == expected_components
    assert mixture.components[1] == comp2
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.5, 0.5], rtol=rtol, atol=atol)


def test_setitem(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    """Test replacing a component at a specific index."""
    comp1, comp2, comp3 = param_comps
    mixture = MixtureModel([comp1, comp2], weights=[0.4, 0.6])

    _ = mixture == mixture  # noqa: PLR0124
    assert mixture._sorted_pairs_cache is not None

    mixture[1] = comp3

    expected_components = 2
    assert mixture.n_components == expected_components
    assert mixture.components[0] == comp1
    assert mixture.components[1] == comp3
    assert mixture.components[1] is comp3

    assert mixture._sorted_pairs_cache is None


def test_add_component_invalid_weight(mock_comp: ParametricFamilyDistribution) -> None:
    """Test adding component with invalid weight bounds."""
    mixture = MixtureModel([mock_comp])
    comp2 = cast(ParametricFamilyDistribution, MockContinuousDistribution())

    message = r"^The weight of the new component must be in the range \(0, 1\)\.$"
    # The bounds 0 and 1 themselves are not allowed either.
    for weight in (-0.1, 0.0, 1.0, 1.1):
        with pytest.raises(ValueError, match=message):
            mixture.add_component(comp2, weight=weight)


def test_remove_component(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    """Test removing a component renormalizes remaining weights."""
    comp1, comp2, comp3 = param_comps
    mixture = MixtureModel([comp1, comp2, comp3], weights=[0.2, 0.3, 0.5])
    mixture.remove_component(1)
    rtol, atol = 1e-5, 1e-8

    expected_components = 2
    assert mixture.n_components == expected_components
    assert mixture.components[0] is comp1
    assert mixture.components[1] is comp3
    assert_probabilities_sum_to_one(mixture.weights, rtol=rtol, atol=atol)
    assert_allclose(mixture.weights, [0.2 / 0.7, 0.5 / 0.7], rtol=rtol, atol=atol)


def test_remove_last_component(mock_comp: ParametricFamilyDistribution) -> None:
    """Test removing the last component raises ValueError."""
    mixture = MixtureModel([mock_comp])
    with pytest.raises(ValueError, match=r"^The last component cannot be removed from the model\.$"):
        mixture.remove_component(0)


def test_remove_component_out_of_bounds(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test removing a component with invalid index raises IndexError."""
    mixture = MixtureModel(mock_comps)
    with pytest.raises(IndexError, match=r"^Index 2 out of range \[0, 1\]$"):
        mixture.remove_component(2)


def test_pdf_and_lpdf(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test PDF and LPDF computations with scalar and array inputs."""
    mixture = MixtureModel(mock_comps, weights=[0.4, 0.6])
    rtol, atol = 1e-5, 1e-8

    x_scalar = 1.0
    lpdf_scalar = mixture.lpdf(x_scalar)
    pdf_scalar = mixture.pdf(x_scalar)
    assert_allclose(pdf_scalar, np.exp(lpdf_scalar), rtol=rtol, atol=atol)

    x_array = np.array([1.0, 2.0, 3.0])
    lpdf_array = mixture.lpdf(x_array)
    pdf_array = mixture.pdf(x_array)

    assert lpdf_array.shape == (3,)
    assert pdf_array.shape == (3,)
    assert_allclose(pdf_array, np.exp(lpdf_array), rtol=rtol, atol=atol)


def test_lpdf_stability() -> None:
    """Test LPDF stability and assert no NaN or positive inf."""
    comp1 = cast(ParametricFamilyDistribution, MockInfLpdfContinuousDistribution())
    comp2 = cast(ParametricFamilyDistribution, MockContinuousDistribution())
    mixture = MixtureModel([comp1, comp2], weights=[0.5, 0.5])

    X = np.array([-1.0, 0.0, 1.0])
    lpdf_vals = mixture.lpdf(X)
    assert_computational_stability(cast(np.ndarray, lpdf_vals))


def test_lpdf_does_not_use_core_mixture(
    mock_comps: tuple[ParametricFamilyDistribution, ...], mock_finite_mixture: MagicMock
) -> None:
    """Test LPDF is computed by the model itself, not by the pysatl-core mixture."""
    mixture = MixtureModel(mock_comps)

    mixture.lpdf(np.array([1.0, 2.0]))

    mock_finite_mixture.assert_not_called()


def test_lpdf_is_weighted_logsumexp() -> None:
    """Test LPDF applies the mixture weights to the component log-densities."""
    comps = (
        cast(ParametricFamilyDistribution, MockConstantLpdfContinuousDistribution(-1.0)),
        cast(ParametricFamilyDistribution, MockConstantLpdfContinuousDistribution(-3.0)),
    )
    mixture = MixtureModel(comps, weights=[0.25, 0.75])
    expected = np.log(0.25 * np.exp(-1.0) + 0.75 * np.exp(-3.0))
    rtol, atol = 1e-12, 1e-12

    assert_allclose(mixture.lpdf(np.array([0.0, 1.0])), [expected, expected], rtol=rtol, atol=atol)
    assert_allclose(mixture.lpdf(0.5), expected, rtol=rtol, atol=atol)


@pytest.mark.filterwarnings("error")
def test_lpdf_all_components_neg_inf() -> None:
    """Test LPDF is -inf, without NaN or warnings, where every component has zero density."""
    comps = (
        cast(ParametricFamilyDistribution, MockInfLpdfContinuousDistribution()),
        cast(ParametricFamilyDistribution, MockInfLpdfContinuousDistribution()),
    )
    mixture = MixtureModel(comps)

    assert np.all(np.isneginf(mixture.lpdf(np.array([-1.0, 0.0, 1.0]))))


def test_component_lpdf_matrix_shape(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test the component log-density matrix has the component axis last."""
    mixture = MixtureModel(mock_comps)
    n_components = 2

    assert mixture.component_lpdf_matrix(np.zeros(5)).shape == (5, n_components)
    assert mixture.component_lpdf_matrix(0.5).shape == (n_components,)
    assert mixture.component_lpdf_matrix(np.zeros((3, 4))).shape == (3, 4, n_components)


def test_component_lpdf_matrix_columns_are_contiguous(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test each component's log-densities are contiguous in memory.

    Reducing the short component axis (``logsumexp(axis=-1)`` in the E-step) is several times slower
    on a row-major ``(n, k)`` matrix than on one whose columns are contiguous.
    """
    mixture = MixtureModel(mock_comps)

    matrix = mixture.component_lpdf_matrix(np.zeros(5))

    assert all(matrix[:, j].flags.c_contiguous for j in range(matrix.shape[1]))


def test_component_lpdf_matrix_values_exclude_weights() -> None:
    """Test the matrix holds raw component log-densities, one column per component."""
    comps = (
        cast(ParametricFamilyDistribution, MockConstantLpdfContinuousDistribution(-1.0)),
        cast(ParametricFamilyDistribution, MockConstantLpdfContinuousDistribution(-3.0)),
    )
    mixture = MixtureModel(comps, weights=[0.25, 0.75])

    matrix = mixture.component_lpdf_matrix(np.array([0.0, 1.0, 2.0]))

    assert_allclose(matrix, np.tile([-1.0, -3.0], (3, 1)))


def test_loglikelihood(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test log-likelihood computes the sum of lpdf correctly."""
    mixture = MixtureModel(mock_comps, weights=[0.5, 0.5])
    X = np.array([1.0, 2.5, 3.1])

    ll = mixture.loglikelihood(X)
    expected_ll = float(np.sum(mixture.lpdf(X)))
    rtol, atol = 1e-5, 1e-8
    assert_allclose(ll, expected_ll, rtol=rtol, atol=atol)


def test_generate(mock_comp: ParametricFamilyDistribution) -> None:
    """Test random sample generation."""
    mixture = MixtureModel([mock_comp])

    sample_scalar = mixture.generate(size=None)
    assert np.isscalar(sample_scalar)

    sample_1d = mixture.generate(size=5)
    assert cast(np.ndarray, sample_1d).shape == (5,)

    sample_2d = mixture.generate(size=(3, 2))
    assert cast(np.ndarray, sample_2d).shape == (3, 2)


def test_generate_is_reproducible_with_seed(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test the same seed gives the same sample, and a different seed gives a different one."""
    mixture = MixtureModel(mock_comps)
    seed, other_seed, size = 7, 8, 20

    first = cast(np.ndarray, mixture.generate(size=size, random_state=seed))
    second = cast(np.ndarray, mixture.generate(size=size, random_state=seed))
    other = cast(np.ndarray, mixture.generate(size=size, random_state=other_seed))

    assert_allclose(first, second)
    assert not np.allclose(first, other)


def test_generate_zero_size(mock_comp: ParametricFamilyDistribution) -> None:
    """Test generation with zero sizes."""
    mixture = MixtureModel([mock_comp])

    res_int = mixture.generate(size=0)
    assert cast(np.ndarray, res_int).shape == (0,)

    res_tuple = mixture.generate(size=(0, 2))
    assert cast(np.ndarray, res_tuple).shape == (0, 2)


def test_dunder_methods(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    """Test __getitem__, __iter__, __copy__, __eq__ and __hash__."""
    comp1, comp2, _ = param_comps
    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])

    assert mixture1[0] is comp1
    assert mixture1[1] is comp2

    comps = list(mixture1)
    assert comps == [comp1, comp2]

    mixture2 = copy(mixture1)
    assert mixture2 is not mixture1
    assert mixture2.components[0] is comp1
    assert mixture2.weights is not mixture1.weights

    assert mixture1 == mixture2
    assert hash(mixture1) == hash(mixture2)


def test_eq_unrelated_type(mock_comp: ParametricFamilyDistribution) -> None:
    """Test equality with an unrelated type."""
    mixture = MixtureModel([mock_comp])
    assert mixture != "a string"


def test_eq_different_length(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    """Test equality returns False when number of components differs."""
    comp1, comp2, _ = param_comps
    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
    mixture2 = MixtureModel([comp1])
    assert mixture1 != mixture2


def test_eq_different_length_with_zero_weight(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    """Test a zero-weight extra component does not make mixtures equal: the number of components must match."""
    comp1, comp2, _ = param_comps
    with_extra = MixtureModel([comp1, comp2], weights=[1.0, 0.0])
    without_extra = MixtureModel([comp1])
    assert with_extra != without_extra
    assert without_extra != with_extra


def test_eq_is_independent_of_component_order(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    """Test mixtures with the same (component, weight) pairs in a different order are equal."""
    comp1, comp2, _ = param_comps
    mixture1 = MixtureModel([comp1, comp2], weights=[0.2, 0.8])
    mixture2 = MixtureModel([comp2, comp1], weights=[0.8, 0.2])
    assert mixture1 == mixture2
    assert hash(mixture1) == hash(mixture2)


def test_eq_equal_components_with_swapped_weights(mock_comp: ParametricFamilyDistribution) -> None:
    """Test identical components with swapped weights give equal mixtures."""
    other = cast(ParametricFamilyDistribution, MockContinuousDistribution())
    mixture1 = MixtureModel([mock_comp, other], weights=[0.2, 0.8])
    mixture2 = MixtureModel([mock_comp, other], weights=[0.8, 0.2])
    assert mixture1 == mixture2
    assert hash(mixture1) == hash(mixture2)


def test_eq_different_components(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    """Test equality returns False when components differ."""
    comp1, comp2, comp3 = param_comps
    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
    mixture2 = MixtureModel([comp1, comp3], weights=[0.5, 0.5])
    assert mixture1 != mixture2


def test_eq_different_weights(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    """Test equality returns False when weights differ."""
    comp1, comp2, _ = param_comps
    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
    mixture2 = MixtureModel([comp1, comp2], weights=[0.2, 0.8])
    assert mixture1 != mixture2


def test_eq_cache_usage(mock_comp: ParametricFamilyDistribution) -> None:
    """Test that multiple equality checks utilize the sorted pairs cache correctly."""
    mixture1 = MixtureModel([mock_comp])
    mixture2 = MixtureModel([mock_comp])
    assert mixture1 == mixture2
    assert mixture1 == mixture2


def test_hash_agrees_with_eq_for_weights_within_tolerance(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    """Test mixtures whose weights differ only by rounding are equal and have equal hashes."""
    comp1, comp2, _ = param_comps
    mixture1 = MixtureModel([comp1, comp2], weights=[0.3, 0.7])
    mixture2 = MixtureModel([comp1, comp2], weights=[0.3, 0.7])
    mixture2.log_weights = mixture1.log_weights + np.array([1e-9, 0.0])

    assert mixture1 == mixture2
    assert hash(mixture1) == hash(mixture2)


def test_hash_does_not_depend_on_weights(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    """Test the hash ignores weights: they are compared with a tolerance, so they cannot be hashed consistently."""
    comp1, comp2, _ = param_comps
    mixture1 = MixtureModel([comp1, comp2], weights=[0.5, 0.5])
    mixture2 = MixtureModel([comp1, comp2], weights=[0.2, 0.8])

    assert mixture1 != mixture2
    assert hash(mixture1) == hash(mixture2)


def test_hash_differs_for_different_components(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    """Test mixtures of different components get different hashes."""
    comp1, comp2, comp3 = param_comps

    assert hash(MixtureModel([comp1, comp2])) != hash(MixtureModel([comp1, comp3]))


def test_log_weights_getter(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test that log_weights getter returns correct log weights."""

    mixture = MixtureModel(mock_comps, weights=[0.3, 0.7])
    expected = np.log(np.array([0.3, 0.7], dtype=np.float64) + np.finfo(np.float64).tiny)
    assert_allclose(mixture.log_weights, expected)


def test_setitem_out_of_bounds(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Test that replacing a component with invalid index raises IndexError."""

    mixture = MixtureModel(mock_comps)
    comp = cast(ParametricFamilyDistribution, MockContinuousDistribution())
    # The index equal to the number of components is the closest invalid one.
    for invalid_index in (len(mock_comps), 99):
        with pytest.raises(IndexError, match=r"^Index of component must be between 0 and 1$"):
            mixture[invalid_index] = comp


# --- Details found by mutation testing ---


def test_default_log_weights_are_normalized() -> None:
    """Without weights the components are equally likely, and the log-weights are those of the equal weights."""
    components = [cast(ParametricFamilyDistribution, MockContinuousDistribution()) for _ in range(3)]

    mixture = MixtureModel(components)

    assert_allclose(mixture.log_weights, np.log(np.full(3, 1.0 / 3.0)))


def test_add_component_keeps_the_proportions_of_several_components(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    comp1, comp2, comp3 = param_comps
    mixture = MixtureModel([comp1, comp2], weights=[0.4, 0.6])
    mixture.add_component(comp3, weight=0.5)

    assert_allclose(mixture.weights, [0.2, 0.3, 0.5])
    assert_allclose(mixture.log_weights, np.log([0.2, 0.3, 0.5]))


def test_remove_component_keeps_the_log_weights_normalized(
    param_comps: tuple[ParametricFamilyDistribution, ParametricFamilyDistribution, ParametricFamilyDistribution],
) -> None:
    mixture = MixtureModel(list(param_comps), weights=[0.2, 0.3, 0.5])

    mixture.remove_component(1)

    assert_allclose(mixture.log_weights, np.log([0.2 / 0.7, 0.5 / 0.7]))


def test_pdf_returns_float64_scalar_for_a_scalar() -> None:
    mixture = MixtureModel([cast(ParametricFamilyDistribution, MockContinuousDistribution())])

    assert_is_scalar_type(mixture.pdf(1.0))
    assert_is_array_type(mixture.pdf(np.array([1.0, 2.0])), (2,))


@pytest.mark.parametrize("data", [[1, 2, 3], np.array([1, 2, 3], dtype=np.int32)])
def test_pdf_returns_float64_for_integer_input(data: ArrayLike) -> None:
    """PDF evaluations return float64 arrays for lists and integer arrays."""
    mixture = MixtureModel([cast(ParametricFamilyDistribution, MockContinuousDistribution())])

    assert_is_array_type(mixture.pdf(data), (3,))


def test_log_weights_setter_normalizes_unnormalized_input(
    mock_comps: tuple[ParametricFamilyDistribution, ...],
) -> None:
    """Assigned log-weights define normalized component probabilities."""
    mixture = MixtureModel(mock_comps)
    mixture.log_weights = np.log([2.0, 6.0])

    assert mixture.log_weights == pytest.approx(np.log([0.25, 0.75]))
    assert np.sum(np.exp(mixture.log_weights)) == pytest.approx(1.0, abs=1e-12)
    assert mixture.weights == pytest.approx([0.25, 0.75])


def test_init_normalizes_log_weights(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Constructor weights accepted within tolerance are normalized in log space."""
    mixture = MixtureModel(mock_comps, weights=[0.3, 0.700001])

    assert np.sum(np.exp(mixture.log_weights)) == pytest.approx(1.0, rel=0.0, abs=1e-12)


@pytest.mark.parametrize("offset", [-1e100, 1e100])
def test_log_weights_normalization_handles_large_common_offsets(
    mock_comps: tuple[ParametricFamilyDistribution, ...], offset: float
) -> None:
    """A large common log-weight offset does not erase the normalization constant."""
    mixture = MixtureModel(mock_comps)
    mixture.log_weights = [offset, offset]

    assert mixture.log_weights == pytest.approx(np.log([0.5, 0.5]))
    assert np.sum(np.exp(mixture.log_weights)) == pytest.approx(1.0, abs=1e-12)


def test_log_weights_setter_copies_input_array(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Mutating the caller's array cannot change assigned log-weights."""
    mixture = MixtureModel(mock_comps)
    raw = np.array([0.0, 0.0])
    mixture.log_weights = raw
    before = mixture.log_weights.copy()

    raw[0] = 999.0

    assert np.array_equal(mixture.log_weights, before)


def test_log_weights_getter_is_read_only(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """The public log-weight array cannot be modified in place."""
    mixture = MixtureModel(mock_comps)
    before = mixture.log_weights.copy()

    with pytest.raises(ValueError, match="read-only"):
        mixture.log_weights[0] = 0.0

    assert np.array_equal(mixture.log_weights, before)


@pytest.mark.parametrize("log_weights", [[-np.inf, -np.inf], [np.nan, 0.0], [np.inf, 0.0]])
def test_log_weights_setter_rejects_values_that_cannot_be_normalized(
    mock_comps: tuple[ParametricFamilyDistribution, ...], log_weights: list[float]
) -> None:
    """Invalid log-weights are rejected without changing the mixture."""
    mixture = MixtureModel(mock_comps)
    before = mixture.log_weights.copy()

    with pytest.raises(ValueError, match="normalizable"):
        mixture.log_weights = log_weights

    assert np.array_equal(mixture.log_weights, before)


def test_log_weights_setter_accepts_zero_weight(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """A negative infinite log-weight is valid when another component has finite mass."""
    mixture = MixtureModel(mock_comps)
    mixture.log_weights = [-np.inf, 0.0]

    assert np.array_equal(mixture.log_weights, [-np.inf, 0.0])
    assert np.array_equal(mixture.weights, [0.0, 1.0])


def test_copy_preserves_log_weights_exactly(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Copying preserves tiny log-weights and keeps component membership independent."""
    mixture = MixtureModel(mock_comps)
    mixture.log_weights = [-800.0, 0.0]
    copied = copy(mixture)

    assert np.array_equal(copied.log_weights, mixture.log_weights)
    assert copied.log_weights is not mixture.log_weights
    assert copied.components == mixture.components
    assert all(left is right for left, right in zip(copied.components, mixture.components))

    copied.add_component(mock_comps[0], weight=0.5)

    assert mixture.components == mock_comps
    assert np.array_equal(mixture.log_weights, [-800.0, 0.0])
    assert copied.n_components == mixture.n_components + 1


def test_pdf_matches_exp_lpdf_exactly() -> None:
    """Densities agree at the center, in the tail, and outside the support."""
    exponential = family(FamilyName.EXPONENTIAL)
    mixture = MixtureModel([exponential(lambda_=1.0), exponential(lambda_=3.0)])
    mixture.log_weights = np.log([2.0, 6.0])
    data = np.array([-1.0, 0.0, 0.5, 1.0, 800.0])

    density = mixture.pdf(data)

    assert np.array_equal(density, np.exp(mixture.lpdf(data)))
    assert density[0] == density[-1] == 0.0
    assert np.isneginf(mixture.lpdf(data[0]))
    assert np.isfinite(mixture.lpdf(data[-1]))
    assert isinstance(mixture.pdf(0.5), np.float64)
    assert mixture.pdf(0.5) == np.exp(mixture.lpdf(0.5))
    on_support = data[1:-1]
    assert mixture.loglikelihood(on_support) == pytest.approx(np.sum(np.log(mixture.pdf(on_support))))


def test_pdf_does_not_construct_core_mixture(
    mock_comps: tuple[ParametricFamilyDistribution, ...], mock_finite_mixture: MagicMock
) -> None:
    """Evaluating a mixture density uses component log-densities directly."""
    mixture = MixtureModel(mock_comps)

    mixture.pdf([0.0, 1.0])

    mock_finite_mixture.assert_not_called()


def test_add_component_with_prior_unnormalized_weights(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Adding a component preserves the normalized proportions of assigned weights."""
    mixture = MixtureModel(mock_comps)
    mixture.log_weights = np.log([2.0, 6.0])

    mixture.add_component(mock_comps[0], weight=0.5)

    assert mixture.weights == pytest.approx([0.125, 0.375, 0.5])
    assert np.sum(np.exp(mixture.log_weights)) == pytest.approx(1.0, abs=1e-12)
    assert not mixture.log_weights.flags.writeable


def test_add_component_does_not_modify_the_array_given_to_the_setter(
    mock_comps: tuple[ParametricFamilyDistribution, ...],
) -> None:
    """Adding a component cannot mutate the caller's original log-weight array."""
    mixture = MixtureModel(mock_comps)
    raw = np.log([0.5, 0.5])
    before = raw.copy()
    mixture.log_weights = raw

    mixture.add_component(mock_comps[0], weight=0.5)

    assert np.array_equal(raw, before)


def test_remove_component_rejects_removal_of_all_remaining_mass(
    mock_comps: tuple[ParametricFamilyDistribution, ...],
) -> None:
    """Removing the only positive weight is rejected before changing the mixture."""
    mixture = MixtureModel(mock_comps)
    mixture.log_weights = [-np.inf, 0.0]

    with pytest.raises(ValueError, match="positive weight"):
        mixture.remove_component(1)

    assert mixture.components == mock_comps
    assert np.array_equal(mixture.log_weights, [-np.inf, 0.0])
    assert np.array_equal(mixture.weights, [0.0, 1.0])


def test_remove_component_keeps_log_weights_read_only(mock_comps: tuple[ParametricFamilyDistribution, ...]) -> None:
    """Removing zero-weight components preserves normalization and read-only storage."""
    mixture = MixtureModel(mock_comps)
    mixture.log_weights = [-np.inf, 0.0]

    mixture.remove_component(0)

    assert np.array_equal(mixture.log_weights, [0.0])
    assert not mixture.log_weights.flags.writeable


def test_to_core_mixture_exports_valid_distribution() -> None:
    """An explicit export constructs a real core mixture with normalized weights."""
    exponential = family(FamilyName.EXPONENTIAL)
    mixture = MixtureModel([exponential(lambda_=1.0), exponential(lambda_=3.0)])
    mixture.log_weights = np.log([2.0, 6.0])

    exported = mixture.to_core_mixture()

    assert isinstance(exported, FiniteMixtureDistribution)
    assert exported.weights == pytest.approx([0.25, 0.75])
