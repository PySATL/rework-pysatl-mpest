"""
Integration tests for the Method of Moments strategy of the normal family (``meanStd`` parametrization).
"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from pysatl_mpest.estimators.iterative._strategies import moments_strategy, q_function_strategy
from tests.helpers.families import family
from tests.helpers.moments import weighted_moments
from tests.helpers.pipeline import make_state, q_block


def moments_block(component_id: int = 0, params: set[str] | None = None) -> OptimizationBlock:
    return OptimizationBlock(component_id, params, MaximizationStrategy.MOMENTS)


def normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


@pytest.fixture
def state() -> PipelineState:
    return make_state(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        H=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]),
    )


def test_raises_if_h_is_not_computed() -> None:
    state = make_state(X=np.array([1.0]), H=None)

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        moments_strategy(normal(0.0, 1.0), state, moments_block(), optimizer=None)  # type: ignore[arg-type]


def test_weighted_mean_and_standard_deviation_of_a_simple_sample() -> None:
    """Data 10 and 20 with weights 0.2 and 0.8: the mean is 18, the variance is 0.2 * 64 + 0.8 * 4 = 16."""
    state = make_state(X=np.array([10.0, 20.0]), H=np.array([[0.2], [0.8]]))

    component_id, new_params = moments_strategy(normal(0.0, 1.0), state, moments_block(), optimizer=None)  # type: ignore[arg-type]

    assert component_id == 0
    assert new_params == pytest.approx({"mu": 18.0, "sigma": 4.0})
    assert all(type(value) is float for value in new_params.values())


@pytest.mark.parametrize("component_id", [0, 1])
def test_agrees_with_the_q_function_estimate(state: PipelineState, component_id: int) -> None:
    """For the normal distribution the method of moments and maximum likelihood give the same estimates."""
    _, by_moments = moments_strategy(normal(0.0, 1.0), state, moments_block(component_id), optimizer=None)  # type: ignore[arg-type]
    _, by_q_function = q_function_strategy(normal(0.0, 1.0), state, q_block(component_id), optimizer=None)  # type: ignore[arg-type]

    assert by_moments == pytest.approx(by_q_function)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (None, {"mu", "sigma"}),
        ({"mu", "sigma"}, {"mu", "sigma"}),
        ({"mu"}, {"mu"}),
        ({"sigma"}, {"sigma"}),
        (set(), set()),
    ],
)
def test_updates_the_requested_parameters(state: PipelineState, params: set[str] | None, expected: set[str]) -> None:
    _, new_params = moments_strategy(normal(0.0, 1.0), state, moments_block(params=params), optimizer=None)  # type: ignore[arg-type]

    assert set(new_params) == expected


def test_view_with_fixed_mean_updates_only_the_standard_deviation(state: PipelineState) -> None:
    fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)

    _, new_params = moments_strategy(fixed_mu, state, moments_block(), optimizer=None)  # type: ignore[arg-type]

    assert set(new_params) == {"sigma"}


def test_view_with_fixed_standard_deviation_updates_only_the_mean(state: PipelineState) -> None:
    fixed_sigma = family(FamilyName.NORMAL).view(sigma=2.0)(mu=0.0)

    _, new_params = moments_strategy(fixed_sigma, state, moments_block(), optimizer=None)  # type: ignore[arg-type]

    assert set(new_params) == {"mu"}


def test_standard_deviation_is_taken_around_the_fixed_mean(state: PipelineState) -> None:
    fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)
    _, std_around_fixed = weighted_moments(state, mu=0.0)

    _, new_params = moments_strategy(fixed_mu, state, moments_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params == pytest.approx({"sigma": std_around_fixed})


def test_rejects_a_fixed_parameter(state: PipelineState) -> None:
    fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)

    with pytest.raises(ValueError, match=r"\['mu'\] are fixed and cannot be optimized"):
        moments_strategy(fixed_mu, state, moments_block(params={"mu", "sigma"}), optimizer=None)  # type: ignore[arg-type]


def test_rejects_unknown_parameters(state: PipelineState) -> None:
    with pytest.raises(ValueError, match=r"\['var'\] are not parameters of the 'meanStd' parametrization"):
        moments_strategy(normal(0.0, 1.0), state, moments_block(params={"var"}), optimizer=None)  # type: ignore[arg-type]


def test_updates_nothing_for_negligible_responsibility(state: PipelineState) -> None:
    assert state.H is not None
    state.H.fill(1e-10)

    assert moments_strategy(normal(0.0, 1.0), state, moments_block(), optimizer=None) == (0, {})  # type: ignore[arg-type]


def test_keeps_the_standard_deviation_if_all_the_data_coincide() -> None:
    """The variance is zero, which pysatl-core would reject as a standard deviation, so the old value is kept."""
    state = make_state(X=np.array([10.0, 10.0, 10.0]), H=np.ones((3, 2)))

    _, new_params = moments_strategy(normal(0.0, 2.5), state, moments_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params == {"mu": 10.0, "sigma": 2.5}


def test_keeps_the_standard_deviation_if_the_data_coincide_with_the_fixed_mean() -> None:
    state = make_state(X=np.array([10.0, 10.0, 10.0]), H=np.ones((3, 2)))
    fixed_mu = family(FamilyName.NORMAL).view(mu=10.0)(sigma=2.5)

    _, new_params = moments_strategy(fixed_mu, state, moments_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params == {"sigma": 2.5}


@st.composite
def normal_sample(draw: st.DrawFn) -> tuple[np.ndarray, float, float]:
    true_mu = draw(st.floats(min_value=-50, max_value=50))
    true_sigma = draw(st.floats(min_value=0.5, max_value=20))
    rng = np.random.default_rng(42)
    return rng.normal(loc=true_mu, scale=true_sigma, size=10000), true_mu, true_sigma


@settings(max_examples=50)
@given(data=normal_sample())
def test_recovers_the_true_parameters_on_ideal_data(data: tuple[np.ndarray, float, float]) -> None:
    X, true_mu, true_sigma = data
    H_j = np.ones_like(X)
    state = make_state(X=X, H=np.vstack([H_j, np.zeros_like(H_j)]).T)

    _, new_params = moments_strategy(normal(-999.0, 0.001), state, moments_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params["mu"] == pytest.approx(true_mu, abs=1.0)
    assert new_params["sigma"] == pytest.approx(true_sigma, rel=0.1)


@st.composite
def normal_sample_with_random_weights(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, float, float]:
    true_mu = draw(st.floats(min_value=-10.0, max_value=10.0))
    true_sigma = draw(st.floats(min_value=0.5, max_value=5.0))
    rng = np.random.default_rng(42)
    X = rng.normal(loc=true_mu, scale=true_sigma, size=15000)
    return X, rng.uniform(low=0.1, high=1.0, size=15000), true_mu, true_sigma


@settings(max_examples=30, deadline=None)
@given(data=normal_sample_with_random_weights())
def test_converges_to_the_true_parameters_with_random_weights(
    data: tuple[np.ndarray, np.ndarray, float, float],
) -> None:
    """Weights independent of the data do not bias the weighted moments."""
    X, weights, true_mu, true_sigma = data
    state = make_state(X=X, H=np.vstack([weights, np.zeros_like(weights)]).T)

    _, new_params = moments_strategy(normal(true_mu + 10, true_sigma + 5), state, moments_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params["mu"] == pytest.approx(true_mu, abs=0.2, rel=0.1)
    assert new_params["sigma"] == pytest.approx(true_sigma, rel=0.15)
