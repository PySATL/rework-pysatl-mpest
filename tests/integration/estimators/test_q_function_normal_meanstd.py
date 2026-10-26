"""
Integration tests for the analytical Q-function strategy of the normal family (``meanStd`` parametrization).
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
from pysatl_mpest.estimators.iterative import PipelineState
from pysatl_mpest.estimators.iterative._strategies import q_function_strategy
from tests.helpers.families import family
from tests.helpers.moments import weighted_moments
from tests.helpers.pipeline import make_state, q_block


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
        q_function_strategy(normal(0.0, 1.0), state, q_block(), optimizer=None)  # type: ignore[arg-type]


def test_returns_the_weighted_mean_and_standard_deviation(state: PipelineState) -> None:
    mean, std = weighted_moments(state)

    component_id, new_params = q_function_strategy(normal(0.0, 1.0), state, q_block(), optimizer=None)  # type: ignore[arg-type]

    assert component_id == 0
    assert new_params == pytest.approx({"mu": mean, "sigma": std})
    assert all(type(value) is float for value in new_params.values())


def test_uses_the_column_of_the_block_component(state: PipelineState) -> None:
    mean, std = weighted_moments(state, component_id=1)

    _, new_params = q_function_strategy(normal(0.0, 1.0), state, q_block(component_id=1), optimizer=None)  # type: ignore[arg-type]

    assert new_params == pytest.approx({"mu": mean, "sigma": std})


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
    _, new_params = q_function_strategy(normal(0.0, 1.0), state, q_block(params=params), optimizer=None)  # type: ignore[arg-type]

    assert set(new_params) == expected


def test_view_with_fixed_mean_updates_only_the_standard_deviation(state: PipelineState) -> None:
    fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)

    _, new_params = q_function_strategy(fixed_mu, state, q_block(), optimizer=None)  # type: ignore[arg-type]

    assert set(new_params) == {"sigma"}


def test_view_with_fixed_standard_deviation_updates_only_the_mean(state: PipelineState) -> None:
    fixed_sigma = family(FamilyName.NORMAL).view(sigma=2.0)(mu=0.0)

    _, new_params = q_function_strategy(fixed_sigma, state, q_block(), optimizer=None)  # type: ignore[arg-type]

    assert set(new_params) == {"mu"}


def test_standard_deviation_is_taken_around_the_fixed_mean(state: PipelineState) -> None:
    """The mean is not optimized, so the spread is measured around its fixed value, not around the sample mean."""
    fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)
    _, std_around_fixed = weighted_moments(state, mu=0.0)
    _, std_around_mean = weighted_moments(state)

    _, new_params = q_function_strategy(fixed_mu, state, q_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params["sigma"] == pytest.approx(std_around_fixed)
    assert new_params["sigma"] != pytest.approx(std_around_mean)


def test_standard_deviation_is_taken_around_the_new_mean(state: PipelineState) -> None:
    mean, std = weighted_moments(state)

    _, new_params = q_function_strategy(normal(10.0, 1.0), state, q_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params == pytest.approx({"mu": mean, "sigma": std})


def test_standard_deviation_uses_the_current_mean_if_only_sigma_is_requested(state: PipelineState) -> None:
    _, std_around_current = weighted_moments(state, mu=10.0)

    _, new_params = q_function_strategy(normal(10.0, 1.0), state, q_block(params={"sigma"}), optimizer=None)  # type: ignore[arg-type]

    assert new_params == pytest.approx({"sigma": std_around_current})


def test_rejects_a_fixed_parameter(state: PipelineState) -> None:
    fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)

    with pytest.raises(ValueError, match=r"\['mu'\] are fixed and cannot be optimized"):
        q_function_strategy(fixed_mu, state, q_block(params={"mu", "sigma"}), optimizer=None)  # type: ignore[arg-type]


def test_rejects_unknown_parameters(state: PipelineState) -> None:
    with pytest.raises(ValueError, match=r"\['var'\] are not parameters of the 'meanStd' parametrization"):
        q_function_strategy(normal(0.0, 1.0), state, q_block(params={"var"}), optimizer=None)  # type: ignore[arg-type]


def test_keeps_the_standard_deviation_if_the_variance_is_zero() -> None:
    """All the data equal the fixed mean, so the variance is zero and the old standard deviation is kept."""
    state = make_state(X=np.array([10.0, 10.0]), H=np.array([[1.0, 0.0], [1.0, 0.0]]))
    fixed_mu = family(FamilyName.NORMAL).view(mu=10.0)(sigma=2.5)

    _, new_params = q_function_strategy(fixed_mu, state, q_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params == {"sigma": 2.5}


def test_updates_nothing_for_negligible_responsibility(state: PipelineState) -> None:
    assert state.H is not None
    state.H.fill(1e-10)

    assert q_function_strategy(normal(0.0, 1.0), state, q_block(), optimizer=None) == (0, {})  # type: ignore[arg-type]


@st.composite
def normal_sample(draw: st.DrawFn) -> tuple[np.ndarray, float, float]:
    true_mu = draw(st.floats(min_value=-50, max_value=50))
    true_sigma = draw(st.floats(min_value=0.5, max_value=20))
    rng = np.random.default_rng(42)
    return rng.normal(loc=true_mu, scale=true_sigma, size=10000), true_mu, true_sigma


@settings(max_examples=50, deadline=None)
@given(data=normal_sample())
def test_recovers_the_true_parameters_on_ideal_data(data: tuple[np.ndarray, float, float]) -> None:
    """With responsibilities of 1.0 the closed-form estimates are the maximum likelihood ones."""
    X, true_mu, true_sigma = data
    H_j = np.ones_like(X)
    state = make_state(X=X, H=np.vstack([H_j, np.zeros_like(H_j)]).T)

    _, new_params = q_function_strategy(normal(-999.0, 0.001), state, q_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params["mu"] == pytest.approx(true_mu, abs=1.0)
    assert new_params["sigma"] == pytest.approx(true_sigma, rel=0.1)
