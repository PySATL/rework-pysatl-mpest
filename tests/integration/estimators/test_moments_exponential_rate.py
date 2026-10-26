"""
Integration tests for the Method of Moments strategy of the exponential family (``rate`` parametrization).
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
from tests.helpers.pipeline import make_state, q_block


def moments_block(component_id: int = 0, params: set[str] | None = None) -> OptimizationBlock:
    return OptimizationBlock(component_id, params, MaximizationStrategy.MOMENTS)


def exponential(lambda_: float) -> ParametricFamilyDistribution:
    return family(FamilyName.EXPONENTIAL)(lambda_=lambda_)


@pytest.fixture
def state() -> PipelineState:
    return make_state(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        H=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]),
    )


def test_raises_if_h_is_not_computed() -> None:
    state = make_state(X=np.array([1.0]), H=None)

    with pytest.raises(ValueError, match=r"^Responsibility matrix H is not computed\.$"):
        moments_strategy(exponential(1.0), state, moments_block(), optimizer=None)  # type: ignore[arg-type]


def test_returns_the_inverse_of_the_weighted_mean(state: PipelineState) -> None:
    assert state.H is not None
    H_j = state.H[:, 0]
    expected = H_j.sum() / np.dot(H_j, state.X)

    component_id, new_params = moments_strategy(exponential(1.0), state, moments_block(), optimizer=None)  # type: ignore[arg-type]

    assert component_id == 0
    assert new_params == pytest.approx({"lambda_": expected})
    assert type(new_params["lambda_"]) is float


def test_agrees_with_the_q_function_estimate(state: PipelineState) -> None:
    """For the exponential distribution the method of moments and maximum likelihood give the same rate."""
    _, by_moments = moments_strategy(exponential(1.0), state, moments_block(component_id=1), optimizer=None)  # type: ignore[arg-type]
    _, by_q_function = q_function_strategy(exponential(1.0), state, q_block(component_id=1), optimizer=None)  # type: ignore[arg-type]

    assert by_moments == pytest.approx(by_q_function)


@pytest.mark.parametrize(("params", "expected"), [(None, {"lambda_"}), ({"lambda_"}, {"lambda_"}), (set(), set())])
def test_updates_the_requested_parameters(state: PipelineState, params: set[str] | None, expected: set[str]) -> None:
    _, new_params = moments_strategy(exponential(1.0), state, moments_block(params=params), optimizer=None)  # type: ignore[arg-type]

    assert set(new_params) == expected


def test_rejects_unknown_parameters(state: PipelineState) -> None:
    with pytest.raises(ValueError, match=r"\['beta'\] are not parameters of the 'rate' parametrization"):
        moments_strategy(exponential(1.0), state, moments_block(params={"beta"}), optimizer=None)  # type: ignore[arg-type]


def test_updates_nothing_for_negligible_responsibility(state: PipelineState) -> None:
    assert state.H is not None
    state.H.fill(1e-10)

    assert moments_strategy(exponential(1.0), state, moments_block(), optimizer=None) == (0, {})  # type: ignore[arg-type]


@pytest.mark.parametrize("values", [[0.0, 0.0], [-1.0, -2.0]], ids=["zero mean", "negative mean"])
def test_keeps_the_rate_if_the_weighted_mean_is_not_positive(values: list[float]) -> None:
    """The estimate would be infinite or negative, and pysatl-core rejects a negative rate."""
    state = make_state(X=np.array(values), H=np.ones((2, 1)))

    _, new_params = moments_strategy(exponential(2.5), state, moments_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params == {"lambda_": 2.5}


@st.composite
def exponential_sample(draw: st.DrawFn) -> tuple[np.ndarray, float]:
    true_rate = draw(st.floats(min_value=0.1, max_value=100))
    rng = np.random.default_rng(42)
    return rng.exponential(scale=1.0 / true_rate, size=10000), true_rate


@settings(max_examples=50)
@given(data=exponential_sample())
def test_recovers_the_true_rate_on_ideal_data(data: tuple[np.ndarray, float]) -> None:
    X, true_rate = data
    H_j = np.ones_like(X)
    state = make_state(X=X, H=np.vstack([H_j, np.zeros_like(H_j)]).T)

    _, new_params = moments_strategy(exponential(0.001), state, moments_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params["lambda_"] == pytest.approx(true_rate, rel=0.1)
