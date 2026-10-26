"""
Integration tests for the analytical Q-function strategy of the exponential family (``rate`` parametrization).
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
from tests.helpers.pipeline import make_state, q_block


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
        q_function_strategy(exponential(1.0), state, q_block(), optimizer=None)  # type: ignore[arg-type]


def test_returns_the_inverse_of_the_weighted_mean(state: PipelineState) -> None:
    assert state.H is not None
    H_j = state.H[:, 0]
    expected = H_j.sum() / np.dot(H_j, state.X)

    component_id, new_params = q_function_strategy(exponential(1.0), state, q_block(), optimizer=None)  # type: ignore[arg-type]

    assert component_id == 0
    assert new_params == pytest.approx({"lambda_": expected})
    assert type(new_params["lambda_"]) is float


def test_uses_the_column_of_the_block_component(state: PipelineState) -> None:
    assert state.H is not None
    H_j = state.H[:, 1]

    _, new_params = q_function_strategy(exponential(1.0), state, q_block(component_id=1), optimizer=None)  # type: ignore[arg-type]

    assert new_params["lambda_"] == pytest.approx(H_j.sum() / np.dot(H_j, state.X))


@pytest.mark.parametrize("params", [None, {"lambda_"}])
def test_updates_the_rate_when_it_is_requested(state: PipelineState, params: set[str] | None) -> None:
    _, new_params = q_function_strategy(exponential(1.0), state, q_block(params=params), optimizer=None)  # type: ignore[arg-type]

    assert set(new_params) == {"lambda_"}


def test_updates_nothing_when_nothing_is_requested(state: PipelineState) -> None:
    assert q_function_strategy(exponential(1.0), state, q_block(params=set()), optimizer=None) == (0, {})  # type: ignore[arg-type]


def test_rejects_unknown_parameters(state: PipelineState) -> None:
    with pytest.raises(ValueError, match=r"\['beta'\] are not parameters of the 'rate' parametrization"):
        q_function_strategy(exponential(1.0), state, q_block(params={"beta"}), optimizer=None)  # type: ignore[arg-type]


def test_keeps_the_rate_if_the_weighted_mean_is_zero() -> None:
    """The mean of the data is zero, so the rate would be infinite; the current one is kept."""
    state = make_state(X=np.array([0.0]), H=np.array([[1.0, 0.0]]))

    _, new_params = q_function_strategy(exponential(2.5), state, q_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params == {"lambda_": 2.5}


def test_updates_nothing_for_negligible_responsibility(state: PipelineState) -> None:
    assert state.H is not None
    state.H.fill(1e-10)

    assert q_function_strategy(exponential(1.0), state, q_block(), optimizer=None) == (0, {})  # type: ignore[arg-type]


@st.composite
def exponential_sample(draw: st.DrawFn) -> tuple[np.ndarray, float]:
    true_rate = draw(st.floats(min_value=0.1, max_value=100))
    rng = np.random.default_rng(42)
    return rng.exponential(scale=1.0 / true_rate, size=10000), true_rate


@settings(max_examples=50)
@given(data=exponential_sample())
def test_recovers_the_true_rate_on_ideal_data(data: tuple[np.ndarray, float]) -> None:
    """With responsibilities of 1.0 the closed-form estimate is the maximum likelihood one."""
    X, true_rate = data
    H_j = np.ones_like(X)
    state = make_state(X=X, H=np.vstack([H_j, np.zeros_like(H_j)]).T)

    _, new_params = q_function_strategy(exponential(0.001), state, q_block(), optimizer=None)  # type: ignore[arg-type]

    assert new_params["lambda_"] == pytest.approx(true_rate, rel=0.05)
