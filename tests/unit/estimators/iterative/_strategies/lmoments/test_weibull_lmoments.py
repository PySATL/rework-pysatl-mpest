"""Tests for L-Moments optimization strategy for Weibull distribution"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pysatl_mpest.distributions import Weibull
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from pysatl_mpest.estimators.iterative._strategies import lmoments_strategy
from pysatl_mpest.exceptions import NumericalStabilityError

DTYPES_TO_TEST = [np.float32, np.float64]


@pytest.fixture(params=DTYPES_TO_TEST)
def parametrized_weibull_setup(request) -> tuple[Weibull, PipelineState, np.floating]:
    dtype = request.param
    component = Weibull(loc=0.0, scale=1.0, shape=1.0, dtype=dtype)
    state = PipelineState(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype),
        H=np.array([[0.2], [0.2], [0.2], [0.2], [0.2]], dtype=dtype),
        prev_mixture=None,
        curr_mixture=None,
        error=None,
    )
    return component, state, dtype


def test_lmoments_weibull_raises_value_error_if_h_is_none(parametrized_weibull_setup):
    weibull_component, state, _ = parametrized_weibull_setup
    state.H = None
    block = OptimizationBlock(0, {"loc", "scale", "shape"}, MaximizationStrategy.LMOMENTS)
    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        lmoments_strategy(weibull_component, state, block, optimizer=None)


def test_lmoments_weibull_returns_correct_types(parametrized_weibull_setup):
    weibull_component, pipeline_state, dtype = parametrized_weibull_setup
    block = OptimizationBlock(0, {"loc", "scale", "shape"}, MaximizationStrategy.LMOMENTS)
    result = lmoments_strategy(weibull_component, pipeline_state, block, optimizer=None)

    assert isinstance(result, tuple)
    assert isinstance(result[1], dict)
    for value in result[1].values():
        assert isinstance(value, dtype)


def test_lmoments_weibull_calculation_correctness(parametrized_weibull_setup):
    """
    Concrete example verification:
    X = [10, 20], H = [0.5, 0.5] => L1 = 15.0, L2 = 2.5
    For fixed k=1 (Exponential):
    lambda = 2 * L2 = 5.0
    mu = L1 - lambda = 10.0
    """
    component, state, dtype = parametrized_weibull_setup
    state.X = np.array([10.0, 20.0], dtype=dtype)
    state.H = np.array([[0.5], [0.5]], dtype=dtype)

    component._shape = dtype(1.0)
    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)
    _, new_params = lmoments_strategy(component, state, block, optimizer=None)

    assert new_params[Weibull.PARAM_LOC] == pytest.approx(10.0, rel=1e-4)
    assert new_params[Weibull.PARAM_SCALE] == pytest.approx(5.0, rel=1e-4)


def test_lmoments_weibull_t3_clipping(parametrized_weibull_setup):
    """Verifies that t3 (L-skewness) is clipped to Hosking's approximation range."""
    component, state, dtype = parametrized_weibull_setup
    # Create extreme data to force t3 > 0.5 or t3 < 0
    state.X = np.array([1.0, 1.1, 100.0], dtype=dtype)
    state.H = np.array([[0.45], [0.45], [0.1]], dtype=dtype)

    block = OptimizationBlock(0, {"loc", "scale", "shape"}, MaximizationStrategy.LMOMENTS)
    # Should not raise ValueError because of np.clip(t3, 0.001, 0.499)
    _, new_params = lmoments_strategy(component, state, block, optimizer=None)
    assert Weibull.PARAM_SHAPE in new_params


@pytest.mark.parametrize("inf_moment", ["l1", "l2", "l3"])
def test_lmoments_weibull_inf_handling(parametrized_weibull_setup, inf_moment):
    """Verifies NumericalStabilityError is set when any relevant L-moment is infinite."""
    component, state, dtype = parametrized_weibull_setup

    # Use a large enough value to ensure overflow across float32 and float64

    if inf_moment == "l1":
        state.X = np.array([np.inf, np.inf], dtype=dtype)
        state.H = np.array([[0.5], [0.5]], dtype=dtype)
    elif inf_moment == "l2":
        # Force l2 to be inf by creating extreme spread
        state.X = np.array([-np.inf, np.inf], dtype=dtype)
        state.H = np.array([[0.5], [0.5]], dtype=dtype)
    elif inf_moment == "l3":
        # Force l3 to inf while keeping l2 finite if possible,
        # or simply trigger the non-finite check.
        state.X = np.array([0.0, 1.0, np.inf], dtype=dtype)
        state.H = np.array([[0.33], [0.33], [0.34]], dtype=dtype)

    block = OptimizationBlock(0, {"loc", "scale", "shape"}, MaximizationStrategy.LMOMENTS)

    _, new_params = lmoments_strategy(component, state, block, optimizer=None)

    assert state.error is not None
    assert isinstance(state.error, NumericalStabilityError)
    assert new_params == {}


def test_lmoments_weibull_handles_negligible_responsibility(parametrized_weibull_setup):
    component, state, dtype = parametrized_weibull_setup
    state.H.fill(dtype(1e-12))
    block = OptimizationBlock(0, {"loc", "scale", "shape"}, MaximizationStrategy.LMOMENTS)
    _, new_params = lmoments_strategy(component, state, block, optimizer=None)
    assert new_params == {}


@st.composite
def stable_weibull_data(draw):
    dtype = np.float64
    # Ranges where L-moments are numerically stable
    true_loc = draw(st.floats(min_value=0.1, max_value=5.0))
    true_scale = draw(st.floats(min_value=1.0, max_value=3.0))
    true_shape = draw(st.floats(min_value=1.2, max_value=3.5))
    size = 8000
    rng = np.random.default_rng(42)
    X = (rng.weibull(true_shape, size) * true_scale + true_loc).astype(dtype)
    return X, true_loc, true_scale, true_shape


@settings(max_examples=15, deadline=None)
@given(data=stable_weibull_data())
def test_lmoments_weibull_statistical_recovery(data):
    X, t_loc, t_scale, t_shape = data
    H = np.ones((len(X), 1))
    comp = Weibull(loc=1.0, scale=1.0, shape=1.0)
    state = PipelineState(X=X, H=H, prev_mixture=None, curr_mixture=None, error=None)
    block = OptimizationBlock(0, {"loc", "scale", "shape"}, MaximizationStrategy.LMOMENTS)

    _, res = lmoments_strategy(comp, state, block, optimizer=None)

    assert res[Weibull.PARAM_LOC] == pytest.approx(t_loc, abs=0.4)
    assert res[Weibull.PARAM_SCALE] == pytest.approx(t_scale, rel=0.15)
    assert res[Weibull.PARAM_SHAPE] == pytest.approx(t_shape, rel=0.15)
