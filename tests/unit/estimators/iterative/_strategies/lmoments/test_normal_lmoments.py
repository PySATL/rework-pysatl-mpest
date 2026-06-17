"""Tests for L-Moments optimization strategy for Normal distribution"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pysatl_mpest.distributions import Normal
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from pysatl_mpest.estimators.iterative._strategies import lmoments_strategy
from pysatl_mpest.exceptions import NumericalStabilityError

DTYPES_TO_TEST = [np.float32, np.float64]

# Constants for overflow testing (float16 limits)
_FLOAT16_OVERFLOW_L1 = 65504.0  # Value causing overflow in first L-moment calculation
_FLOAT16_OVERFLOW_L2 = 300.0  # Value causing overflow in second L-moment calculation


@pytest.fixture(params=DTYPES_TO_TEST)
def parametrized_normal_setup(request) -> tuple[Normal, PipelineState, np.floating]:
    """Creates a parametrized fixture providing a Normal component and PipelineState for various dtypes."""
    dtype = request.param
    component = Normal(loc=0.0, scale=1.0, dtype=dtype)

    state = PipelineState(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype),
        H=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=dtype),
        prev_mixture=None,
        curr_mixture=None,
        error=None,
    )
    return component, state, dtype


def test_lmoments_normal_raises_value_error_if_h_is_none(parametrized_normal_setup):
    """Verifies that a ValueError is raised if the responsibility matrix H has not been computed."""
    normal_component, state, _ = parametrized_normal_setup
    state.H = None
    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        lmoments_strategy(normal_component, state, block, optimizer=None)


def test_lmoments_normal_returns_correct_types(parametrized_normal_setup):
    """Verifies that the function returns a tuple with the correct data types (int, dict[str, DType])."""
    normal_component, pipeline_state, dtype = parametrized_normal_setup
    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)

    result = lmoments_strategy(normal_component, pipeline_state, block, optimizer=None)

    assert isinstance(result, tuple)
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, dtype)


def test_lmoments_normal_calculation_correctness(parametrized_normal_setup):
    """
    Test case verification with a concrete example.
    Given data [10, 20] and weights [0.5, 0.5]:
    L1 = (10*0.5 + 20*0.5) / 1.0 = 15.0
    rank_weights:
      w1 = (0.5 - 0.5*0.5)/1 = 0.25
      w2 = (1.0 - 0.5*0.5)/1 = 0.75
    b1 = (10*0.5*0.25 + 20*0.5*0.75) / 1.0 = 1.25 + 7.5 = 8.75
    L2 = 2*8.75 - 15.0 = 2.5
    Sigma = L2 * sqrt(pi) = 2.5 * 1.77245... = 4.4311...
    """
    component, state, dtype = parametrized_normal_setup
    state.X = np.array([10.0, 20.0], dtype=dtype)
    state.H = np.array([[0.5], [0.5]], dtype=dtype)

    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)
    _, new_params = lmoments_strategy(component, state, block, optimizer=None)

    expected_mu = 15.0
    expected_sigma = 2.5 * np.sqrt(np.pi)

    assert new_params[Normal.PARAM_LOC] == pytest.approx(expected_mu, rel=1e-4)
    assert new_params[Normal.PARAM_SCALE] == pytest.approx(expected_sigma, rel=1e-4)


@pytest.mark.parametrize(
    "params_to_optimize_in_block, fixed_params_on_component, expected_keys",
    [
        ({"loc", "scale"}, set(), {"loc", "scale"}),
        ({"loc"}, set(), {"loc"}),
        ({"scale"}, set(), {"scale"}),
        ({"loc", "scale"}, {"loc"}, {"scale"}),
        ({"loc", "scale"}, {"scale"}, {"loc"}),
        ({"loc", "scale"}, {"loc", "scale"}, set()),
        ({"non_existent_param", "loc"}, set(), {"loc"}),
        (set(), set(), set()),
    ],
)
def test_lmoments_normal_respects_fixed_and_optimizable_params(
    parametrized_normal_setup, params_to_optimize_in_block, fixed_params_on_component, expected_keys
):
    """Verifies parameter update logic against component fixes and block constraints."""
    normal_component, pipeline_state, _ = parametrized_normal_setup
    for param in fixed_params_on_component:
        normal_component.fix_param(param)

    block = OptimizationBlock(0, params_to_optimize_in_block, MaximizationStrategy.LMOMENTS)
    _, new_params = lmoments_strategy(normal_component, pipeline_state, block, optimizer=None)
    assert set(new_params.keys()) == expected_keys


def test_lmoments_normal_handles_negligible_responsibility(parametrized_normal_setup):
    """Verifies that components with near-zero responsibility are not updated."""
    normal_component, pipeline_state, dtype = parametrized_normal_setup
    pipeline_state.H.fill(dtype(1e-10))

    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)
    _, new_params = lmoments_strategy(normal_component, pipeline_state, block, optimizer=None)
    assert new_params == {}


@pytest.mark.parametrize(
    "x_val, n_samples",
    [
        pytest.param(_FLOAT16_OVERFLOW_L1, 2, id="overflow_first_lmoment"),
        pytest.param(_FLOAT16_OVERFLOW_L2, 220, id="overflow_second_lmoment"),
    ],
)
def test_lmoments_normal_overflow_handling(parametrized_normal_setup, x_val, n_samples):
    """Verifies NumericalStabilityError is set upon overflow in L-moment calculations."""
    dtype = np.float16
    component = Normal(loc=0.0, scale=1.0, dtype=dtype)

    if x_val == _FLOAT16_OVERFLOW_L1:
        X_data = np.full(n_samples, x_val, dtype=dtype)
    else:
        X_data = np.array([x_val * (i + 1) for i in range(n_samples)], dtype=dtype)

    H = np.ones((n_samples, 1), dtype=dtype)
    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)
    state = PipelineState(X=X_data, H=H, curr_mixture=None, prev_mixture=None, error=None)

    _, new_params = lmoments_strategy(component, state, block, optimizer=None)

    assert state.error is not None
    assert isinstance(state.error, NumericalStabilityError)
    assert "Overflow detected during Lmoments optimization" in str(state.error)
    assert new_params == {}


def test_lmoments_normal_handles_inf_lmoments_directly(parametrized_normal_setup):
    """Directly tests the branch where computed l1 or l2 is infinite."""
    component, state, dtype = parametrized_normal_setup

    # Create artificial data that leads to inf in l1 or l2
    state.X = np.array([np.inf, 1.0, 2.0], dtype=dtype)
    state.H = np.array([[1.0], [0.0], [0.0]], dtype=dtype)  # Only first point has weight

    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)
    _, new_params = lmoments_strategy(component, state, block, optimizer=None)

    # Error should be set and empty dict returned
    assert state.error is not None
    assert isinstance(state.error, NumericalStabilityError)
    assert new_params == {}


def test_lmoments_normal_scale_clipping_to_epsilon(parametrized_normal_setup):
    """Verifies that scale parameter is clipped to machine epsilon when L2 is near zero."""
    component, state, dtype = parametrized_normal_setup

    # All points identical -> L2 ~ 0 -> scale should be clipped to epsilon
    val = dtype(5.0)
    state.X = np.array([val, val, val, val], dtype=dtype)
    state.H = np.ones((4, 1), dtype=dtype)

    block = OptimizationBlock(0, {"scale"}, MaximizationStrategy.LMOMENTS)
    _, new_params = lmoments_strategy(component, state, block, optimizer=None)

    expected_min = np.finfo(dtype).eps
    assert new_params[Normal.PARAM_SCALE] >= expected_min
    assert new_params[Normal.PARAM_SCALE] == pytest.approx(expected_min, rel=1e-6)


@st.composite
def normal_data_and_true_params(draw, dtype_strategy=st.sampled_from([np.float64])):
    """Generates true Normal parameters and a synthetic dataset with full responsibility."""
    dtype = draw(dtype_strategy)
    # Conservative ranges for stable estimation
    true_loc = draw(st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False))
    true_scale = draw(st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False))
    sample_size = draw(st.integers(min_value=8000, max_value=15000))

    rng = np.random.default_rng(42)
    X = rng.normal(loc=true_loc, scale=true_scale, size=sample_size).astype(dtype)
    return X, dtype(true_loc), dtype(true_scale), dtype


@settings(max_examples=50, deadline=None)
@given(data=normal_data_and_true_params())
def test_lmoments_normal_recovers_true_params_on_ideal_data(data):
    """
    Statistical sanity check: verifies that L-moments recover ground-truth
    parameters when all responsibility belongs to a single component.

    Note: Tolerances are adaptive to account for sampling variability that
    scales with the true parameter values.
    """
    X, true_loc, true_scale, dtype = data

    H = np.ones((len(X), 1), dtype=dtype)
    start_component = Normal(loc=-999.0, scale=0.001, dtype=dtype)

    state = PipelineState(X=X, H=H, prev_mixture=None, curr_mixture=None, error=None)
    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)

    _, new_params = lmoments_strategy(start_component, state, block, optimizer=None)

    # Adaptive tolerances: absolute error grows with true_scale
    loc_atol = 0.3 + 0.02 * float(true_scale)  # ~0.3 base + 2% of scale
    scale_rtol = 0.12 + 0.003 * float(true_scale)  # ~12% base + 0.3% of scale

    assert new_params[Normal.PARAM_LOC] == pytest.approx(true_loc, abs=loc_atol)
    assert new_params[Normal.PARAM_SCALE] == pytest.approx(true_scale, rel=scale_rtol)

    assert isinstance(new_params[Normal.PARAM_LOC], dtype)
    assert isinstance(new_params[Normal.PARAM_SCALE], dtype)
