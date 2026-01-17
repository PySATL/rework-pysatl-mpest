"""Tests for Q-function optimization strategy for Pareto type 1 distribution"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rework_pysatl_mpest.distributions import Pareto
from rework_pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from rework_pysatl_mpest.estimators.iterative._strategies import q_function_strategy
from rework_pysatl_mpest.exceptions import NumericalStabilityError

DTYPES_TO_TEST: list[np.floating] = [np.float16, np.float32, np.float64]

# Test Fixtures
# -------------


@pytest.fixture(params=DTYPES_TO_TEST)
def parametrized_pareto_setup(request) -> tuple[Pareto, PipelineState, np.floating]:
    """
    Creates a parametrized fixture providing a default Pareto component, and a
    corresponding PipelineState for various dtypes.
    """
    dtype = request.param

    component = Pareto(shape=1.0, scale=0.5, dtype=dtype)

    state = PipelineState(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype),
        H=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=dtype),
        prev_mixture=None,
        curr_mixture=None,
        error=None,
    )

    return component, state, dtype


# Tests
# -----


def test_q_function_pareto_raises_value_error_if_h_is_none(parametrized_pareto_setup):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed.
    """
    pareto_component, state, _ = parametrized_pareto_setup

    state.H = None

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"shape", "scale"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        q_function_strategy(pareto_component, state, block, optimizer=None)


def test_q_function_pareto_returns_correct_types(parametrized_pareto_setup):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, float]).
    """

    pareto_component, pipeline_state, dtype = parametrized_pareto_setup

    expected_len = 2

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"shape", "scale"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    result = q_function_strategy(pareto_component, pipeline_state, block, optimizer=None)

    assert isinstance(result, tuple)
    assert len(result) == expected_len
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, dtype)


@pytest.mark.parametrize(
    "params_to_optimize_in_block, fixed_params_on_component, expected_keys",
    [
        ({"shape", "scale"}, set(), {"shape", "scale"}),  # Optimize both, none are fixed
        ({"shape"}, set(), {"shape"}),  # Optimize only loc
        ({"scale"}, set(), {"scale"}),  # Optimize only scale
        ({"shape", "scale"}, {"shape"}, {"scale"}),  # Optimize both, but loc is fixed
        ({"shape", "scale"}, {"scale"}, {"shape"}),  # Optimize both, but scale is fixed
        ({"shape", "scale"}, {"shape", "scale"}, set()),  # Optimize both, but both are fixed
        ({"non_existent_param", "shape"}, set(), {"shape"}),  # Ignore non-existent params
        (set(), set(), set()),  # Optimize nothing
    ],
)
def test_q_function_pareto_respects_fixed_and_optimizable_params(
    parametrized_pareto_setup, params_to_optimize_in_block, fixed_params_on_component, expected_keys
):
    """
    Verifies that the strategy correctly identifies which parameters
    to update based on the optimization block and the component's fixed parameters.
    """

    pareto_component, pipeline_state, dtype = parametrized_pareto_setup

    for param in fixed_params_on_component:
        pareto_component.fix_param(param)

    block = OptimizationBlock(
        component_id=0,
        params_to_optimize=params_to_optimize_in_block,
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    _, new_params = q_function_strategy(pareto_component, pipeline_state, block, optimizer=None)

    assert set(new_params.keys()) == expected_keys


def test_q_function_pareto_handles_negligible_responsibility(parametrized_pareto_setup):
    """
    Verifies that if a component's total responsibility (N_j) is near zero,
    its parameters are not updated.
    """

    pareto_component, pipeline_state, dtype = parametrized_pareto_setup

    pipeline_state.H.fill(dtype(1e-10))  # Make all responsibilities negligible
    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={"shape", "scale"},
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    _, new_params = q_function_strategy(pareto_component, pipeline_state, block, optimizer=None)

    assert new_params == {}


def test_pareto_scale_fallback_with_invalid_data():
    """
    Tests that Pareto scale optimization falls back to current value if no valid data
    points are found (mask is empty) (Line 384 else branch).
    """
    dtype = np.float64
    # Responsibilities are high (1.0), so we pass the early N_j check.
    # However, data X is negative. Pareto requires X > 0 (and X >= scale).
    # The mask (H > tol) & (X > 0) will be all False.

    X = np.array([-1.0, -2.0], dtype=dtype)
    H = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=dtype)

    original_scale = 1.5
    comp = Pareto(shape=2.0, scale=original_scale, dtype=dtype)

    state = PipelineState(X=X, H=H, prev_mixture=None, curr_mixture=None, error=None)

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"scale"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    _, params = q_function_strategy(comp, state, block, optimizer=None)

    assert Pareto.PARAM_SCALE in params
    assert params[Pareto.PARAM_SCALE] == original_scale


def test_pareto_shape_fallback_zero_denominator():
    """
    Tests the fallback branch for Pareto 'shape' (Line 394).
    It triggers when the denominator (weighted sum of log(X/scale)) is close to zero.
    This implies X is very close to scale (log(1) = 0).
    """
    dtype = np.float64
    # log(X/scale) = log(1) = 0
    # denominator = sum(H * 0) = 0

    X = np.array([2.0], dtype=dtype)
    H = np.array([[1.0, 0.0]], dtype=dtype)

    original_shape = 3.0
    original_scale = 2.0
    comp = Pareto(shape=original_shape, scale=original_scale, dtype=dtype)

    state = PipelineState(X=X, H=H, prev_mixture=None, curr_mixture=None, error=None)

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"shape"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    _, params = q_function_strategy(comp, state, block, optimizer=None)

    # Should fall back to the original shape because denominator is 0
    assert Pareto.PARAM_SHAPE in params
    assert params[Pareto.PARAM_SHAPE] == original_shape


def test_q_function_pareto_handles_numerical_overflow():
    """
    Verifies that if a numerical overflow occurs during calculations,
    a `NumericalStabilityError` is correctly registered in the pipeline state.
    """
    # --- Arrange ---
    # Use float16, which has a small range, and large values to force an overflow.
    # The max value for float16 is ~65504. The sum of X will exceed this.
    dtype = np.float16
    component = Pareto(shape=1.0, scale=1.0, dtype=dtype)
    X = np.full(1000000, 600, dtype=dtype)
    H_j = np.full(1000000, 1.0, dtype=dtype)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T
    state = PipelineState(X=X, H=H, prev_mixture=None, curr_mixture=None, error=None)

    block = OptimizationBlock(0, {"shape"}, MaximizationStrategy.QFUNCTION)

    # --- Act ---
    _, new_params = q_function_strategy(component, state, block, optimizer=None)

    # --- Assert ---
    assert state.error is not None
    assert isinstance(state.error, NumericalStabilityError)
    assert "Overflow detected during Q-function optimization" in str(state.error)
    assert new_params == {}


# Property-Based Test with Hypothesis
# -----------------------------------


@st.composite
def pareto_data_and_true_params(draw, dtype_strategy=st.sampled_from([np.float32, np.float64])):
    """Generates a true Pareto component and a data sample from it."""

    dtype = draw(dtype_strategy)

    # 1. Generate realistic parameters for the true distribution
    true_shape = draw(st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
    true_scale = draw(st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
    true_component = Pareto(shape=true_shape, scale=true_scale, dtype=dtype)

    # 2. Generate a large data sample from this distribution
    X = true_component.generate(size=1000)

    return (X, dtype(true_shape), dtype(true_scale), dtype)


@settings(max_examples=50, deadline=None)
@given(data=pareto_data_and_true_params())
def test_q_function_pareto_recovers_true_params_on_ideal_data(data):
    """
    Verifies that the analytical formulas can recover the original parameters
    from a sample when responsibilities are 1.0. This confirms the statistical
    validity of the implemented formulas in an ideal case (maximum likelihood estimation).
    """

    # --- Arrange ---
    X, true_shape, true_scale, dtype = data

    # This is the key assumption for this test: perfect knowledge that all
    # data points belong to our component of interest (responsibilities are all 1.0).
    H_j = np.ones_like(X, dtype=dtype)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T  # Simulate a 2-component mixture context

    # Use a starting component with completely different parameters to ensure
    # the update is based on data, not the initial guess.
    start_component = Pareto(shape=0.001, scale=0.001, dtype=dtype)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"shape", "scale"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    # --- Act ---
    _, new_params = q_function_strategy(start_component, state, block, optimizer=None)

    # --- Assert ---
    assert new_params[Pareto.PARAM_SHAPE] == pytest.approx(true_shape, rel=0.05, abs=0.2)
    assert new_params[Pareto.PARAM_SCALE] == pytest.approx(true_scale, rel=0.05)

    # Verify that the returned parameters have the correct dtype.
    assert isinstance(new_params[Pareto.PARAM_SHAPE], dtype)
    assert isinstance(new_params[Pareto.PARAM_SCALE], dtype)
