"""Tests for Q-function optimization strategy for Normal distribution"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pysatl_mpest.distributions import Normal
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from pysatl_mpest.estimators.iterative._strategies import q_function_strategy
from pysatl_mpest.exceptions import NumericalStabilityError

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]

# Test Fixtures
# -------------


@pytest.fixture(params=DTYPES_TO_TEST)
def parametrized_normal_setup(request) -> tuple[Normal, PipelineState, np.floating]:
    """
    Creates a parametrized fixture providing a Normal component (standard normal) and a
    corresponding PipelineState for various dtypes.
    """
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


# Tests
# -----


def test_q_function_normal_raises_value_error_if_h_is_none(parametrized_normal_setup):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed.
    """
    normal_component, state, _ = parametrized_normal_setup
    state.H = None

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "scale"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        q_function_strategy(normal_component, state, block, optimizer=None)


def test_q_function_normal_returns_correct_types(parametrized_normal_setup):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, DType]).
    """
    normal_component, pipeline_state, dtype = parametrized_normal_setup

    expected_len = 2

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "scale"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    result = q_function_strategy(normal_component, pipeline_state, block, optimizer=None)

    assert isinstance(result, tuple)
    assert len(result) == expected_len
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, dtype)


def test_q_function_normal_zero_variance_update(parametrized_normal_setup):
    """
    Tests the branch where weighted variance is close to 0.
    """
    component, state, dtype = parametrized_normal_setup

    # All data points equal to current mean -> variance = 0
    state.X = np.array([10.0, 10.0], dtype=dtype)
    state.H = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=dtype)

    component.loc = dtype(10.0)
    component.scale = dtype(2.5)

    component.fix_param("loc")

    block = OptimizationBlock(0, {"scale"}, MaximizationStrategy.QFUNCTION)

    _, new_params = q_function_strategy(component, state, block, optimizer=None)

    # Should keep original scale
    assert new_params["scale"] == component.scale


@pytest.mark.parametrize(
    "params_to_optimize_in_block, fixed_params_on_component, expected_keys",
    [
        ({"loc", "scale"}, set(), {"loc", "scale"}),  # Optimize both, none are fixed
        ({"loc"}, set(), {"loc"}),  # Optimize only loc
        ({"scale"}, set(), {"scale"}),  # Optimize only scale
        ({"loc", "scale"}, {"loc"}, {"scale"}),  # Optimize both, but loc is fixed
        ({"loc", "scale"}, {"scale"}, {"loc"}),  # Optimize both, but scale is fixed
        ({"loc", "scale"}, {"loc", "scale"}, set()),  # Optimize both, but both are fixed
        ({"non_existent_param", "loc"}, set(), {"loc"}),  # Ignore non-existent params
        (set(), set(), set()),  # Optimize nothing
    ],
)
def test_q_function_normal_respects_fixed_and_optimizable_params(
    parametrized_normal_setup, params_to_optimize_in_block, fixed_params_on_component, expected_keys
):
    """
    Verifies that the strategy correctly identifies which parameters
    to update based on the optimization block and the component's fixed parameters.
    """
    normal_component, pipeline_state, _ = parametrized_normal_setup

    for param in fixed_params_on_component:
        normal_component.fix_param(param)

    block = OptimizationBlock(
        component_id=0,
        params_to_optimize=params_to_optimize_in_block,
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    _, new_params = q_function_strategy(normal_component, pipeline_state, block, optimizer=None)

    assert set(new_params.keys()) == expected_keys


def test_q_function_normal_handles_negligible_responsibility(parametrized_normal_setup):
    """
    Verifies that if a component's total responsibility (N_j) is near zero,
    its parameters are not updated.
    """

    normal_component, pipeline_state, dtype = parametrized_normal_setup

    pipeline_state.H.fill(dtype(1e-10))  # Make all responsibilities negligible
    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={"shape", "loc", "scale"},
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    _, new_params = q_function_strategy(normal_component, pipeline_state, block, optimizer=None)

    assert new_params == {}


@pytest.mark.parametrize("param_to_optimize_in_block", ["loc", "scale"])
def test_q_function_exponential_handles_numerical_overflow(param_to_optimize_in_block):
    """
    Verifies that if a numerical overflow occurs during calculations,
    a `NumericalStabilityError` is correctly registered in the pipeline state.
    """
    # --- Arrange ---
    # Use float16, which has a small range, and large values to force an overflow.
    # The max value for float16 is ~65504. The sum of X will exceed this.
    dtype = np.float16
    component = Normal(loc=0.0, scale=1.0, dtype=dtype)
    X = np.array([60000, 60000], dtype=dtype)
    H_j = np.array([1.0, 1.0], dtype=dtype)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T
    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={param_to_optimize_in_block},
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )
    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)

    _, new_params = q_function_strategy(component, state, block, optimizer=None)

    assert state.error is not None
    assert isinstance(state.error, NumericalStabilityError)
    assert "Overflow detected during Q-function optimization" in str(state.error)
    assert new_params == {}


# Property-Based Test with Hypothesis
# -----------------------------------


@st.composite
def normal_data_and_true_params(draw, dtype_strategy=st.sampled_from([np.float32, np.float64])):
    """
    Generates a true Normal component and a data sample from it,
    all configured with a specific dtype.
    """
    dtype = draw(dtype_strategy)

    # 1. Generate realistic parameters for the true distribution
    true_loc = draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
    true_scale = draw(st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
    true_component = Normal(loc=true_loc, scale=true_scale, dtype=dtype)

    # 2. Generate a large data sample from this distribution
    X = true_component.generate(size=1000000)

    return (X, dtype(true_loc), dtype(true_scale), dtype)


@settings(max_examples=50, deadline=None)
@given(data=normal_data_and_true_params())
def test_q_function_normal_recovers_true_params_on_ideal_data(data):
    """
    Verifies that the analytical formulas can recover the original parameters
    from a sample when responsibilities are 1.0. This confirms the statistical
    validity of the implemented formulas in an ideal case (maximum likelihood estimation).
    """

    # --- Arrange ---
    X, true_loc, true_scale, dtype = data

    # This is the key assumption for this test: perfect knowledge that all
    # data points belong to our component of interest (responsibilities are all 1.0).
    H_j = np.ones_like(X, dtype=dtype)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T  # Simulate a 2-component mixture context

    # Use a starting component with completely different parameters to ensure
    # the update is based on data, not the initial guess.
    start_component = Normal(loc=-999.0, scale=0.001, dtype=dtype)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "scale"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    # --- Act ---
    _, new_params = q_function_strategy(start_component, state, block, optimizer=None)

    # --- Assert ---
    tolerance = {"rel": 0.05, "abs": 0.2} if dtype == np.float64 else {"rel": 2.0, "abs": 1.0}

    assert new_params[Normal.PARAM_LOC] == pytest.approx(true_loc, **tolerance)
    assert new_params[Normal.PARAM_SCALE] == pytest.approx(true_scale, **tolerance)

    # Verify that the returned parameters have the correct dtype.
    assert isinstance(new_params[Normal.PARAM_LOC], dtype)
    assert isinstance(new_params[Normal.PARAM_SCALE], dtype)
