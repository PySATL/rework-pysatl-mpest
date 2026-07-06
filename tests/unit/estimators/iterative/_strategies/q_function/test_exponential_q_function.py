"""Tests for Q-function optimization strategy for Exponential distribution"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pysatl_mpest.distributions import Exponential
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from pysatl_mpest.estimators.iterative._strategies import q_function_strategy

# Test Fixtures
# -------------


@pytest.fixture()
def parametrized_exponential_setup() -> tuple[Exponential, PipelineState]:
    """
    Creates a parametrized fixture providing an Exponential component and a
    corresponding PipelineState for various dtypes.
    """

    component = Exponential(lambda_=1.0)

    state = PipelineState(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        H=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]),
        prev_mixture=None,
        curr_mixture=None,
        error=None,
    )
    return component, state


# Tests
# -----


def test_q_function_exponential_raises_value_error_if_h_is_none(parametrized_exponential_setup):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed.
    """
    exponential_component, state = parametrized_exponential_setup
    state.H = None

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"lambda_"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        q_function_strategy(exponential_component, state, block, optimizer=None)


def test_q_function_exponential_returns_correct_types(parametrized_exponential_setup):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, FloatT]).
    """
    exponential_component, pipeline_state = parametrized_exponential_setup

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"lambda_"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    result = q_function_strategy(exponential_component, pipeline_state, block, optimizer=None)

    assert isinstance(result, tuple)
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, float)


def test_q_function_exponential_denominator_zero_rate_update(parametrized_exponential_setup):
    """
    Tests the branch where weighted average matches location (denominator ~ 0)
    in rate update logic.
    """
    component, state = parametrized_exponential_setup

    # X = 0.0 -> weighted average is 0.
    state.X = np.array([0.0])
    state.H = np.array([[1.0, 0.0]])  # 1 sample

    block = OptimizationBlock(0, {"lambda_"}, MaximizationStrategy.QFUNCTION)

    _, new_params = q_function_strategy(component, state, block, optimizer=None)

    # Should keep original rate to avoid division by zero
    assert "lambda_" in new_params
    assert new_params["lambda_"] == component.lambda_


@pytest.mark.parametrize(
    "params_to_optimize_in_block, fixed_params_on_component, expected_keys",
    [
        ({"lambda_"}, set(), {"lambda_"}),  # Optimize lambda_, none are fixed
        ({"lambda_"}, {"lambda_"}, set()),  # Optimize lambda_, but it is fixed
        ({"non_existent_param", "lambda_"}, set(), {"lambda_"}),  # Ignore non-existent params
        (set(), set(), set()),  # Optimize nothing
    ],
)
def test_q_function_exponential_respects_fixed_and_optimizable_params(
    parametrized_exponential_setup, params_to_optimize_in_block, fixed_params_on_component, expected_keys
):
    """
    Verifies that the strategy correctly identifies which parameters
    to update based on the optimization block and the component's fixed parameters.
    """
    exponential_component, pipeline_state = parametrized_exponential_setup

    for param in fixed_params_on_component:
        exponential_component.fix_param(param)

    block = OptimizationBlock(
        component_id=0,
        params_to_optimize=params_to_optimize_in_block,
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    _, new_params = q_function_strategy(exponential_component, pipeline_state, block, optimizer=None)

    assert set(new_params.keys()) == expected_keys


def test_q_function_exponential_handles_negligible_responsibility(parametrized_exponential_setup):
    """
    Verifies that if a component's total responsibility (N_j) is near zero,
    its parameters are not updated.
    """

    exponential_component, pipeline_state = parametrized_exponential_setup

    pipeline_state.H.fill(1e-10)  # Make all responsibilities negligible
    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={"lambda_"},
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    _, new_params = q_function_strategy(exponential_component, pipeline_state, block, optimizer=None)

    assert new_params == {}


# Property-Based Test with Hypothesis
# -----------------------------------


@st.composite
def exponential_data_and_true_params(draw):
    """
    Generates a true Exponential component and a data sample from it,
    all configured with a specific dtype.
    """
    true_rate = draw(st.floats(min_value=0.1, max_value=100))
    Exponential(lambda_=true_rate)

    # 2. Generate a data sample from this distribution
    sample_size = draw(st.integers(min_value=10000, max_value=10000))
    rng = np.random.default_rng(42)
    X = rng.exponential(scale=1.0 / true_rate, size=sample_size)

    return (X, true_rate)


@settings(max_examples=50)
@given(data=exponential_data_and_true_params())
def test_q_function_exponential_recovers_true_params_on_ideal_data(data):
    """
    Scenario 3 (Statistical Sanity Check): Verifies that the analytical formulas
    can recover the original parameters from a sample when responsibilities are 1.0.
    This confirms the statistical validity of the implemented formulas in an ideal case.
    """
    # --- Arrange ---
    X, true_rate = data

    # This is the key assumption for this test: perfect knowledge that all
    # data points belong to our component of interest.
    H_j = np.ones_like(X)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T  # Simulate a 2-component mixture

    # Use a starting component with completely different parameters
    start_component = Exponential(lambda_=0.001)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"lambda_"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    _, new_params = q_function_strategy(start_component, state, block, optimizer=None)

    # The estimated parameters won't be exactly the same due to sampling noise,
    # so we use pytest.approx with a relative tolerance.
    # For a large enough sample, the estimates should be reasonably close.
    assert new_params["lambda_"] == pytest.approx(true_rate, rel=0.05)
