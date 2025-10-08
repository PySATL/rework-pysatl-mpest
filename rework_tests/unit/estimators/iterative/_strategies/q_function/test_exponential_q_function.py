"""Tests for Q-function optimization strategy for Exponential distribution"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rework_pysatl_mpest.distributions import Exponential
from rework_pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from rework_pysatl_mpest.estimators.iterative._strategies import q_function_strategy

# Test Fixtures
# -------------


@pytest.fixture
def exponential_component() -> Exponential:
    """Fixture that creates a default Exponential component."""
    return Exponential(loc=0.0, rate=1.0)


@pytest.fixture
def pipeline_state() -> PipelineState:
    """Fixture that creates a basic PipelineState with some data."""
    state = PipelineState(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        H=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]),
        prev_mixture=None,
        curr_mixture=None,
        error=None,
    )
    return state


# Tests
# -----


def test_q_function_exponential_raises_value_error_if_h_is_none(exponential_component):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed.
    """

    state = PipelineState(X=np.array([1, 2, 3]), H=None, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "rate"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        q_function_strategy(exponential_component, state, block, optimizer=None)


def test_q_function_exponential_returns_correct_types(exponential_component, pipeline_state):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, float]).
    """

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "rate"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    result = q_function_strategy(exponential_component, pipeline_state, block, optimizer=None)

    assert isinstance(result, tuple)
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, float)


@pytest.mark.parametrize(
    "params_to_optimize_in_block, fixed_params_on_component, expected_keys",
    [
        ({"loc", "rate"}, set(), {"loc", "rate"}),  # Optimize both, none are fixed
        ({"loc"}, set(), {"loc"}),  # Optimize only loc
        ({"rate"}, set(), {"rate"}),  # Optimize only rate
        ({"loc", "rate"}, {"loc"}, {"rate"}),  # Optimize both, but loc is fixed
        ({"loc", "rate"}, {"rate"}, {"loc"}),  # Optimize both, but rate is fixed
        ({"loc", "rate"}, {"loc", "rate"}, set()),  # Optimize both, but both are fixed
        ({"non_existent_param", "loc"}, set(), {"loc"}),  # Ignore non-existent params
        (set(), set(), set()),  # Optimize nothing
    ],
)
def test_q_function_exponential_respects_fixed_and_optimizable_params(
    exponential_component, pipeline_state, params_to_optimize_in_block, fixed_params_on_component, expected_keys
):
    """
    Verifies that the strategy correctly identifies which parameters
    to update based on the optimization block and the component's fixed parameters.
    """

    for param in fixed_params_on_component:
        exponential_component.fix_param(param)

    block = OptimizationBlock(
        component_id=0,
        params_to_optimize=params_to_optimize_in_block,
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    _, new_params = q_function_strategy(exponential_component, pipeline_state, block, optimizer=None)

    assert set(new_params.keys()) == expected_keys


# Property-Based Test with Hypothesis
# -----------------------------------


@st.composite
def exponential_data_and_true_params(draw):
    """Generates a true component, and a data sample from it."""
    # 1. Generate realistic parameters for the true distribution
    true_loc = draw(st.floats(min_value=-100, max_value=100))
    true_rate = draw(st.floats(min_value=0.1, max_value=100))
    true_component = Exponential(loc=true_loc, rate=true_rate)

    # 2. Generate a data sample from this distribution
    sample_size = draw(st.integers(min_value=10000, max_value=10000))
    X = true_component.generate(size=sample_size)

    return (X, true_loc, true_rate)


@settings(max_examples=50)
@given(data=exponential_data_and_true_params())
def test_q_function_exponential_recovers_true_params_on_ideal_data(data):
    """
    Scenario 3 (Statistical Sanity Check): Verifies that the analytical formulas
    can recover the original parameters from a sample when responsibilities are 1.0.
    This confirms the statistical validity of the implemented formulas in an ideal case.
    """
    # --- Arrange ---
    X, true_loc, true_rate = data

    # This is the key assumption for this test: perfect knowledge that all
    # data points belong to our component of interest.
    H_j = np.ones_like(X)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T  # Simulate a 2-component mixture

    # Use a starting component with completely different parameters
    start_component = Exponential(loc=-999.0, rate=0.001)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "rate"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    _, new_params = q_function_strategy(start_component, state, block, optimizer=None)

    # The estimated parameters won't be exactly the same due to sampling noise,
    # so we use pytest.approx with a relative tolerance.
    # For a large enough sample, the estimates should be reasonably close.
    assert new_params[Exponential.PARAM_LOC] == pytest.approx(true_loc, abs=0.01)
    assert new_params[Exponential.PARAM_RATE] == pytest.approx(true_rate, rel=0.01)
