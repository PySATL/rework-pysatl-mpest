"""Tests for Q-function optimization strategy for Pareto type 1 distribution"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rework_pysatl_mpest.distributions import Pareto
from rework_pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from rework_pysatl_mpest.estimators.iterative._strategies import q_function_strategy

# Test Fixtures
# -------------


@pytest.fixture
def pareto_component() -> Pareto:
    """Fixture that creates a default Pareto component."""

    return Pareto(shape=1.0, scale=0.5)


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


def test_q_function_pareto_raises_value_error_if_h_is_none(pareto_component):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed.
    """

    state = PipelineState(X=np.array([1, 2, 3]), H=None, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"shape", "scale"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        q_function_strategy(pareto_component, state, block, optimizer=None)


def test_q_function_pareto_returns_correct_types(pareto_component, pipeline_state):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, float]).
    """

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
        assert isinstance(value, float)


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
    pareto_component, pipeline_state, params_to_optimize_in_block, fixed_params_on_component, expected_keys
):
    """
    Verifies that the strategy correctly identifies which parameters
    to update based on the optimization block and the component's fixed parameters.
    """

    for param in fixed_params_on_component:
        pareto_component.fix_param(param)

    block = OptimizationBlock(
        component_id=0,
        params_to_optimize=params_to_optimize_in_block,
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    _, new_params = q_function_strategy(pareto_component, pipeline_state, block, optimizer=None)

    assert set(new_params.keys()) == expected_keys


# Property-Based Test with Hypothesis
# -----------------------------------


@st.composite
def pareto_data_and_true_params(draw):
    """Generates a true Pareto component and a data sample from it."""

    # 1. Generate realistic parameters for the true distribution
    true_shape = draw(st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
    true_scale = draw(st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
    true_component = Pareto(shape=true_shape, scale=true_scale)

    # 2. Generate a large data sample from this distribution
    X = true_component.generate(size=1000000)

    return (X, true_shape, true_scale)


@settings(max_examples=50, deadline=None)
@given(data=pareto_data_and_true_params())
def test_q_function_pareto_recovers_true_params_on_ideal_data(data):
    """
    Verifies that the analytical formulas can recover the original parameters
    from a sample when responsibilities are 1.0. This confirms the statistical
    validity of the implemented formulas in an ideal case (maximum likelihood estimation).
    """

    # --- Arrange ---
    X, true_shape, true_scale = data

    # This is the key assumption for this test: perfect knowledge that all
    # data points belong to our component of interest (responsibilities are all 1.0).
    H_j = np.ones_like(X)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T  # Simulate a 2-component mixture context

    # Use a starting component with completely different parameters to ensure
    # the update is based on data, not the initial guess.
    start_component = Pareto(shape=0.001, scale=0.001)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"shape", "scale"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    # --- Act ---
    _, new_params = q_function_strategy(start_component, state, block, optimizer=None)

    # --- Assert ---
    assert new_params[Pareto.PARAM_SHAPE] == pytest.approx(true_shape, rel=0.05, abs=0.2)
    assert new_params[Pareto.PARAM_SCALE] == pytest.approx(true_scale, rel=0.05)
