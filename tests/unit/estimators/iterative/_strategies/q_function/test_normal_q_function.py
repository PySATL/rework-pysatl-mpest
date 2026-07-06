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

# Test Fixtures
# -------------


@pytest.fixture()
def parametrized_normal_setup() -> tuple[Normal, PipelineState]:
    """
    Creates a parametrized fixture providing a Normal component (standard normal) and a
    corresponding PipelineState for various dtypes.
    """
    component = Normal(mu=0.0, sigma=1.0)

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


def test_q_function_normal_raises_value_error_if_h_is_none(parametrized_normal_setup):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed.
    """
    normal_component, state = parametrized_normal_setup
    state.H = None

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"mu", "sigma"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        q_function_strategy(normal_component, state, block, optimizer=None)


def test_q_function_normal_returns_correct_types(parametrized_normal_setup):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, FloatT]).
    """
    normal_component, pipeline_state = parametrized_normal_setup

    expected_len = 2

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"mu", "sigma"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    result = q_function_strategy(normal_component, pipeline_state, block, optimizer=None)

    assert isinstance(result, tuple)
    assert len(result) == expected_len
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, float)


def test_q_function_normal_zero_variance_update(parametrized_normal_setup):
    """
    Tests the branch where weighted variance is close to 0.
    """
    component, state = parametrized_normal_setup

    # All data points equal to current mean -> variance = 0
    state.X = np.array([10.0, 10.0])
    state.H = np.array([[1.0, 0.0], [1.0, 0.0]])

    component = Normal(mu=10.0, sigma=2.5)
    component.fix_param("mu")

    block = OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION)

    _, new_params = q_function_strategy(component, state, block, optimizer=None)

    # Should keep original scale
    assert new_params["sigma"] == component.sigma


@pytest.mark.parametrize(
    "params_to_optimize_in_block, fixed_params_on_component, expected_keys",
    [
        ({"mu", "sigma"}, set(), {"mu", "sigma"}),  # Optimize both, none are fixed
        ({"mu"}, set(), {"mu"}),  # Optimize only loc
        ({"sigma"}, set(), {"sigma"}),  # Optimize only scale
        ({"mu", "sigma"}, {"mu"}, {"sigma"}),  # Optimize both, but loc is fixed
        ({"mu", "sigma"}, {"sigma"}, {"mu"}),  # Optimize both, but scale is fixed
        ({"mu", "sigma"}, {"mu", "sigma"}, set()),  # Optimize both, but both are fixed
        ({"non_existent_param", "mu"}, set(), {"mu"}),  # Ignore non-existent params
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
    normal_component, pipeline_state = parametrized_normal_setup

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

    normal_component, pipeline_state = parametrized_normal_setup

    pipeline_state.H.fill(1e-10)  # Make all responsibilities negligible
    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={"shape", "mu", "sigma"},
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    _, new_params = q_function_strategy(normal_component, pipeline_state, block, optimizer=None)

    assert new_params == {}


# Property-Based Test with Hypothesis
# -----------------------------------


@st.composite
def normal_data_and_true_params(draw):
    """
    Generates a true Normal component and a data sample from it.
    Restricted to float64 to ensure stability during random generation tests.
    """
    # 1. Generate realistic parameters for the true distribution
    true_mu = draw(st.floats(min_value=-50, max_value=50))
    # Avoid extremely small scales to prevent random noise issues in tests
    true_sigma = draw(st.floats(min_value=0.5, max_value=20))
    true_component = Normal(mu=true_mu, sigma=true_sigma)

    np.random.seed(42)

    # 2. Generate a large data sample from this distribution
    sample_size = 10000
    X = true_component.generate(size=sample_size)

    return (X, true_mu, true_sigma)


@settings(max_examples=50, deadline=None)
@given(data=normal_data_and_true_params())
def test_q_function_normal_recovers_true_params_on_ideal_data(data):
    """
    Verifies that the analytical formulas can recover the original parameters
    from a sample when responsibilities are 1.0. This confirms the statistical
    validity of the implemented formulas in an ideal case (maximum likelihood estimation).
    """

    # --- Arrange ---
    X, true_mu, true_sigma = data

    # This is the key assumption for this test: perfect knowledge that all
    # data points belong to our component of interest (responsibilities are all 1.0).
    H_j = np.ones_like(X, dtype=np.float64)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T  # Simulate a 2-component mixture context

    # Use a starting component with completely different parameters to ensure
    # the update is based on data, not the initial guess.
    start_component = Normal(mu=-999.0, sigma=0.001)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"mu", "sigma"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    # --- Act ---
    _, new_params = q_function_strategy(start_component, state, block, optimizer=None)

    # --- Assert ---
    # Relaxed tolerance for scale because sample std deviation has variance itself
    assert new_params["mu"] == pytest.approx(true_mu, abs=1.0)
    assert new_params["sigma"] == pytest.approx(true_sigma, rel=0.1)
