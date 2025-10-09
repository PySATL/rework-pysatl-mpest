"""Tests for Q-function optimization strategy for Weibull distribution"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rework_pysatl_mpest.distributions import Weibull
from rework_pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from rework_pysatl_mpest.estimators.iterative._strategies import q_function_strategy
from rework_pysatl_mpest.optimizers import Optimizer

# Test Fixtures
# -------------


@pytest.fixture
def weibull_component() -> Weibull:
    """Fixture that creates a default Weibull component."""

    return Weibull(shape=2.0, loc=0.0, scale=1.0)


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


@pytest.fixture
def mock_optimizer(mocker) -> Optimizer:
    """Fixture that creates a mock Optimizer object."""

    optimizer = mocker.create_autospec(Optimizer, instance=True)
    # Default return value for optimizer
    optimizer.minimize.return_value = [1.5]  # e.g., for a single param like shape
    return optimizer


# Tests
# -----


def test_q_function_weibull_raises_value_error_if_h_is_none(weibull_component, mock_optimizer):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed.
    """

    state = PipelineState(X=np.array([1, 2, 3]), H=None, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={"shape", "loc", "scale"},
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        q_function_strategy(weibull_component, state, block, optimizer=mock_optimizer)


def test_q_function_weibull_returns_correct_types(weibull_component, pipeline_state, mock_optimizer):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, float]).
    """

    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={"shape", "loc", "scale"},
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )
    mock_optimizer.minimize.return_value = [1.9, 0.1]  # shape, loc

    result = q_function_strategy(weibull_component, pipeline_state, block, optimizer=mock_optimizer)

    assert isinstance(result, tuple)
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, float)


@pytest.mark.parametrize(
    "params_to_optimize_in_block, fixed_params_on_component, expected_numerical_params, expected_final_keys",
    [
        ({"shape", "loc", "scale"}, set(), {"shape", "loc"}, {"shape", "loc", "scale"}),
        ({"scale"}, set(), set(), {"scale"}),
        ({"shape", "loc"}, set(), {"shape", "loc"}, {"shape", "loc"}),
        ({"shape"}, set(), {"shape"}, {"shape"}),
        ({"loc"}, set(), {"loc"}, {"loc"}),
        ({"shape", "scale"}, set(), {"shape"}, {"shape", "scale"}),
        ({"loc", "scale"}, set(), {"loc"}, {"loc", "scale"}),
        ({"shape", "loc", "scale"}, {"shape"}, {"loc"}, {"loc", "scale"}),
        ({"shape", "loc", "scale"}, {"loc"}, {"shape"}, {"shape", "scale"}),
        ({"shape", "loc", "scale"}, {"scale"}, {"shape", "loc"}, {"shape", "loc"}),
        ({"shape", "loc", "scale"}, {"shape", "loc"}, set(), {"scale"}),
        (set(), set(), set(), set()),
    ],
)
def test_q_function_weibull_respects_params_and_calls_optimizer_correctly(
    mocker,
    weibull_component,
    pipeline_state,
    mock_optimizer,
    params_to_optimize_in_block,
    fixed_params_on_component,
    expected_numerical_params,
    expected_final_keys,
):
    """
    Verifies that the hybrid strategy for Weibull correctly identifies which parameters
    go to the numerical optimizer and which are solved analytically. It also checks
    that fixed parameters are always respected.
    """

    for param in fixed_params_on_component:
        weibull_component.fix_param(param)

    block = OptimizationBlock(
        component_id=0,
        params_to_optimize=params_to_optimize_in_block,
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    # 1. Define the plausible result that the generic numerical function should return.
    mock_numerical_result = {param: (i + 1) * 1.1 for i, param in enumerate(sorted(list(expected_numerical_params)))}

    # 2. Create a mock for the generic function itself.
    mock_generic_function = mocker.Mock(return_value=(0, mock_numerical_result))

    # 3. Patch the .dispatch method to return our new mock function.
    mocker.patch(
        "rework_pysatl_mpest.estimators.iterative._strategies.q_function_strategy.dispatch",
        return_value=mock_generic_function,
    )

    _, new_params = q_function_strategy(weibull_component, pipeline_state, block, mock_optimizer)

    assert set(new_params.keys()) == expected_final_keys

    if expected_numerical_params:
        # The generic strategy should have been called once.
        mock_generic_function.assert_called_once()
        # Check that it was called with a reduced block containing only the numerical params.
        args, _ = mock_generic_function.call_args
        called_block = args[2]  # The block is the 3rd argument
        assert isinstance(called_block, OptimizationBlock)
        assert set(called_block.params_to_optimize) == expected_numerical_params
    else:
        # If no params for numerical opt, the generic strategy should not be called.
        mock_generic_function.assert_not_called()


def test_q_function_weibull_handles_negligible_responsibility(weibull_component, pipeline_state, mock_optimizer):
    """
    Verifies that if a component's total responsibility (N_j) is near zero,
    its parameters are not updated.
    """

    pipeline_state.H.fill(1e-10)  # Make all responsibilities negligible
    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={"shape", "loc", "scale"},
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    _, new_params = q_function_strategy(weibull_component, pipeline_state, block, mock_optimizer)

    assert new_params == {}


def test_q_function_weibull_handles_invalid_loc_for_scale_calculation(
    weibull_component, pipeline_state, mock_optimizer
):
    """
    Verifies that if `loc` is >= X, the `scale` parameter is not updated to
    avoid numerical errors, and the original scale is kept.
    """

    # Set a loc that is larger than the first data point
    weibull_component.loc = 1.5
    original_scale = weibull_component.scale
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"scale"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    _, new_params = q_function_strategy(weibull_component, pipeline_state, block, optimizer=None)

    # The analytical formula should fail gracefully and return the original scale
    assert new_params[Weibull.PARAM_SCALE] == original_scale


# Property-Based Test with Hypothesis
# -----------------------------------


@st.composite
def weibull_data_and_true_params(draw):
    """Generates a true Weibull component and a data sample from it."""

    true_shape = draw(st.floats(min_value=1.0, max_value=5.0))
    true_loc = draw(st.floats(min_value=-10.0, max_value=10.0))
    true_scale = draw(st.floats(min_value=0.5, max_value=10.0))
    true_component = Weibull(shape=true_shape, loc=true_loc, scale=true_scale)

    sample_size = draw(st.integers(min_value=50000, max_value=50000))
    X = true_component.generate(size=sample_size)

    return (X, true_shape, true_loc, true_scale)


@settings(max_examples=30, deadline=None)
@given(data=weibull_data_and_true_params())
def test_q_function_weibull_analytical_scale_recovers_true_param(data):
    """
    Verifies that the analytical formula for the 'scale' parameter can recover
    the original parameter from a sample, given perfect responsibilities and
    the correct 'shape' and 'loc'. This confirms the statistical validity
    of the analytical part of the hybrid strategy.
    """

    # --- Arrange ---
    X, true_shape, true_loc, true_scale = data

    # Assume perfect knowledge: all data points belong to our component (H_j = 1.0)
    H_j = np.ones_like(X)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T  # Simulate a 2-component mixture

    # Create a component with the TRUE shape and loc, but a WRONG scale.
    # This isolates the test to only the analytical scale calculation.
    start_component = Weibull(shape=true_shape, loc=true_loc, scale=999.9)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)
    # We only want to optimize the scale parameter
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"scale"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    # --- Act ---
    _, new_params = q_function_strategy(start_component, state, block, optimizer=None)

    # --- Assert ---
    assert new_params[Weibull.PARAM_SCALE] == pytest.approx(true_scale, rel=0.05)
