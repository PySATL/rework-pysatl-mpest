"""Tests for Q-function optimization strategy for arbitrary distribution"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from copy import copy

import numpy as np
import pytest
from pysatl_mpest.core import MixtureModel, Parameter
from pysatl_mpest.distributions import ContinuousDistribution
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from pysatl_mpest.estimators.iterative._strategies import q_function_strategy
from pysatl_mpest.exceptions import NumericalStabilityError
from pysatl_mpest.optimizers import Optimizer

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]

# Helper classes for test isolation
# ---------------------------------


class DummyDistribution(ContinuousDistribution):
    """
    A simple dummy implementation of ContinuousDistribution for testing purposes.
    It has two parameters: 'param1' and 'param2'.
    """

    param1 = Parameter()
    param2 = Parameter()

    def __init__(self, param1: float, param2: float, dtype: np.floating = np.float64):
        super().__init__(dtype=dtype)
        self.param1 = param1
        self.param2 = param2

    @property
    def name(self) -> str:
        return "Dummy"

    @property
    def params(self) -> set[str]:
        return {"param1", "param2"}

    def pdf(self, X):
        return np.array([])

    def ppf(self, P):
        return np.array([])

    def lpdf(self, X):
        return np.log(np.array([0.5] * len(X), dtype=self.dtype))

    def log_gradients(self, X):
        return np.array([])

    def generate(self, size: int):
        return np.array([])

    def q_function(self, X: np.ndarray, H_j: np.ndarray) -> float:
        lpdf_values = self.lpdf(X)
        safe_lpdf = np.where(H_j == 0, 0.0, lpdf_values)
        return np.dot(H_j, safe_lpdf).item()


# --- Test Fixtures ---


@pytest.fixture(params=DTYPES_TO_TEST)
def parametrized_setup(
    mocker, request
) -> tuple[ContinuousDistribution, PipelineState, OptimizationBlock, OptimizationBlock, np.floating]:
    """
    Fixture that creates a more complete PipelineState with data.
    """
    dtype = request.param

    component = DummyDistribution(param1=1.0, param2=2.0, dtype=dtype)

    mock_mixture = mocker.create_autospec(MixtureModel, instance=True)
    mock_mixture.components = (component,)

    state = PipelineState(
        X=np.array([10.0, 20.0, 30.0], dtype=dtype),
        H=np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]], dtype=dtype),
        prev_mixture=None,
        curr_mixture=mock_mixture,
        error=None,
    )
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"param1", "param2"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )

    optimizer = mocker.create_autospec(Optimizer, instance=True)
    optimizer.minimize.return_value = [dtype(99.0), dtype(101.0)]

    return component, state, block, optimizer, dtype


# Tests
# -----


def test_q_function_strategy_does_not_modify_original_component(parametrized_setup):
    """
    Verifies that the original component object is not modified,
    as the function should work with its copy.
    """

    mock_component, pipeline_state, optimization_block, mock_optimizer, dtype = parametrized_setup

    original_component = copy(mock_component)

    q_function_strategy(mock_component, pipeline_state, optimization_block, mock_optimizer)

    assert mock_component.param1 == original_component.param1, "param1 should not have been changed"
    assert mock_component.param2 == original_component.param2, "param2 should not have been changed"


def test_q_function_strategy_calls_optimizer_minimize_once(parametrized_setup):
    """
    Verifies that the optimizer's minimize method is called exactly once.
    """
    mock_component, pipeline_state, optimization_block, mock_optimizer, dtype = parametrized_setup

    q_function_strategy(mock_component, pipeline_state, optimization_block, mock_optimizer)
    mock_optimizer.minimize.assert_called_once()


def test_q_function_strategy_returns_correct_types(parametrized_setup):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, float]).
    """
    mock_component, pipeline_state, optimization_block, mock_optimizer, dtype = parametrized_setup
    result = q_function_strategy(mock_component, pipeline_state, optimization_block, mock_optimizer)

    assert isinstance(result, tuple)
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, dtype)


def test_q_function_strategy_raises_value_error_if_h_is_none(parametrized_setup):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed (is None).
    """

    mock_component, pipeline_state, optimization_block, mock_optimizer, _ = parametrized_setup
    pipeline_state.H = None

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        q_function_strategy(mock_component, pipeline_state, optimization_block, mock_optimizer)


@pytest.mark.parametrize(
    "params_to_optimize_in_block, expected_optimized_params_keys",
    [
        ({"param1", "param2"}, {"param1", "param2"}),
        ({"param1"}, {"param1"}),
        (set(), set()),  # Nothing to optimize
        ({"param1", "non_existent_param"}, {"param1"}),  # A non-existent parameter is ignored
    ],
)
def test_q_function_strategy_correctness_and_interaction(
    parametrized_setup, params_to_optimize_in_block, expected_optimized_params_keys
):
    """
    Comprehensive test for correctness and interaction.
    Verifies that:
    - The correct component ID and a dictionary with new parameters are returned.
    - The optimizer is called with the correct initial values.
    - The target function logic works as expected (returns -q_function).
    """

    mock_component, pipeline_state, optimization_block, mock_optimizer, dtype = parametrized_setup

    component_id = 0
    block = OptimizationBlock(
        component_id=component_id,
        params_to_optimize=params_to_optimize_in_block,
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    # Simulate the return value from the optimizer.
    num_params_to_optimize = len(expected_optimized_params_keys)
    mocked_new_params_vector = [dtype(p * 10.0) for p in range(1, num_params_to_optimize + 1)]
    mock_optimizer.minimize.return_value = mocked_new_params_vector

    sorted_keys = sorted(list(expected_optimized_params_keys))
    expected_result_dict = dict(zip(sorted_keys, mocked_new_params_vector))

    result_id, new_params_dict = q_function_strategy(mock_component, pipeline_state, block, mock_optimizer)

    # Check the return values
    assert result_id == component_id
    assert new_params_dict == expected_result_dict

    # Check that minimize method called once
    mock_optimizer.minimize.assert_called_once()

    # Get minimize method args
    args, _ = mock_optimizer.minimize.call_args
    target_func, initial_vector = args

    # Check that the initial vector contains the correct values
    expected_initial_vector = mock_component.get_params_vector(sorted_keys)
    assert np.array_equal(initial_vector, expected_initial_vector)

    # Check that the target function correctly calls q_function with a negative sign
    test_vector = [1.0] * num_params_to_optimize
    temp_comp = copy(mock_component)
    temp_comp.set_params_from_vector(sorted_keys, test_vector)
    expected_q_value = temp_comp.q_function(pipeline_state.X, pipeline_state.H[:, component_id])

    assert target_func(test_vector) == -expected_q_value


def test_q_function_strategy_respects_fixed_params(parametrized_setup):
    """
    Verifies that 'fixed' parameters are not passed to the
    optimizer, even if they are specified in the optimization_block.
    """
    mock_component, pipeline_state, optimization_block, mock_optimizer, dtype = parametrized_setup

    # Fix 'param1', it should not be optimized
    mock_component.fix_param("param1")

    # The optimizer should return a value only for 'param2'
    new_param2_value = dtype(99.0)
    mock_optimizer.minimize.return_value = [new_param2_value]

    _, new_params_dict = q_function_strategy(mock_component, pipeline_state, optimization_block, mock_optimizer)

    # Check that the result only contains 'param2'
    assert "param1" not in new_params_dict
    assert new_params_dict == {"param2": new_param2_value}

    # Check that `minimize` was called with a vector of one element
    mock_optimizer.minimize.assert_called_once()
    args, _ = mock_optimizer.minimize.call_args
    _, initial_vector = args
    assert len(initial_vector) == 1
    assert initial_vector[0] == mock_component.param2


def test_q_function_strategy_handles_numerical_overflow():
    """
    Verifies that the generic strategy correctly registers an error
    when a numerical overflow occurs within the target function.
    """
    # --- Arrange ---
    dtype = np.float16

    class OverflowDistribution(DummyDistribution):
        def lpdf(self, X):
            return np.array([-40000, -40000], dtype=self.dtype)

    component = OverflowDistribution(1.0, 2.0, dtype=dtype)
    state = PipelineState(
        X=np.array([1, 2], dtype=dtype),
        H=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=dtype),
        curr_mixture=None,
        prev_mixture=None,
        error=None,
    )
    block = OptimizationBlock(0, {"param1", "param2"}, MaximizationStrategy.QFUNCTION)

    class MockOptimizer(Optimizer):
        def minimize(self, target, initial_vector):
            target(initial_vector)
            return initial_vector

    # --- Act ---
    q_function_strategy(component, state, block, MockOptimizer())

    # --- Assert ---
    assert state.error is not None
    assert isinstance(state.error, NumericalStabilityError)
