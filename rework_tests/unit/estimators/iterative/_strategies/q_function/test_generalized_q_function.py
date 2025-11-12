"""Tests for Q-function optimization strategy for arbitrary distribution"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from copy import copy

import numpy as np
import pytest
from rework_pysatl_mpest.core import MixtureModel, Parameter
from rework_pysatl_mpest.distributions import ContinuousDistribution
from rework_pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from rework_pysatl_mpest.estimators.iterative._strategies import q_function_strategy
from rework_pysatl_mpest.optimizers import Optimizer

# Helper classes for test isolation
# ---------------------------------


class DummyDistribution(ContinuousDistribution):
    """
    A simple dummy implementation of ContinuousDistribution for testing purposes.
    It has two parameters: 'param1' and 'param2'.
    """

    param1 = Parameter()
    param2 = Parameter()

    def __init__(self, param1: float, param2: float):
        super().__init__()
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
        return np.log(np.array([0.5] * len(X)))

    def log_gradients(self, X):
        return np.array([])

    def generate(self, size: int):
        return np.array([])

    def q_function(self, X: np.ndarray, H_j: np.ndarray) -> float:
        lpdf_values = self.lpdf(X)
        safe_lpdf = np.where(H_j == 0, 0.0, lpdf_values)
        return np.dot(H_j, safe_lpdf).item()


# --- Test Fixtures ---


@pytest.fixture
def mock_component() -> DummyDistribution:
    """Fixture that creates an instance of DummyDistribution."""

    return DummyDistribution(param1=1.0, param2=2.0)


@pytest.fixture
def mock_mixture(mocker, mock_component) -> MixtureModel:
    """Fixture that creates a mock MixtureModel."""

    mixture = mocker.create_autospec(MixtureModel, instance=True)
    mixture.components = (mock_component,)  # Make the component accessible
    return mixture


@pytest.fixture
def pipeline_state(mock_mixture) -> PipelineState:
    """
    Fixture that creates a more complete PipelineState with data.
    """

    state = PipelineState(
        X=np.array([10.0, 20.0, 30.0]),
        H=np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]),
        prev_mixture=None,
        curr_mixture=mock_mixture,
        error=None,
    )
    return state


@pytest.fixture
def optimization_block() -> OptimizationBlock:
    """
    Fixture that creates a more complete optimization block for the component with ID=0.
    """

    return OptimizationBlock(
        component_id=0, params_to_optimize={"param1", "param2"}, maximization_strategy=MaximizationStrategy.QFUNCTION
    )


@pytest.fixture
def mock_optimizer(mocker) -> Optimizer:
    """Fixture that creates a mock Optimizer object."""

    optimizer = mocker.create_autospec(Optimizer, instance=True)
    return optimizer


# Tests
# -----


def test_q_function_strategy_does_not_modify_original_component(
    mock_component, pipeline_state, optimization_block, mock_optimizer
):
    """
    Verifies that the original component object is not modified,
    as the function should work with its copy.
    """

    original_component = copy(mock_component)
    mock_optimizer.minimize.return_value = [99.0, 101.0]

    q_function_strategy(mock_component, pipeline_state, optimization_block, mock_optimizer)

    assert mock_component.param1 == original_component.param1, "param1 should not have been changed"
    assert mock_component.param2 == original_component.param2, "param2 should not have been changed"


def test_q_function_strategy_calls_optimizer_minimize_once(
    mock_component, pipeline_state, optimization_block, mock_optimizer
):
    """
    Verifies that the optimizer's minimize method is called exactly once.
    """

    mock_optimizer.minimize.return_value = [5.0, 10.0]
    q_function_strategy(mock_component, pipeline_state, optimization_block, mock_optimizer)
    mock_optimizer.minimize.assert_called_once()


def test_q_function_strategy_returns_correct_types(mock_component, pipeline_state, optimization_block, mock_optimizer):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, float]).
    """

    mock_optimizer.minimize.return_value = [5.0, 10.0]

    result = q_function_strategy(mock_component, pipeline_state, optimization_block, mock_optimizer)

    assert isinstance(result, tuple)
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, float)


def test_q_function_strategy_raises_value_error_if_h_is_none(
    mock_component, pipeline_state, optimization_block, mock_optimizer
):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed (is None).
    """

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
    mock_component, pipeline_state, mock_optimizer, params_to_optimize_in_block, expected_optimized_params_keys
):
    """
    Comprehensive test for correctness and interaction.
    Verifies that:
    - The correct component ID and a dictionary with new parameters are returned.
    - The optimizer is called with the correct initial values.
    - The target function logic works as expected (returns -q_function).
    """

    component_id = 0
    block = OptimizationBlock(
        component_id=component_id,
        params_to_optimize=params_to_optimize_in_block,
        maximization_strategy=MaximizationStrategy.QFUNCTION,
    )

    # Simulate the return value from the optimizer.
    num_params_to_optimize = len(expected_optimized_params_keys)
    mocked_new_params_vector = [p * 10.0 for p in range(1, num_params_to_optimize + 1)]
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


def test_q_function_strategy_respects_fixed_params(mock_component, pipeline_state, optimization_block, mock_optimizer):
    """
    Verifies that 'fixed' parameters are not passed to the
    optimizer, even if they are specified in the optimization_block.
    """

    # Fix 'param1', it should not be optimized
    mock_component.fix_param("param1")

    # The optimizer should return a value only for 'param2'
    new_param2_value = 99.0
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
