"""Tests for MaximizationStep"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from pytest_mock import MockerFixture
from rework_pysatl_mpest.core import MixtureModel
from rework_pysatl_mpest.distributions import ContinuousDistribution
from rework_pysatl_mpest.estimators.iterative import (
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    PipelineState,
    PipelineStep,
)
from rework_pysatl_mpest.optimizers import Optimizer

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]


@pytest.fixture
def mock_optimizer(mocker: MockerFixture) -> Optimizer:
    """Fixture to create a mock Optimizer."""

    return mocker.MagicMock(spec=Optimizer)


@pytest.fixture(params=DTYPES_TO_TEST)
def parametrized_state(request, mocker: MockerFixture) -> PipelineState:
    """
    Creates a parametrized PipelineState fixture for various dtypes.

    This fixture runs for each data type in DTYPES_TO_TEST, constructing a
    mock MixtureModel and PipelineState where all relevant arrays (X, H) and
    model attributes share the same parametrized dtype. This enables robust
    testing of data type preservation.
    """
    dtype = request.param

    comp1 = mocker.MagicMock(spec=ContinuousDistribution)
    comp2 = mocker.MagicMock(spec=ContinuousDistribution)
    mock_components = [comp1, comp2]

    mixture = mocker.MagicMock(spec=MixtureModel)
    mixture.__getitem__.side_effect = lambda i: mock_components[i]
    mixture.__iter__.return_value = iter(mock_components)
    mixture.dtype = dtype

    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=dtype)
    H = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]], dtype=dtype)
    return PipelineState(X=X, H=H, prev_mixture=None, curr_mixture=mixture, error=None)


class TestMaximizationStep:
    """Tests for the MaximizationStep class."""

    def test_maximization_step_initialization(self, mock_optimizer: Optimizer):
        """
        Verifies the correct initialization of MaximizationStep.
        """

        blocks = [
            OptimizationBlock(
                component_id=0,
                params_to_optimize={"loc", "rate"},
                maximization_strategy=MaximizationStrategy.QFUNCTION,
            )
        ]

        step = MaximizationStep(blocks=blocks, optimizer=mock_optimizer)

        assert step.blocks == blocks
        assert step.optimizer == mock_optimizer
        assert isinstance(step, PipelineStep)

    def test_available_next_steps_property(self, mock_optimizer: Optimizer):
        """
        Verifies that the available_next_steps property returns the correct step type.
        """

        step = MaximizationStep(blocks=[], optimizer=mock_optimizer)

        next_steps = step.available_next_steps

        assert len(next_steps) == 1
        assert next_steps[0] == ExpectationStep

    def test_run_with_none_h_matrix_sets_error_and_returns_state(
        self, mock_optimizer: Optimizer, parametrized_state: PipelineState
    ):
        """
        Verifies that if state.H is None, the step sets an
        error and returns the state unmodified.
        """
        state = parametrized_state

        state.H = None
        step = MaximizationStep(blocks=[], optimizer=mock_optimizer)

        result_state = step.run(state)

        assert result_state is state
        assert isinstance(result_state.error, ValueError)
        assert "Responsibility matrix H is not computed" in str(result_state.error)

    def test_run_calls_correct_strategy_and_updates_params(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        parametrized_state: PipelineState,
    ):
        """
        Verifies that the correct strategy is called with
        the correct arguments and that the component parameters are updated.
        """
        state = parametrized_state
        dtype = state.curr_mixture.dtype

        block = OptimizationBlock(
            component_id=0,
            params_to_optimize={"loc", "rate"},
            maximization_strategy=MaximizationStrategy.QFUNCTION,
        )
        step = MaximizationStep(blocks=[block], optimizer=mock_optimizer)

        # Strategy patching
        mock_strategy = mocker.patch(
            "rework_pysatl_mpest.estimators.iterative.steps.maximization_step.q_function_strategy"
        )
        optimized_params = {"loc": dtype(1.5), "rate": dtype(2.5)}
        mock_strategy.return_value = (0, optimized_params)

        # _strategies dict patching
        new_strategies = {MaximizationStrategy.QFUNCTION: mock_strategy}
        mocker.patch.object(MaximizationStep, "_strategies", new_strategies)

        target_component = state.curr_mixture[0]

        step.run(state)

        # mock_strategy was called once
        mock_strategy.assert_called_once_with(target_component, state, block, mock_optimizer)

        # The component parameters were updated
        # _update_components_params calls set_params_from_vector
        param_names = list(optimized_params.keys())
        param_values = list(optimized_params.values())
        target_component.set_params_from_vector.assert_called_once_with(param_names, param_values)

        # type correct
        for value in param_values:
            assert isinstance(value, dtype)

    def test_run_updates_mixture_weights_correctly(
        self, mocker: MockerFixture, mock_optimizer: Optimizer, parametrized_state: PipelineState
    ):
        """
        Verifies the correct update of mixture weights.
        """
        state = parametrized_state
        dtype = state.curr_mixture.dtype

        # Sum of responsibilities for component 0: 0.8 + 0.7 + 0.1 + 0.2 = 1.8
        # Sum of responsibilities for component 1: 0.2 + 0.3 + 0.9 + 0.8 = 2.2
        # New weights: [1.8/4, 2.2/4] = [0.45, 0.55]
        expected_new_weights = np.array([0.45, 0.55], dtype=dtype)
        expected_log_weights = np.log(expected_new_weights + 1e-30).astype(dtype)

        # `log_weigths` patching
        p = mocker.PropertyMock()
        type(state.curr_mixture).log_weights = p

        step = MaximizationStep(blocks=[], optimizer=mock_optimizer)
        step.run(state)

        # Check that `log_weigths` setter was called once
        property_mock = p
        property_mock.assert_called_once()

        # Check that `log_weigths` was set correctly
        actual_log_weights = p.call_args.args[0]
        atol = 1e-4 if dtype == np.float16 else 1e-7
        np.testing.assert_allclose(actual_log_weights, expected_log_weights, atol=atol)

        # type correct
        assert actual_log_weights.dtype == dtype

    def test_run_processes_blocks_sequentially(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        parametrized_state: PipelineState,
    ):
        """
        Verifies that optimization blocks are processed sequentially in the specified order.
        """
        state = parametrized_state

        expected_call_count = 2

        # Arrange
        block1 = OptimizationBlock(
            component_id=1,
            params_to_optimize={"rate"},
            maximization_strategy=MaximizationStrategy.QFUNCTION,
        )
        block0 = OptimizationBlock(
            component_id=0,
            params_to_optimize={"loc"},
            maximization_strategy=MaximizationStrategy.QFUNCTION,
        )
        # Order matters: block1 first, then block0
        step = MaximizationStep(blocks=[block1, block0], optimizer=mock_optimizer)

        mock_strategy = mocker.patch(
            "rework_pysatl_mpest.estimators.iterative.steps.maximization_step.q_function_strategy"
        )
        # Configure the return values for each call
        mock_strategy.side_effect = [
            (1, {"rate": 1.1}),  # Return value for block1
            (0, {"loc": 2.2}),  # Return value for block0
        ]

        component0, component1 = state.curr_mixture

        new_strategies = {MaximizationStrategy.QFUNCTION: mock_strategy}
        mocker.patch.object(MaximizationStep, "_strategies", new_strategies)

        # Act
        step.run(state)

        # Assert
        assert mock_strategy.call_count == expected_call_count

        # Assert the first call: should be for block1 and component1
        first_call_args = mock_strategy.call_args_list[0].args
        assert first_call_args[0] is component1
        assert first_call_args[2] is block1

        # Assert the second call: should be for block0 and component0
        second_call_args = mock_strategy.call_args_list[1].args
        assert second_call_args[0] is component0
        assert second_call_args[2] is block0

        # Assert that parameters were updated for both components
        component1.set_params_from_vector.assert_called_once_with(["rate"], [1.1])
        component0.set_params_from_vector.assert_called_once_with(["loc"], [2.2])

    def test_clear_after_prune_removes_blocks_for_pruned_components(self, mock_optimizer: Optimizer):
        """Tests that clear_after_prune removes optimization blocks for pruned components."""

        CONST = 2

        blocks = [
            OptimizationBlock(0, {"loc"}, "q_function"),
            OptimizationBlock(1, {"rate"}, "q_function"),
            OptimizationBlock(2, {"loc", "rate"}, "q_function"),
        ]

        step = MaximizationStep(blocks=blocks, optimizer=mock_optimizer)

        # Remove component 1
        removed_indices = [1]
        step.clear_after_prune(removed_indices)

        # Should have 2 blocks left
        assert len(step.blocks) == CONST
        # Blocks should be for original components 0 and 2
        assert step.blocks[0].component_id == 0
        assert step.blocks[1].component_id == 1  # Reindexed from 2 to 1

    def test_clear_after_prune_preserves_optimization_parameters(self, mock_optimizer: Optimizer):
        """Tests that clear_after_prune preserves optimization parameters for remaining blocks."""

        blocks = [
            OptimizationBlock(0, {"loc"}, "q_function"),
            OptimizationBlock(1, {"rate", "scale"}, "q_function"),
            OptimizationBlock(2, {"loc", "rate"}, "q_function"),
        ]

        step = MaximizationStep(blocks=blocks, optimizer=mock_optimizer)

        # Remove component 1
        removed_indices = [1]
        step.clear_after_prune(removed_indices)

        # Check that optimization parameters are preserved
        assert step.blocks[0].params_to_optimize == {"loc"}
        assert step.blocks[1].params_to_optimize == {"loc", "rate"}

    def test_clear_after_prune_with_empty_removal(self, mock_optimizer: Optimizer):
        """Tests that clear_after_prune does nothing when no components are removed."""

        blocks = [OptimizationBlock(0, {"loc"}, "q_function"), OptimizationBlock(1, {"rate"}, "q_function")]

        step = MaximizationStep(blocks=blocks, optimizer=mock_optimizer)
        original_blocks = list(step.blocks)  # Create a copy

        step.clear_after_prune([])

        # Blocks should remain unchanged
        assert step.blocks == original_blocks

    def test_clear_after_prune_with_none_blocks(self, mock_optimizer: Optimizer):
        """Tests that clear_after_prune handles None blocks gracefully."""

        step = MaximizationStep(blocks=[], optimizer=mock_optimizer)

        # Should not raise an error
        step.clear_after_prune([0, 1])
        assert step.blocks == []

    def test_clear_after_prune_with_multiple_removals(self):
        """Tests clear_after_prune with multiple components removed."""
        NUMBER_OF_BLOCKS = 2
        blocks = [
            OptimizationBlock(0, {"loc"}, "q_function"),
            OptimizationBlock(1, {"rate"}, "q_function"),
            OptimizationBlock(2, {"scale"}, "q_function"),
            OptimizationBlock(3, {"loc", "rate"}, "q_function"),
        ]

        step = MaximizationStep(blocks=blocks, optimizer=None)

        # Remove components 1 and 3
        removed_indices = [1, 3]
        step.clear_after_prune(removed_indices)

        assert len(step.blocks) == NUMBER_OF_BLOCKS
        assert step.blocks[0].component_id == 0  # Originally component 0
        assert step.blocks[1].component_id == 1  # Originally component 2, now reindexed to 1

        assert step.blocks[0].params_to_optimize == {"loc"}
        assert step.blocks[1].params_to_optimize == {"scale"}

    def test_run_aborts_if_error_occurs_in_strategy(self, mock_optimizer, parametrized_state, mocker):
        """
        Tests that if a strategy returns an error state, subsequent blocks are ignored.
        """
        state = parametrized_state
        block1 = OptimizationBlock(0, {"loc"}, MaximizationStrategy.QFUNCTION)
        block2 = OptimizationBlock(1, {"loc"}, MaximizationStrategy.QFUNCTION)

        step = MaximizationStep([block1, block2], mock_optimizer)

        # Strategy that sets error on first call
        def failing_strategy(*args):
            state.error = ValueError("Fail")
            return 0, {}

        mocker.patch.object(MaximizationStep, "_strategies", {MaximizationStrategy.QFUNCTION: failing_strategy})
        spy = mocker.spy(step, "_update_components_params")

        result = step.run(state)

        # Should return state with error
        assert isinstance(result, PipelineState)
        assert str(state.error) == "Fail"
        # Params update should NOT be called because we returned early
        spy.assert_not_called()
