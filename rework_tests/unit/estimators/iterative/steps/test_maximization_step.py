"""Tests for MaximizationStep"""

__author__ = "Danil Totmyanin"
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


@pytest.fixture
def mock_optimizer(mocker: MockerFixture) -> Optimizer:
    """Fixture to create a mock Optimizer."""

    return mocker.MagicMock(spec=Optimizer)


@pytest.fixture
def mock_components(mocker: MockerFixture) -> list[ContinuousDistribution]:
    """Fixture to create a list of mock distribution components."""

    comp1 = mocker.MagicMock(spec=ContinuousDistribution)
    comp2 = mocker.MagicMock(spec=ContinuousDistribution)
    return [comp1, comp2]


@pytest.fixture
def mock_mixture(mocker: MockerFixture, mock_components: list) -> MixtureModel:
    """Fixture to create a mock MixtureModel with two components."""

    mixture = mocker.MagicMock(spec=MixtureModel)
    mixture.__getitem__.side_effect = lambda i: mock_components[i]
    return mixture


@pytest.fixture
def pipeline_state(mock_mixture: MixtureModel) -> PipelineState:
    """Fixture to create a basic PipelineState."""

    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    H = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]])
    return PipelineState(X=X, H=H, prev_mixture=None, curr_mixture=mock_mixture, error=None)


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
        self, mock_optimizer: Optimizer, pipeline_state: PipelineState
    ):
        """
        Verifies that if state.H is None, the step sets an
        error and returns the state unmodified.
        """

        pipeline_state.H = None
        step = MaximizationStep(blocks=[], optimizer=mock_optimizer)

        result_state = step.run(pipeline_state)

        assert result_state is pipeline_state
        assert isinstance(result_state.error, ValueError)
        assert "Responsibility matrix H is not computed" in str(result_state.error)

    def test_run_calls_correct_strategy_and_updates_params(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_components: list,
        pipeline_state: PipelineState,
    ):
        """
        Verifies that the correct strategy is called with
        the correct arguments and that the component parameters are updated.
        """

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
        optimized_params = {"loc": 1.5, "rate": 2.5}
        mock_strategy.return_value = (0, optimized_params)

        # _strategies dict patching
        new_strategies = {MaximizationStrategy.QFUNCTION: mock_strategy}
        mocker.patch.object(MaximizationStep, "_strategies", new_strategies)

        target_component = mock_components[0]

        step.run(pipeline_state)

        # mock_strategy was called once
        mock_strategy.assert_called_once_with(target_component, pipeline_state, block, mock_optimizer)

        # The component parameters were updated
        # _update_components_params calls set_params_from_vector
        param_names = list(optimized_params.keys())
        param_values = list(optimized_params.values())
        target_component.set_params_from_vector.assert_called_once_with(param_names, param_values)

    def test_run_updates_mixture_weights_correctly(
        self, mocker: MockerFixture, mock_optimizer: Optimizer, pipeline_state: PipelineState
    ):
        """
        Verifies the correct update of mixture weights.
        """

        # Sum of responsibilities for component 0: 0.8 + 0.7 + 0.1 + 0.2 = 1.8
        # Sum of responsibilities for component 1: 0.2 + 0.3 + 0.9 + 0.8 = 2.2
        # New weights: [1.8/4, 2.2/4] = [0.45, 0.55]
        expected_new_weights = np.array([0.45, 0.55])
        expected_log_weights = np.log(expected_new_weights + 1e-30)

        # `log_weigths` patching
        p = mocker.PropertyMock()
        type(pipeline_state.curr_mixture).log_weights = p

        step = MaximizationStep(blocks=[], optimizer=mock_optimizer)
        step.run(pipeline_state)

        # Check that `log_weigths` setter was called once
        property_mock = p
        property_mock.assert_called_once()

        # Check that `log_weigths` was set correctly
        actual_log_weights = p.call_args.args[0]
        np.testing.assert_allclose(actual_log_weights, expected_log_weights)

    def test_run_processes_blocks_sequentially(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_components: list,
        pipeline_state: PipelineState,
    ):
        """
        Verifies that optimization blocks are processed sequentially in the specified order.
        """

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

        component0 = mock_components[0]
        component1 = mock_components[1]

        new_strategies = {MaximizationStrategy.QFUNCTION: mock_strategy}
        mocker.patch.object(MaximizationStep, "_strategies", new_strategies)

        # Act
        step.run(pipeline_state)

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
