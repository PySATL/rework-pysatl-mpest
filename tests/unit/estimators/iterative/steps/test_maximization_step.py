"""Tests for MaximizationStep"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from unittest.mock import call

import numpy as np
import pytest
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import (
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    PipelineState,
    PipelineStep,
)
from pysatl_mpest.optimizers import Optimizer
from pytest_mock import MockerFixture
from tests.helpers.families import family


@pytest.fixture
def mock_optimizer(mocker: MockerFixture) -> Optimizer:
    """Fixture to create a mock Optimizer."""

    return mocker.MagicMock(spec=Optimizer)


@pytest.fixture
def parametrized_state(mocker: MockerFixture) -> PipelineState:
    """
    Creates a PipelineState fixture for testing.

    This fixture constructs a mock MixtureModel and PipelineState where
    all relevant arrays (X, H) and model attributes use np.float64.
    """

    comp1 = mocker.MagicMock(spec=ParametricFamilyDistribution)
    comp2 = mocker.MagicMock(spec=ParametricFamilyDistribution)
    mock_components = [comp1, comp2]

    mixture = mocker.MagicMock(spec=MixtureModel)
    mixture.__getitem__.side_effect = lambda i: mock_components[i]
    mixture.__iter__.return_value = iter(mock_components)

    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    H = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]], dtype=np.float64)
    return PipelineState(X=X, H=H, prev_mixture=None, curr_mixture=mixture, error=None)


@pytest.fixture
def real_state() -> PipelineState:
    """Create a state with real Normal components and unequal responsibility sums."""
    normal = family(FamilyName.NORMAL)
    mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=2.0)])
    return PipelineState(
        X=np.array([1.0, 2.0, 3.0, 4.0]),
        H=np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]]),
        prev_mixture=None,
        curr_mixture=mixture,
        error=None,
    )


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
        assert str(result_state.error) == "Responsibility matrix H is not computed."

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

        block = OptimizationBlock(
            component_id=0,
            params_to_optimize={"loc", "rate"},
            maximization_strategy=MaximizationStrategy.QFUNCTION,
        )
        step = MaximizationStep(blocks=[block], optimizer=mock_optimizer)

        # Strategy patching
        mock_strategy = mocker.patch("pysatl_mpest.estimators.iterative.steps.maximization_step.q_function_strategy")
        optimized_params = {"loc": np.float64(1.5), "rate": np.float64(2.5)}
        mock_strategy.return_value = (0, optimized_params)

        # _strategies dict patching
        new_strategies = {MaximizationStrategy.QFUNCTION: mock_strategy}
        mocker.patch.object(MaximizationStep, "_strategies", new_strategies)

        updated_component = mocker.MagicMock(spec=ParametricFamilyDistribution)
        update = mocker.patch.object(MaximizationStep, "_update_components_params", return_value=updated_component)

        target_component = state.curr_mixture[0]

        step.run(state)

        # mock_strategy was called once
        mock_strategy.assert_called_once_with(target_component, state, block, mock_optimizer)
        # the returned parameters were applied to the component, and the result replaced it in the mixture
        update.assert_called_once_with(target_component, optimized_params)
        state.curr_mixture.__setitem__.assert_called_once_with(0, updated_component)

    def test_run_updates_mixture_weights_correctly(
        self, mocker: MockerFixture, mock_optimizer: Optimizer, parametrized_state: PipelineState
    ):
        """
        Verifies the correct update of mixture weights.
        """
        state = parametrized_state

        # Sum of responsibilities for component 0: 0.8 + 0.7 + 0.1 + 0.2 = 1.8
        # Sum of responsibilities for component 1: 0.2 + 0.3 + 0.9 + 0.8 = 2.2
        # New weights: [1.8/4, 2.2/4] = [0.45, 0.55]
        expected_new_weights = np.array([0.45, 0.55], dtype=np.float64)
        expected_log_weights = np.log(expected_new_weights + 1e-30).astype(np.float64)

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
        np.testing.assert_allclose(actual_log_weights, expected_log_weights, atol=1e-7)

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

        mock_strategy = mocker.patch("pysatl_mpest.estimators.iterative.steps.maximization_step.q_function_strategy")
        # Configure the return values for each call
        mock_strategy.side_effect = [
            (1, {"rate": 1.1}),  # Return value for block1
            (0, {"loc": 2.2}),  # Return value for block0
        ]

        component0, component1 = state.curr_mixture

        new_strategies = {MaximizationStrategy.QFUNCTION: mock_strategy}
        mocker.patch.object(MaximizationStep, "_strategies", new_strategies)
        mocker.patch.object(
            MaximizationStep, "_update_components_params", side_effect=lambda component, params: component
        )

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

        # Assert the results are applied to the mixture in the same order
        state.curr_mixture.__setitem__.assert_has_calls([call(1, component1), call(0, component0)])

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

    def test_clear_after_prune_with_unoptimized_components_no_key_error(self, mock_optimizer: Optimizer) -> None:
        """Reindex sparse blocks even when intermediate components have no blocks."""
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(4, {"sigma"}, MaximizationStrategy.QFUNCTION),
        ]
        step = MaximizationStep(blocks, mock_optimizer)

        step.clear_after_prune([1])

        assert [block.component_id for block in step.blocks] == [0, 3]

    def test_clear_after_prune_with_multiple_blocks_per_component(self, mock_optimizer: Optimizer) -> None:
        """Preserve block order and parameter selections when a component has multiple blocks."""
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(2, {"rate"}, MaximizationStrategy.QFUNCTION),
        ]
        step = MaximizationStep(blocks, mock_optimizer)

        step.clear_after_prune([1])

        assert [block.component_id for block in step.blocks] == [0, 0, 1]
        assert [block.params_to_optimize for block in step.blocks] == [{"mu"}, {"sigma"}, {"rate"}]

    def test_clear_after_prune_handles_duplicate_indices_and_multiple_removals(self, mock_optimizer: Optimizer) -> None:
        """Count each removed component once when shifting sparse component indices."""
        blocks = [OptimizationBlock(index, None, MaximizationStrategy.QFUNCTION) for index in [0, 2, 5]]
        step = MaximizationStep(blocks, mock_optimizer)

        step.clear_after_prune([3, 1, 3])

        assert [block.component_id for block in step.blocks] == [0, 1, 3]

    def test_clear_after_prune_does_not_modify_the_blocks_passed_by_the_caller(self, mock_optimizer: Optimizer) -> None:
        """Replace surviving blocks without changing caller-owned configuration objects."""
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, None, MaximizationStrategy.MOMENTS),
            OptimizationBlock(2, {"sigma"}, MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD),
        ]
        step = MaximizationStep(blocks, mock_optimizer)

        step.clear_after_prune([1])

        assert [block.component_id for block in blocks] == [0, 1, 2]
        assert [block.component_id for block in step.blocks] == [0, 1]
        for surviving, original in zip(step.blocks, [blocks[0], blocks[2]], strict=True):
            assert surviving is not original
            assert surviving.params_to_optimize is original.params_to_optimize
            assert surviving.maximization_strategy is original.maximization_strategy

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

    def test_run_propagates_not_implemented_error_from_strategy(
        self, mock_optimizer: Optimizer, real_state: PipelineState
    ) -> None:
        """Propagate the real Moments strategy error for an unsupported parametrization."""
        real_state.curr_mixture[0] = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0)
        step = MaximizationStep([OptimizationBlock(0, None, MaximizationStrategy.MOMENTS)], mock_optimizer)

        with pytest.raises(NotImplementedError, match="Moments strategy is not implemented for the 'meanVar'"):
            step.run(real_state)

        assert real_state.error is None

    def test_run_propagates_value_error_for_invalid_parameters(
        self, mock_optimizer: Optimizer, real_state: PipelineState
    ) -> None:
        """Propagate parameter validation errors from a real analytical strategy."""
        block = OptimizationBlock(0, {"invalid_param"}, MaximizationStrategy.QFUNCTION)
        step = MaximizationStep([block], mock_optimizer)

        with pytest.raises(ValueError, match="invalid_param.*are not parameters"):
            step.run(real_state)

        assert real_state.error is None

    def test_run_propagates_index_error_for_out_of_range_component_id(
        self, mock_optimizer: Optimizer, real_state: PipelineState
    ) -> None:
        """Propagate indexing errors when a block refers past the last real component."""
        block = OptimizationBlock(real_state.curr_mixture.n_components, None, MaximizationStrategy.QFUNCTION)
        step = MaximizationStep([block], mock_optimizer)

        with pytest.raises(IndexError, match="index out of range"):
            step.run(real_state)

        assert real_state.error is None

    def test_run_does_not_update_anything_if_a_strategy_raises(
        self, mock_optimizer: Optimizer, real_state: PipelineState, mocker: MockerFixture
    ) -> None:
        """Leave all components and weights untouched when a later strategy raises."""
        state = real_state
        mixture = state.curr_mixture
        original_components = list(mixture)
        original_log_weights = mixture.log_weights.copy()
        blocks = [OptimizationBlock(index, {"mu"}, MaximizationStrategy.QFUNCTION) for index in range(2)]
        step = MaximizationStep(blocks, mock_optimizer)
        error = RuntimeError("Optimization diverged")
        strategy = mocker.Mock(side_effect=[(0, {"mu": 10.0}), error])
        mocker.patch.object(MaximizationStep, "_strategies", {MaximizationStrategy.QFUNCTION: strategy})
        update = mocker.spy(step, "_update_components_params")

        with pytest.raises(RuntimeError, match="Optimization diverged") as caught:
            step.run(state)

        assert caught.value is error
        assert strategy.call_count == len(blocks)
        update.assert_not_called()
        assert state.curr_mixture is mixture
        assert all(current is original for current, original in zip(mixture, original_components, strict=True))
        np.testing.assert_array_equal(mixture.log_weights, original_log_weights)
        assert state.error is None


class TestDetailsFoundByMutationTesting:
    def test_run_keeps_the_log_weight_of_a_component_without_responsibility_finite(
        self, mocker: MockerFixture, mock_optimizer: Optimizer, parametrized_state: PipelineState
    ):
        """A component that has no responsibility gets a tiny weight, not a weight whose logarithm is undefined."""
        state = parametrized_state
        state.H = np.array([[1.0, 0.0]] * 4)
        log_weights = mocker.PropertyMock()
        type(state.curr_mixture).log_weights = log_weights

        MaximizationStep(blocks=[], optimizer=mock_optimizer).run(state)

        actual = log_weights.call_args.args[0]
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, [0.0, np.log(1e-30)], atol=1e-9)

    def test_clear_after_prune_renumbers_all_blocks_after_the_removed_one(self, mock_optimizer: Optimizer):
        """Removing the first of four components shifts every other block by one: 1 -> 0, 2 -> 1, 3 -> 2."""
        blocks = [OptimizationBlock(i, None, MaximizationStrategy.QFUNCTION) for i in range(4)]
        step = MaximizationStep(blocks=blocks, optimizer=mock_optimizer)

        step.clear_after_prune([0])

        assert [block.component_id for block in step.blocks] == [0, 1, 2]
