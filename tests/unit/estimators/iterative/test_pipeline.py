"""Tests for Pipeline class"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable
from copy import copy

import numpy as np
import pytest
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators.iterative import (
    Breakpointer,
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    Pipeline,
    PipelineState,
    PipelineStep,
    PriorThresholdPruner,
    Pruner,
    StepBreakpointer,
)
from pysatl_mpest.estimators.iterative._iteration_history import IterationRecord, IterationsHistory
from pysatl_mpest.optimizers import ScipyNelderMead
from tests.helpers.families import family

# --- Mock objects for isolated testing ---


class MockStepA(PipelineStep):
    """A concrete mock step 'A' for testing sequences."""

    def __init__(self, call_log: list[str] | None = None):
        self.call_log = call_log if call_log is not None else []

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [MockStepB]

    def run(self, state: PipelineState) -> PipelineState:
        self.call_log.append("A")
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass


class MockStepB(PipelineStep):
    """A concrete mock step 'B' that can follow 'A' and loop back."""

    def __init__(self, call_log: list[str] | None = None):
        self.call_log = call_log if call_log is not None else []

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [MockStepA]

    def run(self, state: PipelineState) -> PipelineState:
        self.call_log.append("B")
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass


class MockSingleStep(PipelineStep):
    """A concrete mock step that can only be followed by itself."""

    def __init__(self, call_log: list[str] | None = None):
        self.call_log = call_log if call_log is not None else []

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [MockSingleStep]

    def run(self, state: PipelineState) -> PipelineState:
        if self.call_log is not None:
            self.call_log.append("SingleStep")
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass


class MockStepWithError(MockSingleStep):
    """A mock step that simulates an error occurring."""

    def run(self, state: PipelineState) -> PipelineState:
        super().run(state)
        state.error = ValueError(f"Simulated error in {self.__class__.__name__}")
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass


class ModifyingStep(PipelineStep):
    """A concrete mock step that actively modifies the mixture's state."""

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [ModifyingStep]  # Может идти сам за собой

    def run(self, state: PipelineState) -> PipelineState:
        if state.curr_mixture.n_components > 0:
            new_weights = np.zeros(state.curr_mixture.n_components)
            new_weights[0] = 1.0
            state.curr_mixture.log_weights = np.log(new_weights + 1e-30)

        if state.curr_mixture.n_components > 0:
            comp = state.curr_mixture.components[0]
            new_comp = cc.with_parameters(comp, lambda_=999.0)
            state.curr_mixture[0] = new_comp

        if state.curr_mixture.n_components > 1:
            state.curr_mixture.remove_component(1)

        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass


class ParameterIncrementStep(PipelineStep):
    """A mock step that predictably modifies mixture parameters on each run."""

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [ParameterIncrementStep]

    def run(self, state: PipelineState) -> PipelineState:
        # Increment 'lambda_' for each component by 1
        for i, component in enumerate(state.curr_mixture.components):
            new_lambda = cc.all_parameters(component)["lambda_"] + 1.0
            state.curr_mixture[i] = cc.with_parameters(component, lambda_=new_lambda)

        # Predictably shift weights (e.g., increase first, decrease second)
        new_weights = state.curr_mixture.weights.copy()
        if len(new_weights) > 1:
            new_weights[0] += 0.1
            new_weights[1] -= 0.1

            new_weights = np.clip(new_weights, 0, 1)
            new_weights /= np.sum(new_weights)

        state.curr_mixture.log_weights = np.log(new_weights + 1e-30)
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass


class MockBreakpointer(Breakpointer):
    """A mock object for Breakpointer that stops at a specified iteration."""

    def __init__(self, stop_at_iteration: int, call_log: list[str] | None = None):
        self.stop_at_iteration = stop_at_iteration
        self.current_iteration = 1
        self.call_log = call_log

    def check(self, state: PipelineState) -> bool:
        if self.call_log is not None:
            self.call_log.append("check_breakpointer")
        self.current_iteration += 1
        return self.current_iteration == self.stop_at_iteration


class MockPruner(Pruner):
    """A mock object for Pruner that removes a component and logs its calls."""

    def __init__(self, name: str = "pruner", call_log: list[str] | None = None):
        self.name = name
        self.call_log = call_log if call_log is not None else []

    def prune(self, state: PipelineState) -> tuple[PipelineState, list[int] | None]:
        self.call_log.append(self.name)
        removed_components_indices = []
        if state.curr_mixture.n_components > 1:
            state.curr_mixture.remove_component(0)
            removed_components_indices.append(0)
        return (state, removed_components_indices)


# --- Fixtures for tests ---


@pytest.fixture
def initial_mixture() -> MixtureModel:
    """Provides a basic MixtureModel with two components."""
    components = [family(FamilyName.EXPONENTIAL)(lambda_=1.0), family(FamilyName.EXPONENTIAL)(lambda_=2.0)]
    return MixtureModel(components, weights=[0.5, 0.5])


@pytest.fixture
def sample_data() -> np.ndarray:
    """Provides a simple data array."""
    return np.array([1, 2, 6, 7], dtype=np.float64)


# --- Initialization (`__init__`) Tests ---


class TestPipelineInitialization:
    """A group of tests for the Pipeline constructor."""

    @pytest.mark.parametrize("container", [list, tuple])
    def test_init_accepts_sequence_types(self, container: Callable):
        """Tests that the constructor accepts list and tuple for its arguments."""

        steps = container([MockSingleStep()])
        breakpointers = container([MockBreakpointer(stop_at_iteration=2)])
        pruners = container([MockPruner()])

        pipeline = Pipeline(steps, breakpointers, pruners)

        assert isinstance(pipeline.steps, list)
        assert isinstance(pipeline.breakpointers, list)
        assert isinstance(pipeline.pruners, list)
        assert isinstance(pipeline.history, IterationsHistory)

    def test_init_with_none_pruners_creates_empty_list(self):
        """Tests that an empty list is created if pruners=None."""

        steps = [MockSingleStep()]
        breakpointers = [MockBreakpointer(stop_at_iteration=2)]

        pipeline = Pipeline(steps, breakpointers, pruners=None)

        assert pipeline.pruners == []

    def test_init_raises_error_for_empty_breakpointers(self):
        """Tests that an empty list of breakpointers raises a ValueError."""

        steps = [MockSingleStep()]

        with pytest.raises(
            ValueError,
            match=(
                r"^The 'breakpointers' list cannot be empty\. "
                r"At least one stopping criterion must be provided to prevent an infinite loop\.$"
            ),
        ):
            Pipeline(steps, [])

    def test_init_creates_history_with_correct_frequency(self):
        """Tests that the history is created with the correct frequency parameter."""

        steps = [MockSingleStep()]
        breakpointers = [MockBreakpointer(stop_at_iteration=2)]

        pipeline_default = Pipeline(steps, breakpointers)
        assert pipeline_default.history.once_in_iterations == 1

        pipeline_custom = Pipeline(steps, breakpointers, once_in_iterations=5)
        FREQUENCY_OF_LOGS = 5
        assert pipeline_custom.history.once_in_iterations == FREQUENCY_OF_LOGS


# --- Step Validation Tests ---


class TestPipelineValidation:
    """A group of tests for the step validation logic."""

    def test_validate_steps_raises_error_for_empty_steps(self):
        """Tests that an empty list of steps raises a ValueError."""

        breakpointers = [MockBreakpointer(stop_at_iteration=2)]
        with pytest.raises(ValueError, match=r"^The 'steps' list cannot be empty for a Pipeline\.$"):
            Pipeline([], breakpointers)

    def test_validate_steps_raises_error_for_invalid_order(self):
        """Tests that an incorrect sequence of steps raises a ValueError."""

        # Invalid sequence: MockStepA cannot be followed by another MockStepA
        invalid_steps = [MockStepA(), MockStepA()]
        breakpointers = [MockBreakpointer(stop_at_iteration=2)]

        with pytest.raises(ValueError, match="Wrong pipeline configuration"):
            Pipeline(invalid_steps, breakpointers)

    def test_validate_steps_succeeds_for_valid_cyclic_order(self):
        """Tests that a correct cyclic sequence passes validation."""

        # Valid sequence: MockStepA -> MockStepB -> MockStepA (loops back)
        valid_steps = [MockStepA(), MockStepB()]
        breakpointers = [MockBreakpointer(stop_at_iteration=2)]

        try:
            Pipeline(valid_steps, breakpointers)
        except ValueError:
            pytest.fail("Valid cyclic step configuration raised ValueError unexpectedly.")


# --- Fit Method Tests ---

NUMBER_OF_BLOCKS = 2


class TestPipelineFit:
    """A group of tests for the fit method."""

    def test_fit_modifies_mixture_to_expected_state_after_iterations(self, initial_mixture, sample_data):
        """
        Tests that the pipeline correctly modifies the mixture model's parameters
        over a fixed number of iterations to a predictable final state.
        """

        expected_n_components = 2

        steps = [ParameterIncrementStep()]
        breakpointers = [MockBreakpointer(stop_at_iteration=3)]
        pipeline = Pipeline(steps, breakpointers)

        # Initial state from fixture:
        # Component 0: lambda_=1.0, weight=0.5
        # Component 1: lambda_=2.0, weight=0.5

        # Expected state after 2 iterations:
        # Iter 1: lambdas=[2, 3], weights=[0.6, 0.4]
        # Iter 2: lambdas=[3, 4], weights=[0.7, 0.3]

        expected_final_lambdas = np.array([3.0, 4.0], dtype=np.float64)
        expected_final_weights = np.array([0.7, 0.3], dtype=np.float64)

        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        assert fitted_mixture.n_components == expected_n_components

        # Check weights
        rtol = 1e-7

        np.testing.assert_allclose(
            fitted_mixture.weights,
            expected_final_weights,
            rtol=rtol,
            err_msg="Mixture weights did not reach the expected values.",
        )

        final_lambdas = np.array(
            [cc.all_parameters(comp)["lambda_"] for comp in fitted_mixture.components], dtype=np.float64
        )

        np.testing.assert_allclose(final_lambdas, expected_final_lambdas, rtol=rtol)

    def test_fit_does_not_modify_original_mixture(self, initial_mixture, sample_data):
        """
        Tests that the fit method does not modify the original mixture object,
        even when the internal steps actively try to modify the state.
        """

        original_mixture_copy = copy(initial_mixture)

        steps = [ModifyingStep()]
        breakpointers = [MockBreakpointer(stop_at_iteration=2)]
        pipeline = Pipeline(steps, breakpointers)

        pipeline.fit(sample_data, initial_mixture)

        # Comparing weights
        np.testing.assert_array_equal(
            initial_mixture.weights, original_mixture_copy.weights, err_msg="Original mixture's weights were modified."
        )

        # Comparing number of components
        assert initial_mixture.n_components == original_mixture_copy.n_components

        # Comparing parameters of the components
        for i in range(initial_mixture.n_components):
            original_params = cc.all_parameters(original_mixture_copy.components[i])["lambda_"]
            current_params = cc.all_parameters(initial_mixture.components[i])["lambda_"]
            np.testing.assert_array_equal(
                current_params,
                original_params,
                err_msg=f"Parameters of component {i} in the original mixture were modified.",
            )

    def test_fit_runs_steps_in_correct_order(self, initial_mixture, sample_data):
        """Tests that the steps are executed in the correct order."""

        call_log = []

        # Configuration: A -> B -> A (cycle)
        steps = [MockStepA(call_log), MockStepB(call_log)]
        breakpointers = [MockBreakpointer(stop_at_iteration=3)]

        pipeline = Pipeline(steps, breakpointers)
        pipeline.fit(sample_data, initial_mixture)

        assert call_log == ["A", "B", "A", "B"]

    def test_fit_stops_with_multiple_breakpointers(self, initial_mixture, sample_data):
        """Tests that the loop stops if at least one breakpointer triggers."""

        expected_checks = 4

        call_log = []
        breakpointers = [
            MockBreakpointer(stop_at_iteration=100, call_log=call_log),  # Should not trigger
            MockBreakpointer(stop_at_iteration=3, call_log=call_log),  # Will trigger
        ]

        steps = [MockSingleStep()]

        pipeline = Pipeline(steps, breakpointers)
        pipeline.fit(sample_data, initial_mixture)

        # Iteration 1: current_iteration becomes 2 -> False.
        # Iteration 2: current_iteration becomes 3 -> True.
        # Therefore, the loop body will execute twice.
        # Since there are two breakpointers, check() is called on both in each iteration,
        # so it's called 2 * 2 = 4 times in total.
        assert call_log.count("check_breakpointer") == expected_checks

    def test_fit_warns_and_returns_on_step_error(self, initial_mixture, sample_data):
        """Tests that a RuntimeWarning is issued and the mixture is returned on a step error."""

        call_log = []
        steps = [MockStepWithError(call_log)]
        breakpointers = [MockBreakpointer(stop_at_iteration=10)]

        pipeline = Pipeline(steps, breakpointers)

        with pytest.warns(RuntimeWarning) as record:
            result_mixture = pipeline.fit(sample_data, initial_mixture)

        assert len(record) == 1
        assert "stopped prematurely due to an error" in str(record[0].message)
        assert isinstance(result_mixture, MixtureModel)
        assert call_log == ["SingleStep"]

    def test_fit_with_pruner_modifies_mixture(self, initial_mixture, sample_data):
        """Tests that the pruner is called and can modify the mixture model."""

        expected_n_components = 2
        assert initial_mixture.n_components == expected_n_components

        call_log = []
        steps = [MockSingleStep()]
        pruners = [MockPruner("pruner", call_log)]
        breakpointers = [MockBreakpointer(stop_at_iteration=2)]

        pipeline = Pipeline(steps, breakpointers, pruners)
        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        assert call_log.count("pruner") == 1
        assert fitted_mixture.n_components == 1

    def test_fit_with_multiple_pruners_executes_in_order(self, initial_mixture, sample_data):
        """Test: multiple pruners are executed in the correct order."""

        call_log = []
        steps = [MockSingleStep()]
        pruners = [
            MockPruner(name="pruner1", call_log=call_log),
            MockPruner(name="pruner2", call_log=call_log),
        ]
        breakpointers = [MockBreakpointer(stop_at_iteration=2)]

        pipeline = Pipeline(steps, breakpointers, pruners)

        # The mixture has 2 components. pruner1 will remove one, but pruner2 will not be able to
        # (because only the last one remains).
        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        assert call_log == ["pruner1", "pruner2"]
        assert fitted_mixture.n_components == 1

    def test_fit_executes_pruners_after_all_steps(self, initial_mixture, sample_data):
        """Tests the order of execution: all steps, then all pruners."""
        call_log = []
        steps = [MockStepA(call_log), MockStepB(call_log)]
        pruners = [MockPruner("pruner", call_log)]
        breakpointers = [MockBreakpointer(stop_at_iteration=2, call_log=call_log)]

        pipeline = Pipeline(steps, breakpointers, pruners)
        pipeline.fit(sample_data, initial_mixture)

        expected_order = ["A", "B", "pruner", "check_breakpointer"]

        assert call_log == expected_order

    def test_fit_record_iterations_to_public_history(self, initial_mixture, sample_data):
        """Tests that the fit method record iterations to the public history attribute."""

        steps = [MockSingleStep()]
        breakpointers = [MockBreakpointer(stop_at_iteration=3)]

        pipeline = Pipeline(steps, breakpointers)

        assert len(pipeline.history) == 0

        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        assert len(pipeline.history) > 0
        assert isinstance(pipeline.history[0], IterationRecord)
        assert pipeline.history[0].iteration == 0
        assert pipeline.history[0].mixture == fitted_mixture

    def test_fit_records_with_custom_frequency(self, initial_mixture, sample_data):
        """Tests that recording occurs at the specified frequency."""

        steps = [MockSingleStep()]

        breakpointers = [MockBreakpointer(stop_at_iteration=5)]

        pipeline = Pipeline(steps, breakpointers, once_in_iterations=2)

        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        EXPECTED_LEN_OF_HISTORY = 2
        EXPECTED_ITERATIONS = [0, 2]

        assert len(pipeline.history) == EXPECTED_LEN_OF_HISTORY
        assert pipeline.history[0].iteration == EXPECTED_ITERATIONS[0]
        assert pipeline.history[1].iteration == EXPECTED_ITERATIONS[1]
        assert pipeline.history[1].mixture == fitted_mixture

    def test_clear_after_prune_called_during_fit(self, initial_mixture, sample_data):
        """Tests that clear_after_prune is called for all steps during pipeline execution."""

        call_log = []
        LEN_CALL_LOG = 2

        class TrackingStep(MockSingleStep):
            def clear_after_prune(self, removed_components_indices: list[int]) -> None:
                call_log.append(removed_components_indices)

        # Create steps that track clear_after_prune calls
        steps = [TrackingStep(), TrackingStep()]
        pruners = [MockPruner("pruner")]
        breakpointers = [MockBreakpointer(stop_at_iteration=2)]

        pipeline = Pipeline(steps, breakpointers, pruners)
        pipeline.fit(sample_data, initial_mixture)

        # Verify clear_after_prune was called for each step
        assert len(call_log) == LEN_CALL_LOG  # Called for each step
        for removed_indices in call_log:
            assert removed_indices == [0]


# --- Tests for MaximizationStep clear_after_prune method ---


class TestMaximizationStepClearAfterPrune:
    """Unit tests for the clear_after_prune method of MaximizationStep."""

    def test_clear_after_prune_removes_blocks_for_pruned_components(self):
        """Tests that clear_after_prune removes optimization blocks for pruned components."""

        blocks = [
            OptimizationBlock(0, {"param1"}, "q_function"),
            OptimizationBlock(1, {"param2"}, "q_function"),
            OptimizationBlock(2, {"param1", "param2"}, "q_function"),
        ]

        step = MaximizationStep(blocks=blocks, optimizer=None)

        # Remove component 1
        removed_indices = [1]
        step.clear_after_prune(removed_indices)

        # Should have 2 blocks left
        assert len(step.blocks) == NUMBER_OF_BLOCKS
        # Blocks should be for original components 0 and 2
        assert step.blocks[0].component_id == 0
        assert step.blocks[1].component_id == 1  # Reindexed from 2 to 1

    def test_clear_after_prune_preserves_optimization_parameters(self):
        """Tests that clear_after_prune preserves optimization parameters for remaining blocks."""

        blocks = [
            OptimizationBlock(0, {"param1"}, "q_function"),
            OptimizationBlock(1, {"param2", "param3"}, "q_function"),
            OptimizationBlock(2, {"param1", "param2"}, "q_function"),
        ]

        step = MaximizationStep(blocks=blocks, optimizer=None)

        # Remove component 1
        removed_indices = [1]
        step.clear_after_prune(removed_indices)

        # Check that optimization parameters are preserved
        assert step.blocks[0].params_to_optimize == {"param1"}
        assert step.blocks[1].params_to_optimize == {"param1", "param2"}

    def test_clear_after_prune_with_empty_removal(self):
        """Tests that clear_after_prune does nothing when no components are removed."""

        blocks = [OptimizationBlock(0, {"param1"}, "q_function"), OptimizationBlock(1, {"param2"}, "q_function")]

        step = MaximizationStep(blocks=blocks, optimizer=None)
        original_blocks = list(step.blocks)  # Create a copy

        step.clear_after_prune([])

        # Blocks should remain unchanged
        assert step.blocks == original_blocks

    def test_clear_after_prune_with_none_blocks(self):
        """Tests that clear_after_prune handles None blocks gracefully."""

        step = MaximizationStep(blocks=[], optimizer=None)

        # Should not raise an error
        step.clear_after_prune([0, 1])
        assert step.blocks == []

    def test_clear_after_prune_with_multiple_removals(self):
        """Tests clear_after_prune with multiple components removed."""

        blocks = [
            OptimizationBlock(0, {"param1"}, "q_function"),
            OptimizationBlock(1, {"param2"}, "q_function"),
            OptimizationBlock(2, {"param3"}, "q_function"),
            OptimizationBlock(3, {"param1", "param2"}, "q_function"),
        ]

        step = MaximizationStep(blocks=blocks, optimizer=None)

        # Remove components 1 and 3
        removed_indices = [1, 3]
        step.clear_after_prune(removed_indices)

        assert len(step.blocks) == NUMBER_OF_BLOCKS
        assert step.blocks[0].component_id == 0  # Originally component 0
        assert step.blocks[1].component_id == 1  # Originally component 2, now reindexed to 1

        assert step.blocks[0].params_to_optimize == {"param1"}
        assert step.blocks[1].params_to_optimize == {"param3"}


# --- Bookkeeping of pruned components ---


class RecordingStep(PipelineStep):
    """A step that logs when it runs and which removed components it is told about."""

    def __init__(self, name: str, log: list[tuple[str, ...]]):
        self.name = name
        self.log = log

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [RecordingStep]

    def run(self, state: PipelineState) -> PipelineState:
        self.log.append(("run", self.name))
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        self.log.append(("clear", self.name, str(removed_components_indices)))


class ResponsibilityStep(RecordingStep):
    """A step that sets H to uniform responsibilities for the current components."""

    def run(self, state: PipelineState) -> PipelineState:
        n_components = state.curr_mixture.n_components
        state.H = np.full((len(state.X), n_components), 1.0 / n_components)
        return super().run(state)


class ScriptedPruner(Pruner):
    """A pruner that removes the components given for each call, in order, and logs its calls."""

    def __init__(self, name: str, log: list[tuple[str, ...]], removals: list[list[int]]):
        self.name = name
        self.log = log
        self.removals = iter(removals)

    def prune(self, state: PipelineState) -> tuple[PipelineState, list[int]]:
        removed = next(self.removals, [])
        self.log.append(("prune", self.name, str(removed)))
        for index in sorted(removed, reverse=True):
            state.curr_mixture.remove_component(index)
        return state, removed


def exponential_mixture(n_components: int) -> MixtureModel:
    components = [family(FamilyName.EXPONENTIAL)(lambda_=float(i + 1)) for i in range(n_components)]
    return MixtureModel(components)


class TestPipelinePruningBookkeeping:
    """Tests that the steps learn about pruned components right away and exactly once."""

    def test_real_maximization_step_prunes_components_without_optimization_blocks(self) -> None:
        """Continue fitting sparse blocks after real pruning without mutating caller blocks."""
        mixture = MixtureModel(
            [family(FamilyName.EXPONENTIAL)(lambda_=float(index + 1)) for index in range(4)],
            weights=[0.3, 1e-8, 0.3, 0.4 - 1e-8],
        )
        blocks = [OptimizationBlock(index, {"lambda_"}, MaximizationStrategy.QFUNCTION) for index in [0, 3]]
        maximization = MaximizationStep(blocks, ScipyNelderMead())
        iterations = 3
        remaining_components = 3
        pipeline = Pipeline(
            [ExpectationStep(), maximization],
            [StepBreakpointer(iterations)],
            [PriorThresholdPruner(1e-4)],
        )

        fitted = pipeline.fit(np.array([0.5, 1.0, 1.5, 2.0]), mixture)

        assert fitted.n_components == remaining_components
        assert len(pipeline.history) == iterations
        assert all(record.error is None for record in pipeline.history)
        assert all(record.mixture.n_components == remaining_components for record in pipeline.history)
        assert all(record.H is not None and record.H.shape == (4, remaining_components) for record in pipeline.history)
        assert [block.component_id for block in maximization.blocks] == [0, 2]
        assert [block.component_id for block in blocks] == [0, 3]
        assert maximization.blocks[1] is not blocks[1]
        assert cc.all_parameters(fitted[1]) == cc.all_parameters(mixture[2])

    def test_steps_are_notified_right_after_pruning(self, sample_data):
        log: list[tuple[str, ...]] = []
        steps = [RecordingStep("A", log), RecordingStep("B", log)]
        pipeline = Pipeline(steps, [StepBreakpointer(2)], [ScriptedPruner("p", log, [[0], []])])

        pipeline.fit(sample_data, exponential_mixture(3))

        assert log == [
            ("run", "A"),
            ("run", "B"),
            ("prune", "p", "[0]"),
            ("clear", "A", "[0]"),
            ("clear", "B", "[0]"),
            ("run", "A"),
            ("run", "B"),
            ("prune", "p", "[]"),
        ]

    def test_each_removal_is_reported_once(self, sample_data):
        log: list[tuple[str, ...]] = []
        pruner = ScriptedPruner("p", log, [[0], [], [1], []])
        pipeline = Pipeline([RecordingStep("A", log)], [StepBreakpointer(4)], [pruner])

        pipeline.fit(sample_data, exponential_mixture(4))

        assert [entry[2] for entry in log if entry[0] == "clear"] == ["[0]", "[1]"]

    def test_steps_are_not_notified_when_nothing_is_removed(self, sample_data):
        log: list[tuple[str, ...]] = []
        pipeline = Pipeline([RecordingStep("A", log)], [StepBreakpointer(3)], [ScriptedPruner("p", log, [])])

        pipeline.fit(sample_data, exponential_mixture(2))

        assert not [entry for entry in log if entry[0] == "clear"]

    def test_each_pruner_reports_indices_relative_to_the_mixture_it_was_given(self, sample_data):
        """The second pruner sees the mixture without the component the first one removed."""
        log: list[tuple[str, ...]] = []
        pruners = [ScriptedPruner("p1", log, [[0]]), ScriptedPruner("p2", log, [[1]])]
        pipeline = Pipeline([RecordingStep("A", log)], [StepBreakpointer(1)], pruners)

        fitted = pipeline.fit(sample_data, exponential_mixture(3))

        assert log == [
            ("run", "A"),
            ("prune", "p1", "[0]"),
            ("clear", "A", "[0]"),
            ("prune", "p2", "[1]"),
            ("clear", "A", "[1]"),
        ]
        # The components had the rates 1, 2 and 3; the first pruner removed the rate 1, the second one the rate 3.
        remaining_rate = 2.0
        assert fitted.n_components == 1
        assert cc.all_parameters(fitted[0])["lambda_"] == remaining_rate

    def test_columns_of_h_of_pruned_components_are_dropped(self, sample_data):
        log: list[tuple[str, ...]] = []
        pipeline = Pipeline(
            [ResponsibilityStep("A", log)],
            [StepBreakpointer(2)],
            [ScriptedPruner("p", log, [[0, 2], []])],
        )

        pipeline.fit(sample_data, exponential_mixture(4))

        first, second = pipeline.history[0], pipeline.history[1]
        assert first.mixture.n_components == 2  # noqa: PLR2004
        assert first.H.shape == (len(sample_data), 2)
        assert second.H.shape == (len(sample_data), 2)


# --- Details found by mutation testing ---


class TestPipelineStepValidation:
    """Tests that the order of the steps is checked for every pair of neighbours."""

    def test_the_last_pair_of_steps_is_checked(self):
        """The invalid pair is B followed by B at the end of the list."""
        with pytest.raises(ValueError, match="Wrong pipeline configuration"):
            Pipeline([MockStepA(), MockStepB(), MockStepB()], [StepBreakpointer(1)])

    def test_the_first_pair_of_steps_is_checked(self):
        """The invalid pair is A followed by A at the start of the list, then comes B."""
        with pytest.raises(ValueError, match="Wrong pipeline configuration"):
            Pipeline([MockStepA(), MockStepA(), MockStepB()], [StepBreakpointer(1)])

    def test_the_pair_closing_the_cycle_is_checked(self):
        """The neighbours in the list are fine, but the last step A cannot be followed by the first step A."""
        with pytest.raises(ValueError, match="Wrong pipeline configuration"):
            Pipeline([MockStepA(), MockStepB(), MockStepA()], [StepBreakpointer(1)])


class DataProbeStep(MockSingleStep):
    """A step that records the data it receives."""

    def __init__(self):
        super().__init__([])
        self.dtypes: list[np.dtype] = []

    def run(self, state: PipelineState) -> PipelineState:
        self.dtypes.append(state.X.dtype)
        return state


class PreviousMixtureProbeStep(MockSingleStep):
    """A step that records the previous mixture next to a copy of the current one, then changes the current one."""

    def __init__(self):
        super().__init__([])
        self.seen: list[tuple[MixtureModel | None, MixtureModel]] = []

    def run(self, state: PipelineState) -> PipelineState:
        self.seen.append((state.prev_mixture, copy(state.curr_mixture)))
        component = state.curr_mixture[0]
        state.curr_mixture[0] = cc.with_parameters(component, lambda_=cc.all_parameters(component)["lambda_"] + 1.0)
        return state


class FailingStep(MockSingleStep):
    """A step that sets H and fails on the given run."""

    def __init__(self, failing_run: int):
        super().__init__([])
        self.failing_run = failing_run
        self.runs = 0
        self.error = ValueError("simulated failure")

    def run(self, state: PipelineState) -> PipelineState:
        self.runs += 1
        state.H = np.full((len(state.X), state.curr_mixture.n_components), 0.5)
        if self.runs == self.failing_run:
            state.error = self.error
        return state


class ErrorSettingPruner(Pruner):
    """A pruner that removes nothing but leaves an error in the state."""

    def __init__(self):
        self.error = ValueError("left by the pruner")

    def prune(self, state: PipelineState) -> tuple[PipelineState, list[int]]:
        state.error = self.error
        return state, []


class TestPipelineFitDetails:
    def test_fit_gives_float64_data_to_the_steps(self, initial_mixture):
        step = DataProbeStep()

        Pipeline([step], [StepBreakpointer(2)]).fit([1, 2, 6, 7], initial_mixture)

        assert step.dtypes == [np.float64, np.float64]

    def test_fit_gives_each_iteration_the_mixture_it_started_with_as_the_previous_one(self, initial_mixture):
        step = PreviousMixtureProbeStep()

        Pipeline([step], [StepBreakpointer(3)]).fit(np.array([1.0, 2.0]), initial_mixture)

        for previous, current_at_start in step.seen:
            assert previous is not None
            assert previous == current_at_start
        assert step.seen[0][0] == initial_mixture
        assert step.seen[1][0] != step.seen[0][0]

    def test_a_step_error_in_the_first_iteration_is_recorded_in_the_history(self, initial_mixture, sample_data):
        step = FailingStep(failing_run=1)
        pipeline = Pipeline([step], [StepBreakpointer(5)])

        with pytest.warns(RuntimeWarning):
            result = pipeline.fit(sample_data, initial_mixture)

        assert len(pipeline.history) == 1
        record = pipeline.history[0]
        assert record.iteration == 0
        assert record.mixture is result
        np.testing.assert_array_equal(record.X, sample_data)
        assert record.H is not None
        assert record.H.shape == (len(sample_data), 2)
        assert record.pruners_used is None
        assert record.error is step.error

    def test_a_step_error_in_a_later_iteration_is_added_to_the_last_record(self, initial_mixture, sample_data):
        step = FailingStep(failing_run=2)
        pipeline = Pipeline([step], [StepBreakpointer(5)])

        with pytest.warns(RuntimeWarning):
            pipeline.fit(sample_data, initial_mixture)

        assert len(pipeline.history) == 1
        assert pipeline.history[0].error is step.error

    def test_the_record_of_an_iteration_holds_the_data_the_pruners_and_the_error(self, initial_mixture, sample_data):
        pruner = ErrorSettingPruner()
        pipeline = Pipeline([MockSingleStep([])], [StepBreakpointer(1)], [pruner])

        pipeline.fit(sample_data, initial_mixture)

        record = pipeline.history[0]
        np.testing.assert_array_equal(record.X, sample_data)
        assert record.pruners_used == [pruner]
        assert record.error is pruner.error
