"""Tests for Pipeline class"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable
from copy import copy

import numpy as np
import pytest
from rework_pysatl_mpest.core import MixtureModel
from rework_pysatl_mpest.distributions import Exponential
from rework_pysatl_mpest.estimators.iterative import Breakpointer, Pipeline, PipelineState, PipelineStep, Pruner
from rework_pysatl_mpest.estimators.iterative._logger import IterationRecord, IterationsHistory
from rework_pysatl_mpest.estimators.iterative.steps import MaximizationStep, OptimizationBlock
from rework_pysatl_mpest.exceptions import NumericalStabilityError

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]

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


class MockStepThatOverlowsOnFloat16(MockSingleStep):
    """A mock step that simulates an error occurring only on float16"""

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [ParameterIncrementStep]

    def run(self, state: PipelineState) -> PipelineState:
        if state.curr_mixture.dtype == np.float16:
            state.error = NumericalStabilityError("Overflow!")
        return state


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
            state.curr_mixture.components[0].set_params_from_vector(["loc"], [999.0])

        if state.curr_mixture.n_components > 1:
            state.curr_mixture.remove_component(1)

        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass


class ParameterIncrementStep(PipelineStep):
    """A mock step that predictably modifies mixture parameters on each run."""

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [MockStepThatOverlowsOnFloat16, ParameterIncrementStep]

    def run(self, state: PipelineState) -> PipelineState:
        # Increment 'loc' for each component by 1
        for component in state.curr_mixture.components:
            new_loc = component.get_params_vector(["loc"])[0] + 1.0
            component.set_params_from_vector(["loc"], [new_loc])

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


@pytest.fixture(params=DTYPES_TO_TEST)
def initial_mixture(request) -> MixtureModel:
    """Provides a basic MixtureModel with two components with parametrized dtype."""
    dtype = request.param
    components = [Exponential(loc=0, rate=1, dtype=dtype), Exponential(loc=5, rate=2, dtype=dtype)]
    return MixtureModel(components, weights=[0.5, 0.5], dtype=dtype)


@pytest.fixture(params=DTYPES_TO_TEST)
def sample_data(request) -> np.ndarray:
    """Provides a simple data array with parametrized dtype."""
    dtype = request.param
    return np.array([1, 2, 6, 7], dtype=dtype)


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
        assert isinstance(pipeline.logger, IterationsHistory)

    def test_init_with_none_pruners_creates_empty_list(self):
        """Tests that an empty list is created if pruners=None."""

        steps = [MockSingleStep()]
        breakpointers = [MockBreakpointer(stop_at_iteration=2)]

        pipeline = Pipeline(steps, breakpointers, pruners=None)

        assert pipeline.pruners == []

    def test_init_raises_error_for_empty_breakpointers(self):
        """Tests that an empty list of breakpointers raises a ValueError."""

        steps = [MockSingleStep()]

        with pytest.raises(ValueError, match="The 'breakpointers' list cannot be empty"):
            Pipeline(steps, [])

    def test_init_creates_logger_with_correct_frequency(self):
        """Tests that the logger is created with the correct frequency parameter."""

        steps = [MockSingleStep()]
        breakpointers = [MockBreakpointer(stop_at_iteration=2)]

        pipeline_default = Pipeline(steps, breakpointers)
        assert pipeline_default.logger.once_in_iterations == 1

        pipeline_custom = Pipeline(steps, breakpointers, once_in_iterations=5)
        FREQUENCY_OF_LOGS = 5
        assert pipeline_custom.logger.once_in_iterations == FREQUENCY_OF_LOGS


# --- Step Validation Tests ---


class TestPipelineValidation:
    """A group of tests for the step validation logic."""

    def test_validate_steps_raises_error_for_empty_steps(self):
        """Tests that an empty list of steps raises a ValueError."""

        breakpointers = [MockBreakpointer(stop_at_iteration=2)]
        with pytest.raises(ValueError, match="The 'steps' list cannot be empty"):
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
        dtype = initial_mixture.dtype

        steps = [ParameterIncrementStep()]
        breakpointers = [MockBreakpointer(stop_at_iteration=3)]
        pipeline = Pipeline(steps, breakpointers)

        # Initial state from fixture:
        # Component 0: loc=0, rate=1, weight=0.5
        # Component 1: loc=5, rate=2, weight=0.5

        # Expected state after 2 iterations:
        # Iter 1: locs=[1, 6], weights=[0.6, 0.4]
        # Iter 2: locs=[2, 7], weights=[0.7, 0.3]

        expected_final_locs = np.array([2.0, 7.0], dtype=dtype)
        expected_final_rates = np.array([1.0, 2.0], dtype=dtype)  # Rates should not change
        expected_final_weights = np.array([0.7, 0.3], dtype=dtype)

        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        assert fitted_mixture.n_components == expected_n_components
        # Check type
        assert fitted_mixture.dtype == dtype
        assert fitted_mixture.weights.dtype == dtype
        for component in fitted_mixture.components:
            assert component.dtype == dtype

        # Check weights
        if dtype == np.float16:
            rtol = 1e-3
        elif dtype == np.float32:
            rtol = 1e-5
        else:  # float64
            rtol = 1e-7

        np.testing.assert_allclose(
            fitted_mixture.weights,
            expected_final_weights,
            rtol=rtol,
            err_msg="Mixture weights did not reach the expected values.",
        )

        final_locs = np.array([comp.loc for comp in fitted_mixture.components], dtype=dtype)
        final_rates = np.array([comp.rate for comp in fitted_mixture.components], dtype=dtype)

        np.testing.assert_allclose(final_locs, expected_final_locs, rtol=rtol)
        np.testing.assert_allclose(final_rates, expected_final_rates, rtol=rtol)

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
            original_params = original_mixture_copy.components[i].get_params_vector(["loc", "rate"])
            current_params = initial_mixture.components[i].get_params_vector(["loc", "rate"])
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

    def test_fit_logs_iterations_to_public_logger(self, initial_mixture, sample_data):
        """Tests that the fit method logs iterations to the public logger attribute."""

        steps = [MockSingleStep()]
        breakpointers = [MockBreakpointer(stop_at_iteration=3)]

        pipeline = Pipeline(steps, breakpointers)

        assert len(pipeline.logger) == 0

        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        assert len(pipeline.logger) > 0
        assert isinstance(pipeline.logger[0], IterationRecord)
        assert pipeline.logger[0].iteration == 0
        assert pipeline.logger[0].mixture == fitted_mixture

    def test_fit_logs_with_custom_frequency(self, initial_mixture, sample_data):
        """Tests that logging occurs at the specified frequency."""

        steps = [MockSingleStep()]

        breakpointers = [MockBreakpointer(stop_at_iteration=5)]

        pipeline = Pipeline(steps, breakpointers, once_in_iterations=2)

        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        EXPECTED_LEN_OF_LOGGER = 2
        EXPECTED_ITERATIONS = [0, 2]

        assert len(pipeline.logger) == EXPECTED_LEN_OF_LOGGER
        assert pipeline.logger[0].iteration == EXPECTED_ITERATIONS[0]
        assert pipeline.logger[1].iteration == EXPECTED_ITERATIONS[1]
        assert pipeline.logger[1].mixture == fitted_mixture

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

    def test_fit_handles_numerical_stability_error_and_upcasts_dtype(self, mocker):
        """Tests that Pipeline catches NumericalStabilityError and upgrades the dtype precision to float64."""

        initial_mixture_f16 = MixtureModel([Exponential(loc=0, rate=1, dtype=np.float16)], dtype=np.float16)
        sample_data_f16 = np.array([1, 2], dtype=np.float16)

        steps = [MockStepThatOverlowsOnFloat16(), ParameterIncrementStep()]
        breakpointers = [MockBreakpointer(stop_at_iteration=3)]

        pipeline = Pipeline(steps, breakpointers)

        fit_spy = mocker.spy(pipeline, "fit")

        with pytest.warns(UserWarning, match="Restarting pipeline with higher precision"):
            final_mixture = pipeline.fit(sample_data_f16, initial_mixture_f16)

        expected_count_calls = 2  # One original call, one recursive
        assert fit_spy.call_count == expected_count_calls

        # Check that the final model in float64
        assert final_mixture.dtype == np.float64
        assert final_mixture.weights.dtype == np.float64
        for component in final_mixture.components:
            assert component.dtype == np.float64

        # Step logic worked after increasing the precision
        assert final_mixture.components[0].loc == pytest.approx(2.0)


# --- Tests for MaximizationStep clear_after_prune method ---


class TestMaximizationStepClearAfterPrune:
    """Unit tests for the clear_after_prune method of MaximizationStep."""

    def test_clear_after_prune_removes_blocks_for_pruned_components(self):
        """Tests that clear_after_prune removes optimization blocks for pruned components."""

        blocks = [
            OptimizationBlock(0, {"loc"}, "q_function"),
            OptimizationBlock(1, {"rate"}, "q_function"),
            OptimizationBlock(2, {"loc", "rate"}, "q_function"),
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
            OptimizationBlock(0, {"loc"}, "q_function"),
            OptimizationBlock(1, {"rate", "scale"}, "q_function"),
            OptimizationBlock(2, {"loc", "rate"}, "q_function"),
        ]

        step = MaximizationStep(blocks=blocks, optimizer=None)

        # Remove component 1
        removed_indices = [1]
        step.clear_after_prune(removed_indices)

        # Check that optimization parameters are preserved
        assert step.blocks[0].params_to_optimize == {"loc"}
        assert step.blocks[1].params_to_optimize == {"loc", "rate"}

    def test_clear_after_prune_with_empty_removal(self):
        """Tests that clear_after_prune does nothing when no components are removed."""

        blocks = [OptimizationBlock(0, {"loc"}, "q_function"), OptimizationBlock(1, {"rate"}, "q_function")]

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
