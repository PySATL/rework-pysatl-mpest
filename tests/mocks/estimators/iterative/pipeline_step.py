"""Mock implementations for PipelineStep."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

from pysatl_mpest.estimators.iterative.pipeline_state import PipelineState
from pysatl_mpest.estimators.iterative.pipeline_step import PipelineStep


class MockCallbackPipelineStep(PipelineStep):
    """Mock pipeline step executing a custom callback."""

    def __init__(
        self,
        available_steps: list[type[PipelineStep]],
        run_callback: Callable[[PipelineState], PipelineState] | None = None,
    ) -> None:
        self._available_next_steps = available_steps
        self.run_callback = run_callback
        self.cleared_indices: list[int] = []

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return self._available_next_steps

    def run(self, state: PipelineState) -> PipelineState:
        if self.run_callback is not None:
            return self.run_callback(state)
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        self.cleared_indices.extend(removed_components_indices)


class MockErrorPipelineStep(PipelineStep):
    """Mock pipeline step that raises a predefined exception at a specific iteration."""

    def __init__(
        self,
        available_steps: list[type[PipelineStep]],
        error_to_raise: Exception,
        iteration_to_fail: int,
    ) -> None:
        self._available_next_steps = available_steps
        self.error_to_raise = error_to_raise
        self.iteration_to_fail = iteration_to_fail
        self._current_iteration = 0

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return self._available_next_steps

    def run(self, state: PipelineState) -> PipelineState:
        self._current_iteration += 1
        if self._current_iteration == self.iteration_to_fail:
            raise self.error_to_raise
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass
