"""Mock implementations for PipelineStep."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

from pysatl_mpest.estimators.iterative.pipeline_state import PipelineState
from pysatl_mpest.estimators.iterative.pipeline_step import PipelineStep
from pysatl_mpest.typings import FloatingType


class MockCallbackPipelineStep[FloatT: FloatingType](PipelineStep[FloatT]):
    """A mock step that executes a custom callback.

    Parameters
    ----------
    available_steps : list[type[PipelineStep]]
        A list of step types that can legally follow this step.
    run_callback : Callable[[PipelineState[FloatT]], PipelineState[FloatT]], optional
        A function to mutate the state. If None, returns state unchanged.
    """

    def __init__(
        self,
        available_steps: list[type[PipelineStep[FloatT]]],
        run_callback: Callable[[PipelineState[FloatT]], PipelineState[FloatT]] | None = None,
    ) -> None:
        self._available_next_steps = available_steps
        self.run_callback = run_callback
        self.cleared_indices: list[int] = []

    @property
    def available_next_steps(self) -> list[type[PipelineStep[FloatT]]]:
        return self._available_next_steps

    def run(self, state: PipelineState[FloatT]) -> PipelineState[FloatT]:
        """Executes the callback or returns state unchanged.

        Parameters
        ----------
        state : PipelineState[FloatT]
            The current pipeline state.

        Returns
        -------
        PipelineState[FloatT]
            The updated state.
        """

        if self.run_callback is not None:
            return self.run_callback(state)
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        """Records the indices passed for pruning.

        Parameters
        ----------
        removed_components_indices : list[int]
            The indices of components that were removed from the mixture model.
        """

        self.cleared_indices.extend(removed_components_indices)


class MockErrorPipelineStep[FloatT: FloatingType](PipelineStep[FloatT]):
    """A mock step that raises a predefined exception at a specific iteration.

    Parameters
    ----------
    available_steps : list[type[PipelineStep]]
        A list of step types that can legally follow this step.
    error_to_raise : Exception
        The exception instance to raise.
    iteration_to_fail : int
        The exact iteration number at which to raise the exception.
    """

    def __init__(
        self,
        available_steps: list[type[PipelineStep[FloatT]]],
        error_to_raise: Exception,
        iteration_to_fail: int,
    ) -> None:
        self._available_next_steps = available_steps
        self.error_to_raise = error_to_raise
        self.iteration_to_fail = iteration_to_fail
        self._current_iteration = 0

    @property
    def available_next_steps(self) -> list[type[PipelineStep[FloatT]]]:
        return self._available_next_steps

    def run(self, state: PipelineState[FloatT]) -> PipelineState[FloatT]:
        """Raises the exception if the iteration matches, else passes state.

        Parameters
        ----------
        state : PipelineState[FloatT]
            The current pipeline state.

        Returns
        -------
        PipelineState[FloatT]
            The unchanged state.

        Raises
        ------
        Exception
            The predefined exception.
        """

        self._current_iteration += 1
        if self._current_iteration == self.iteration_to_fail:
            raise self.error_to_raise
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        """Does nothing.

        Parameters
        ----------
        removed_components_indices : list[int]
            The indices of components that were removed from the mixture model.
        """

        pass
