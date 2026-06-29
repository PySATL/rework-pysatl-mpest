"""Mock implementations for Breakpointer."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

from pysatl_mpest.estimators.iterative.breakpointer import Breakpointer
from pysatl_mpest.estimators.iterative.pipeline_state import PipelineState
from pysatl_mpest.typings import FloatingType


class MockMaxIterationsBreakpointer[FloatT: FloatingType](Breakpointer[FloatT]):
    """A mock breakpointer that stops after a specific number of iterations.

    Parameters
    ----------
    max_iterations : int
        The number of iterations after which to stop.
    """

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        self._current_iteration = 0

    def check(self, state: PipelineState[FloatT]) -> bool:
        """Evaluates the state and returns True if max iterations are reached.

        Parameters
        ----------
        state : PipelineState[FloatT]
            The current pipeline state.

        Returns
        -------
        bool
            True if iterations >= max_iterations, False otherwise.
        """

        self._current_iteration += 1
        return self._current_iteration >= self.max_iterations


class MockCallbackBreakpointer[FloatT: FloatingType](Breakpointer[FloatT]):
    """A mock breakpointer that delegates the check logic to a callback.

    Parameters
    ----------
    check_callback : Callable[[PipelineState[FloatT]], bool]
        A function to evaluate the state and return a boolean.
    """

    def __init__(self, check_callback: Callable[[PipelineState[FloatT]], bool]) -> None:
        self.check_callback = check_callback

    def check(self, state: PipelineState[FloatT]) -> bool:
        """Evaluates the state using the provided callback.

        Parameters
        ----------
        state : PipelineState[FloatT]
            The current pipeline state.

        Returns
        -------
        bool
            The result of the callback function.
        """

        return self.check_callback(state)


class MockNeverBreakpointer[FloatT: FloatingType](Breakpointer[FloatT]):
    """A mock breakpointer that never signals to stop."""

    def check(self, state: PipelineState[FloatT]) -> bool:
        """Always returns False.

        Parameters
        ----------
        state : PipelineState[FloatT]
            The current pipeline state.

        Returns
        -------
        bool
            Always False.
        """

        return False
