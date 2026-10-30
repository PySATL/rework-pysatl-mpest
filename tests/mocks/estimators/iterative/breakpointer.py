"""Mock implementations for Breakpointer."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

from pysatl_mpest.estimators.iterative.breakpointer import Breakpointer
from pysatl_mpest.estimators.iterative.pipeline_state import PipelineState


class MockMaxIterationsBreakpointer(Breakpointer):
    """Mock breakpointer that stops after a specific number of iterations."""

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        self._current_iteration = 0

    def check(self, state: PipelineState) -> bool:
        self._current_iteration += 1
        return self._current_iteration >= self.max_iterations


class MockCallbackBreakpointer(Breakpointer):
    """Mock breakpointer that delegates check logic to a callback."""

    def __init__(self, check_callback: Callable[[PipelineState], bool]) -> None:
        self.check_callback = check_callback

    def check(self, state: PipelineState) -> bool:
        return self.check_callback(state)


class MockNeverBreakpointer(Breakpointer):
    """Mock breakpointer that never signals to stop."""

    def check(self, state: PipelineState) -> bool:
        return False
