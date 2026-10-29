"""Stopping strategy based on maximum iteration count."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from ..breakpointer import Breakpointer
from ..pipeline_state import PipelineState


class StepBreakpointer(Breakpointer):
    """Stops the pipeline after a fixed number of iterations.

    The counter resets to 0 only when this breakpointer reaches the limit. If the pipeline is stopped by
    another breakpointer or by a step error first, the count carries over to the next use of the instance.

    Parameters
    ----------
    max_steps : int
        Maximum number of iterations before stopping. Must be >= 1.

    Attributes
    ----------
    max_steps : int
        Maximum iteration limit.

    Raises
    ------
    ValueError
        If ``max_steps`` is <= 0.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        check
    """

    def __init__(self, max_steps: int):
        self._validate(max_steps)

        self.max_steps = max_steps
        self._current_step = 0

    def _validate(self, max_steps: int):
        if max_steps <= 0:
            raise ValueError("The maximum number of steps must be greater than or equal to 1")

    def check(self, state: PipelineState) -> bool:
        """Evaluate if the iteration limit has been reached, resetting counter on True."""

        self._current_step += 1

        if self._current_step >= self.max_steps:
            self._current_step = 0
            return True

        return False
