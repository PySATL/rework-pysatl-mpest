"""Module that provides a :class:`rework_pysatl_mpest.estimators.iterative.Pipeline`
stopping strategy based on the number of iterations"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from ....typings import DType
from ..breakpointer import Breakpointer
from ..pipeline_state import PipelineState


class StepBreakpointer(Breakpointer[DType]):
    """Stops the pipeline after a fixed number of iterations.

    This breakpointer terminates the iterative process once a specified
    maximum number of steps (iterations) has been completed. It maintains an
    internal counter that increments each time its :meth:`check` method is called.

    Parameters
    ----------
    max_steps : int
        The maximum number of iterations to perform before stopping.
        Must be a positive integer.

    Attributes
    ----------
    max_steps : int
        Stores the maximum number of steps.
    _current_step : int
        The current iteration counter. It is reset to 0 after the limit is
        reached to allow the instance to be reused.

    Raises
    ------
    ValueError
        If :attr:`max_steps` is less than or equal to 0.

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
        """Validates the max_steps parameter."""

        if max_steps <= 0:
            raise ValueError("The maximum number of steps must be greater than or equal to 1")

    def check(self, state: PipelineState[DType]) -> bool:
        """Checks if the maximum number of iterations has been reached.

        This method increments the internal step counter and compares it with
        the :attr:`max_steps` limit. If the limit is reached, it resets the
        counter for potential reuse and returns True.

        Parameters
        ----------
        state : PipelineState[DType]
            The current state of the pipeline. This parameter is unused in
            this specific breakpointer but required by the base class interface.

        Returns
        -------
        bool
            True if the iteration limit is reached, False otherwise.
        """

        self._current_step += 1

        if self._current_step >= self.max_steps:
            self._current_step = 0
            return True

        return False
