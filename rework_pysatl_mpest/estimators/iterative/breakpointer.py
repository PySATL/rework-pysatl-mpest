"""Provides an abstract base class for pipeline stopping criteria.

This module defines the `Breakpointer` abstract class, which serves as a
contract for implementing custom stopping conditions (or convergence criteria)
within an iterative estimation `Pipeline`.
"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import Generic

from rework_pysatl_mpest.typings import DType

from .pipeline_state import PipelineState


class Breakpointer(ABC, Generic[DType]):
    """Abstract base class for a pipeline stopping condition.

    A Breakpointer is responsible for inspecting the :class:`PipelineState` after each
    full iteration of a :class:`Pipeline` to determine whether the estimation process
    should terminate. Concrete implementations could check for convergence,
    maximum number of iterations, or other custom criteria.

    The logic of breakpointers comes after pruners, so it is necessary to
    take into account possible differences in the number of components in
    :attr:`state.prev_mixture` and :attr:`state.curr_mixture`.

    Notes
    -----
    Subclasses must implement the abstract method :meth:`check` to define the
    specific stopping logic.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        check
    """

    @abstractmethod
    def check(self, state: PipelineState[DType]) -> bool:
        """Evaluates the pipeline state to determine if it should stop.

        Parameters
        ----------
        state : PipelineState[DType]
            The current state of the pipeline after a full iteration,
            containing the current and previous mixture models.

        Returns
        -------
        bool
            True if the stopping condition is met and the pipeline should
            terminate, False otherwise.
        """
