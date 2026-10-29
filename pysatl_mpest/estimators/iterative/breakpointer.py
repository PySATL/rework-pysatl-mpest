"""Abstract base class for pipeline stopping criteria."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from .pipeline_state import PipelineState


class Breakpointer(ABC):
    """Abstract base class for pipeline stopping conditions.

    Evaluated after pruning in each iteration; implementations must account for
    differing component counts between ``state.prev_mixture`` and ``state.curr_mixture``.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        check
    """

    @abstractmethod
    def check(self, state: PipelineState) -> bool:
        """Evaluate whether pipeline execution should terminate.

        Parameters
        ----------
        state : PipelineState
            Current state after step execution and pruning.

        Returns
        -------
        bool
            True if stopping condition is satisfied, False otherwise.
        """
