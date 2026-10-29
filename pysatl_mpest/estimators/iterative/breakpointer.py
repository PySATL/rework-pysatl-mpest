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

    ``check`` may keep state between calls and is expected to be called once per iteration. The pipeline stops
    at the first breakpointer that returns True, so the ones after it are not called in that iteration.

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
