"""Mock implementations for Pruner."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_mpest.estimators.iterative.pipeline_state import PipelineState
from pysatl_mpest.estimators.iterative.pruner import Pruner
from pysatl_mpest.typings import FloatingType


class MockPruner[FloatT: FloatingType](Pruner[FloatT]):
    """A mock pruner that removes specific components at a specific iteration.

    Parameters
    ----------
    components_to_remove : list[int]
        The indices of the components to remove.
    iteration_to_prune : int
        The exact iteration number at which the pruning should occur.
    """

    def __init__(self, components_to_remove: list[int], iteration_to_prune: int) -> None:
        self.components_to_remove = components_to_remove
        self.iteration_to_prune = iteration_to_prune
        self._current_iteration = 0

    def prune(self, state: PipelineState[FloatT]) -> tuple[PipelineState[FloatT], list[int]]:
        """Removes specified components if the current iteration matches the target.

        Parameters
        ----------
        state : PipelineState[FloatT]
            The current pipeline state.

        Returns
        -------
        PipelineState[FloatT]
            The modified (or unmodified) pipeline state.
        list[int]
            The list of removed component indices.
        """

        self._current_iteration += 1

        if self._current_iteration == self.iteration_to_prune and self.components_to_remove:
            removed_indices = []
            # Remove in reverse order to keep indices stable during removal
            sorted_to_remove = sorted(self.components_to_remove, reverse=True)
            for idx in sorted_to_remove:
                if 0 <= idx < state.curr_mixture.n_components:
                    state.curr_mixture.remove_component(idx)
                    removed_indices.append(idx)

            # Return in original order as requested, sorted ascending for standard pipeline
            return state, sorted(removed_indices)

        return state, []
