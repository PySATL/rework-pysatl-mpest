"""Mock implementations for Pruner."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_mpest.estimators.iterative.pipeline_state import PipelineState
from pysatl_mpest.estimators.iterative.pruner import Pruner


class MockPruner(Pruner):
    """Mock pruner that removes specific components at a specific iteration."""

    def __init__(self, components_to_remove: list[int], iteration_to_prune: int) -> None:
        self.components_to_remove = components_to_remove
        self.iteration_to_prune = iteration_to_prune
        self._current_iteration = 0

    def prune(self, state: PipelineState) -> tuple[PipelineState, list[int]]:
        self._current_iteration += 1

        if self._current_iteration == self.iteration_to_prune and self.components_to_remove:
            removed_indices = []
            # Remove in reverse order to keep indices stable during removal
            sorted_to_remove = sorted(self.components_to_remove, reverse=True)
            for idx in sorted_to_remove:
                if 0 <= idx < state.curr_mixture.n_components:
                    state.curr_mixture.remove_component(idx)
                    removed_indices.append(idx)
            return state, sorted(removed_indices)

        return state, []
