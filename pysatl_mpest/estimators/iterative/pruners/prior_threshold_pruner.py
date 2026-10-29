"""Pruner for removing mixture components with small weights."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from copy import copy

from ....typings import Scalar
from ..pipeline_state import PipelineState
from ..pruner import Pruner


class PriorThresholdPruner(Pruner):
    """Component pruner based on a prior weight threshold.

    Removes components whose weight is strictly below ``threshold``. Guarantees
    that at least one component remains in the mixture. Remaining weights are
    renormalized.

    Parameters
    ----------
    threshold : Scalar
        Minimum weight to retain a component. Must be strictly between 0 and 1.

    Attributes
    ----------
    threshold : float
        Configured weight threshold.

    Raises
    ------
    ValueError
        If ``threshold`` is not in the exclusive range (0, 1).

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        prune
    """

    def __init__(self, threshold: Scalar) -> None:
        self.threshold = float(threshold)
        if not (0 < self.threshold < 1):
            raise ValueError("Threshold must be between 0 and 1.")

    def prune(self, state: PipelineState) -> tuple[PipelineState, list[int]]:
        """Remove components with weights below threshold, retaining at least one component.

        Candidates are removed in index order, at most ``n_components - 1`` of them. If every weight
        falls below the threshold, the component with the highest index is retained, not the one with
        the largest weight.

        Parameters
        ----------
        state : PipelineState
            Current pipeline state.

        Returns
        -------
        tuple[PipelineState, list[int]]
            Tuple of updated pipeline state and sorted list of removed component indices.
        """

        mixture = state.curr_mixture
        n = mixture.n_components

        candidates_to_remove = [i for i, w in enumerate(mixture.weights) if w < self.threshold]

        max_removals = n - 1

        if len(candidates_to_remove) > max_removals:
            to_remove = candidates_to_remove[:max_removals]
        else:
            to_remove = candidates_to_remove

        if not to_remove:
            return (state, [])

        new_mixture = copy(mixture)
        for i in sorted(to_remove, reverse=True):
            new_mixture.remove_component(i)

        new_state = PipelineState(
            X=state.X, H=state.H, prev_mixture=state.prev_mixture, curr_mixture=new_mixture, error=state.error
        )

        return (new_state, sorted(to_remove))
