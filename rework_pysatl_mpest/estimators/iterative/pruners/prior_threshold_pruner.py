"""Provides a pruner for removing mixture components with small weights.

This module contains the `PriorThresholdsPruner` class, a concrete
implementation of the `Pruner` abstract base class. This pruner is designed
to be used within a `Pipeline` to simplify a mixture model by removing
components whose prior probabilities (weights) fall below a specified
threshold.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from copy import copy
from typing import Optional

from ..pipeline_state import PipelineState
from ..pruner import Pruner


class PriorThresholdPruner(Pruner):
    """A pruner that removes mixture components based on a weight threshold.

    This pruner iterates through the components of a :class:`rework_pysatl_mpest.core.MixtureModel`
    after a pipeline iteration and removes any component whose weight is less than the
    specified :attr:`threshold`. This helps to simplify the model and discard
    insignificant components. The weights of the remaining components are
    automatically renormalized.

    Parameters
    ----------
    threshold : float
        The minimum weight a component must have to be retained. This value
        must be in the exclusive range (0, 1).

    Attributes
    ----------
    threshold : float
        Stores the weight threshold used for pruning.

    Raises
    ------
    ValueError
        If :attr:`threshold` is not strictly between 0 and 1.

    Methods
    -------

    .. autosummary::
        :toctree: generated/

        prune
    """

    def __init__(self, threshold: float) -> None:
        if not (0 < threshold < 1):
            raise ValueError("Threshold must be between 0 and 1.")
        self.threshold = threshold

    def prune(self, state: PipelineState) -> tuple[PipelineState, Optional[list[int]]]:
        """Removes components from the mixture whose weights are below the threshold.

        This method inspects :attr:`state.curr_mixture` and removes any component
        whose weight is less than :attr:`.threshold`. The pruning logic ensures
        that at least one component always remains in the model. The weights of
        the remaining components are automatically re-normalized by the
        :class:`rework_pysatl_mpest.core.MixtureModel`.

        Parameters
        ----------
        state : PipelineState
            The current state of the pipeline, containing the mixture model
            to be pruned.

        Returns
        -------
        PipelineState
            The updated pipeline state. If components were removed, :attr:`state.curr_mixture` will be a new,
            smaller :class:`rework_pysatl_mpest.core.MixtureModel` instance.
            Otherwise, the original state is returned.
        removed_components_indices : list[int] | None
            Tracks which component indices were removed during pruning.
        """

        # TODO: Now this implementation replace mixture with new mixture, so logger will not log this.

        mixture = copy(state.curr_mixture)
        removed_components_indices = []

        for i in range(mixture.n_components - 1, -1, -1):  # Reverse order to avoid indexing confusion
            if mixture.weights[i] < self.threshold and mixture.n_components > 1:
                mixture.remove_component(i)
                removed_components_indices.append(i)

        state.curr_mixture = mixture
        removed_components_indices = sorted(removed_components_indices)

        return (state, removed_components_indices)
