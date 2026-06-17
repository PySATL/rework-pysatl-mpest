"""Provides a pruner for removing mixture components with small weights.

This module contains the `PriorThresholdsPruner` class, a concrete
implementation of the `Pruner` abstract base class. This pruner is designed
to be used within a `Pipeline` to simplify a mixture model by removing
components whose prior probabilities (weights) fall below a specified
threshold.
"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from copy import copy

from ....typings import FloatingType, Scalar
from ..pipeline_state import PipelineState
from ..pruner import Pruner


class PriorThresholdPruner[FloatT: FloatingType](Pruner[FloatT]):
    """A pruner that removes mixture components based on a weight threshold.

    This pruner iterates through the components of a :class:`pysatl_mpest.core.MixtureModel`
    after a pipeline iteration and removes any component whose weight is less than the
    specified :attr:`threshold`. This helps to simplify the model and discard
    insignificant components. The weights of the remaining components are
    automatically renormalized.

    Parameters
    ----------
    threshold : Scalar
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

    def __init__(self, threshold: Scalar) -> None:
        self.threshold = float(threshold)
        if not (0 < self.threshold < 1):
            raise ValueError("Threshold must be between 0 and 1.")

    def prune(self, state: PipelineState[FloatT]) -> tuple[PipelineState[FloatT], list[int]]:
        """Removes components from the mixture whose weights are below the threshold.

        This method inspects :attr:`state.curr_mixture` and removes any component
        whose weight is less than :attr:`.threshold`. The pruning logic ensures
        that at least one component always remains in the model. The weights of
        the remaining components are automatically re-normalized by the
        :class:`pysatl_mpest.core.MixtureModel`.

        Parameters
        ----------
        state : PipelineState[FloatT]
            The current state of the pipeline, containing the mixture model
            to be pruned.

        Returns
        -------
        tuple[PipelineState[FloatT], list[int]]
        A tuple containing:
        - The updated pipeline state. If components were removed,
          :attr:`state.curr_mixture` will be a new,
          smaller :class:`pysatl_mpest.core.MixtureModel` instance.
          Otherwise, the original state object is returned unchanged.
        - A list of the indices of components that were removed (sorted in increasing order).
        """

        # TODO: Now this implementation replace mixture with new mixture, so logger will not log this.

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

        # Create a new PipelineState with the pruned mixture
        new_state = PipelineState(
            X=state.X, H=state.H, prev_mixture=state.prev_mixture, curr_mixture=new_mixture, error=state.error
        )

        return (new_state, sorted(to_remove))
