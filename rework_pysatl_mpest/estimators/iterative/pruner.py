"""Defines the abstract base class for mixture pruning strategies.

This module provides the `Pruner` abstract base class, which serves as an
interface for implementing various strategies to remove components from a
mixture model during an iterative estimation process.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import Generic

from ...utils.typings import DType
from .pipeline_state import PipelineState


class Pruner(ABC, Generic[DType]):
    """Abstract base class for component pruning strategies.

    Pruner subclasses implement the logic for identifying and removing
    redundant, insignificant, or degenerate components from a :class:`rework_pysatl_mpest.core.MixtureModel`
    within an iterative pipeline (:class:`Pipeline`).
    Pruning is typically applied after a full iteration of the pipeline's steps to clean up the model
    before the next iteration.

    Methods
    -------

    .. autosummary::
        :toctree: generated/

        prune

    Notes
    -----

    Subclasses must implement the abstract method :meth:`prune` to define the
    specific component removal logic.
    """

    @abstractmethod
    def prune(self, state: PipelineState[DType]) -> PipelineState[DType]:
        """Analyzes the pipeline state and prunes components from the mixture.

        This method is called by the :class:`Pipeline` to inspect the current mixture
        model and decide whether to remove one or more components based on the
        implemented strategy (e.g., if a component's weight falls below a
        threshold).

        Parameters
        ----------
        state : PipelineState[DType]
            The current state of the pipeline, containing the data, mixture
            model, and other relevant information.

        Returns
        -------
        PipelineState[DType]
            The new pipeline state. If components were removed, this state
            contains the updated mixture model. Otherwise, it returns the
            original state.
        """
