"""Defines the abstract base class for mixture pruning strategies.

This module provides the `Pruner` abstract base class, which serves as an
interface for implementing various strategies to remove components from a
mixture model during an iterative estimation process.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from rework_pysatl_mpest.estimators.iterative.pipeline_state import PipelineState


class Pruner(ABC):
    """Abstract base class for component pruning strategies.

    `Pruner` subclasses implement the logic for identifying and removing
    redundant, insignificant, or degenerate components from a `MixtureModel`
    within an iterative pipeline (`Pipeline`).

    .. rubric:: Implementation Requirements

    Subclasses must:
        1. Implement abstract method `~prune` for estimating mixture parameters.
    """

    @abstractmethod
    def prune(self, state: PipelineState) -> PipelineState:
        """Analyzes the pipeline state and prunes components from the mixture.

        This method is called by the `Pipeline` to inspect the current mixture
        model and decide whether to remove one or more components based on the
        implemented strategy (e.g., if a component's weight falls below a
        threshold).

        Args:
            state (PipelineState): The current state of the pipeline, containing
                the data, mixture model, and other relevant information.

        Returns:
            PipelineState: The new pipeline state. If components were removed,
                this state contains the updated mixture model. Otherwise, it
                returns the original state.
        """
