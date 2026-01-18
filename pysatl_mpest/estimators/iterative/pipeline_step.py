"""Provides an abstract base class for a step in an estimation pipeline.

This module defines the `PipelineStep` abstract base class, which serves as a
contract for all individual processing steps within a :class:`Pipeline` estimator.
Each step takes the current state of the pipeline, performs an operation,
and returns the updated state.
"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import Generic

from ...typings import DType
from .pipeline_state import PipelineState


class PipelineStep(ABC, Generic[DType]):
    """Abstract base class for a single step in a processing pipeline.

    This class defines the interface for an operation that can be executed as
    part of an iterative estimation process managed by a :class:`Pipeline`.

    Each step receives the current :class:`PipelineState`, performs a specific operation
    (e.g., an E-step or M-step), and returns the updated state.

    Attributes
    ----------
    available_next_steps: list[type[PipelineStep]]
        A list of step types that can legally follow this step in a pipeline.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        run

    Notes
    -----
    Subclasses must implement the abstract property :attr:`available_next_steps` and
    the abstract method :meth:`run`.
    """

    @property
    @abstractmethod
    def available_next_steps(self) -> list[type["PipelineStep"]]:
        """list[Type[PipelineStep]]: A list of step types that can follow this step.

        This property is used by the :class:`Pipeline` to validate the sequence of
        steps, ensuring a logical processing flow.
        """

    @abstractmethod
    def run(self, state: PipelineState[DType]) -> PipelineState[DType]:
        """Executes the logic of the pipeline step.

        This method processes the given pipeline state. Implementations can
        either modify the `state` object in-place or create and return a new
        :class:`PipelineState` instance.

        To maintain a flexible and explicit API, this method must always return
        a :class:`PipelineState` object. The calling code will use the returned value
        as the new state.

        Parameters
        ----------
        state : PipelineState[DType]
            The current state of the pipeline to be processed. Note that this
            object can be mutated by the method.

        Returns
        -------
        PipelineState[DType]
            The updated state of the pipeline. This can be the mutated input
            `state` object or a completely new instance.
        """

    @abstractmethod
    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        """
        Cleans up internal per-component state after pruning.

        This method is called by the pipeline after a pruning step has removed
        components from the mixture model. It receives the list of indices
        (relative to the pre-pruning mixture) that were removed, and should
        update or remove any internal data structures that are tied to those
        components.

        Common use cases include:
        - Deleting cached values or buffers associated with pruned components
        - Re-indexing remaining component-specific data to maintain contiguous
          indexing

        The implementation should ensure that after this method completes,
        all internal state is consistent with the new (smaller) mixture model,
        where component indices are renumbered contiguously starting from 0.

        Parameters
        ----------
        removed_components_indices : list[int]
            A list of component indices that were removed during pruning.
            These indices refer to positions in the mixture model **before**
            pruning occurred. The list is guaranteed to be sorted in ascending
            order and contain no duplicates.
        """
