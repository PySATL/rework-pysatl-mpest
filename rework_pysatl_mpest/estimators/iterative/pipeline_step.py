"""Provides an abstract base class for a step in an estimation pipeline.

This module defines the `PipelineStep` abstract base class, which serves as a
contract for all individual processing steps within a `Pipeline` estimator.
Each step takes the current state of the pipeline, performs an operation,
and returns the updated state.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from rework_pysatl_mpest.estimators.iterative.pipeline_state import PipelineState


class PipelineStep(ABC):
    """Abstract base class for a single step in a processing pipeline.

    This class defines the interface for an operation that can be executed as
    part of an iterative estimation process managed by a `Pipeline`.

    .. rubric:: Instance attributes

    :ivar: available_next_steps

    .. rubric:: Implementation Requirements

    Subclasses must:
        1. Implement the :attr:`~available_next_steps` property to determine which steps can be next in pipeline.
        2. Implement abstract method `~run` to define logic of the step.
    """

    @property
    @abstractmethod
    def available_next_steps(self) -> list[type["PipelineStep"]]:
        """list[Type[PipelineStep]]: A list of step types that can follow this step.

        This property is used by the `Pipeline` to validate the sequence of
        steps, ensuring a logical processing flow.
        """

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """Executes the logic of the pipeline step.

        This method processes the given pipeline state. Implementations of this
        method can either modify the `state` object in-place or create and
        return a new `PipelineState` instance.

        To maintain a flexible and explicit API, this method must always
        return a `PipelineState` object. The calling code will always use the
        returned value as the new state.

        Args:
            state (PipelineState): The current state of the pipeline to be
                processed. Note that this object can be mutated by the method.

        Returns:
            PipelineState: The updated state of the pipeline. This can be the
                mutated input `state` object or a completely new instance.
        """
