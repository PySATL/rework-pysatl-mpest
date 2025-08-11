"""Provides an abstract base class for pipeline stopping criteria.

This module defines the `Breakpointer` abstract class, which serves as a
contract for implementing custom stopping conditions (or convergence criteria)
within an iterative estimation `Pipeline`.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from rework_pysatl_mpest.estimators.iterative.pipeline_state import PipelineState


class Breakpointer(ABC):
    """Abstract base class for a pipeline stopping condition.

    A Breakpointer is responsible for inspecting the `PipelineState` after each
    full iteration of a `Pipeline` to determine whether the estimation process
    should terminate. Concrete implementations could check for convergence,
    maximum number of iterations, or other custom criteria.

    .. rubric:: Implementation Requirements

    Subclasses must:
        1. Implement abstract method `~check` for checking stopping criteria.
    """

    @abstractmethod
    def check(self, state: PipelineState) -> bool:
        """Evaluates the pipeline state to determine if it should stop.

        Args:
            state (PipelineState): The current state of the pipeline after a
                full iteration, containing the current and previous mixture
                models.

        Returns:
            bool: `True` if the stopping condition is met and the pipeline
                should terminate, `False` otherwise.
        """
