"""Defines configuration structures for the maximization step in a pipeline.

This module provides data structures used to configure and control the
maximization of an iterative estimation process, such as the one
implemented by :class:`MaximizationStep`.
It includes an enumeration of possible optimization strategies and a dataclass
to define a specific optimization task for a component's parameters.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from dataclasses import dataclass
from enum import Enum, auto


class MaximizationStrategy(Enum):
    """Enumerates the available strategies for the :class:`MaximizationStep`.

    This enumeration defines the objective functions that can be used to
    update the parameters of mixture components during the maximization step.

    Attributes
    ----------
    QFUNCTION
        Indicates that the optimization should maximize the Q-function,
        which is the expected value of the complete-data log-likelihood.
    MOMENTS
    Indicates that the optimization should use the Method of Moments,
    matching theoretical moments to empirical weighted moments.
    """

    QFUNCTION = auto()
    MOMENTS = auto()


@dataclass
class OptimizationBlock:
    """A configuration block for optimizing a specific component's parameters.

    This dataclass specifies a single optimization task within a :class:`~MaximizationStep`.
    It defines which component to update, which of its parameters to optimize,
    and which objective function (strategy) to use for the optimization.

    Parameters
    ----------
    component_id : int
        The zero-based index of the component within the
        :class:`~rework_pysatl_mpest.core.MixtureModel` to be optimized.
    params_to_optimize : set[str]
        A set of parameter names (as strings) for the specified component
        that should be included in the optimization.
    maximization_strategy : MaximizationStrategy
        The optimization strategy to apply, chosen from the
        :class:`MaximizationStrategy` enum.
    """

    component_id: int
    params_to_optimize: set[str]
    maximization_strategy: MaximizationStrategy
