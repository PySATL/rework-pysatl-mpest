"""Defines a data structure for managing the state of an iterative estimator.

This module provides the `PipelineState` dataclass, which acts as a container
for all data passed between different steps of a :class:`Pipeline`, such as
:class:`PipelineStep` or :class:`Pruner`. It holds the input data, the mixture model being
optimized, and other metadata required for the estimation process.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import Optional

from numpy import float64
from numpy.typing import NDArray

from ...core import MixtureModel


@dataclass
class PipelineState:
    """Represents the state of a pipeline at a specific point in its execution.

    This dataclass is a mutable container that centralizes all information
    needed by the steps of a :class:`Pipeline`.
    An instance of this class is created at the beginning of a pipeline run and
    is then passed sequentially through each step, which can modify it.

    Args
    ----------
    X : NDArray[float64]
        The input data sample. This data is typically treated as read-only
        throughout the pipeline's execution.
    H : NDArray[float64] | None
        The responsibility matrix (posterior probabilities). `H[i, j]`
        represents the probability that data point `i` belongs to component
        `j`. It may not be computed at every step.
    prev_mixture : MixtureModel | None
        A snapshot of the mixture model from the previous iteration. This is
        useful for convergence checks, such as comparing log-likelihood values.
        It is `None` at the start of the pipeline.
    curr_mixture : MixtureModel
        The current state of the mixture model that is being actively
        optimized by the pipeline steps.
    error : Exception | None
        A container for any exception that occurs during a pipeline step. If a
        step encounters a non-fatal error, it can place an exception object
        here to signal the pipeline to terminate gracefully.
    optimization_blocks : list[OptimizationBlock] | None
        The list of optimization blocks for component parameter optimization.
        Each block corresponds to a component in the mixture model.
    """

    X: NDArray[float64]
    H: Optional[NDArray[float64]]
    prev_mixture: Optional[MixtureModel]
    curr_mixture: MixtureModel
    error: Optional[Exception]
    optimization_blocks: Optional[list] = None
