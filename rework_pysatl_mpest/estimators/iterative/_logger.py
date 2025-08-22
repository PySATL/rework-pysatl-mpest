"""Provides logging functionality for the iterative estimation pipeline.

This module defines the `PipelineLogger` class for
logging pipeline execution details, including iteration statistics,
mixture models, and timing information.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import Optional

from numpy import float64
from numpy.typing import NDArray

from rework_pysatl_mpest.core.mixture import MixtureModel
from rework_pysatl_mpest.estimators.iterative.pruner import Pruner


@dataclass
class IterationRecord:
    """Data class representing a single pipeline iteration log record.

    This class captures the complete state of a pipeline iteration after
    pruning has been applied, providing a snapshot for analysis and debugging.

    Attributes
    ----------
    iteration : int
        The iteration number (0-based index).
    mixture : MixtureModel
        The state of the mixture model after pruning in this iteration.
    sample : NDArray[float64]
        The input data sample being processed.
    matrix_of_hidden_variables : Optional[NDArray[float64]]
        The responsibility matrix (posterior probabilities) if available,
        where `H[i, j]` represents the probability that data point `i`
        belongs to component `j`. May be `None` if not computed.
    pruners_used : list[Pruner]
        List of pruner instances that were applied.
        Empty if no pruning occurred.
    error : Optional[Exception]
        Any exception that occurred during the iteration, or `None` if
        the iteration completed successfully.
    """

    iteration: int
    mixture: MixtureModel
    sample: NDArray[float64]
    matrix_of_hidden_variables: Optional[NDArray[float64]]
    pruners_used: Optional[list[Pruner]]
    error: Optional[Exception]


class PipelineLogger:
    """A configurable logger for tracking pipeline execution iterations.

    The `PipelineLogger` collects comprehensive information about each
    iteration of a :class:`Pipeline` estimator. It records the state after
    pruning strategies have been applied, making it ideal for analyzing
    the evolution of the mixture model throughout the estimation process.

    The core components used for configuration are:

    - :class:`IterationRecord`
    - :class:`Pruner`

    Parameters
    ----------
    once_in_iterations : int, optional
        The logging frequency. A value of `n` means logging occurs every
        `n` iterations. Defaults to 1 (log every iteration).

    Attributes
    ----------
    once_in_iterations : int
        The configured logging frequency.
    _counter : int
        Internal counter tracking the current iteration number.
    _logs : list[IterationRecord]
        Collection of all logged iteration records.(Access)

    Raises
    ------
    ValueError
        If `once_in_iterations` is not a positive integer.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        log
        reset
    """

    def __init__(self, once_in_iterations: int = 1) -> None:
        """
        Initialize a new PipelineLogger instance.
        """
        if once_in_iterations < 1:
            raise ValueError("once_in_iterations must be a positive integer")

        self._logs: list[IterationRecord] = []
        self._counter: int = 0
        self.once_in_iterations = once_in_iterations

    def log(self, record: IterationRecord) -> None:
        """Record an iteration snapshot based on the configured frequency.

        This method stores the provided iteration record according to the
        logger's frequency setting. Records are only stored when the current
        iteration counter matches the logging frequency.

        Parameters
        ----------
        record : IterationRecord
            The iteration record to potentially log. The record's iteration
            number should match the logger's internal counter.

        Notes
        -----
        The internal iteration counter is incremented after each call to
        this method, regardless of whether the record is actually stored.
        """
        if self._counter % self.once_in_iterations == 0:
            self._logs.append(record)
        self._counter += 1

    def reset(self) -> None:
        """Remove all stored log records and reset the iteration counter.

        This method clears the internal storage while maintaining the
        configured logging frequency.
        """
        self._logs.clear()
        self._counter = 0
        self.once_in_iterations = 1

    def __len__(self) -> int:
        """Return the number of logged iteration records.

        This method enables the use of the built-in `len()` function to determine
        how many iteration records have been stored by the logger.

        Returns
        -------
        int
        The number of iteration records currently stored in the logger.
        This count reflects only the records that were actually logged
        according to the configured frequency (:attr:`once_in_iterations`).
        """
        return len(self._logs)

    def __getitem__(self, index: int) -> IterationRecord:
        """Retrieve a specific iteration record by index.

        This method enables sequence-like access to logged iteration records,
        supporting both positive and negative indexing. It allows convenient
        retrieval of specific iterations for analysis, visualization, or
        debugging purposes.

        Parameters
        ----------
        index : int
            The index of the iteration record to retrieve. Positive indices
            start from 0 (first record), negative indices count backward
            from the end (-1 for last record).

        Returns
        -------
        IterationRecord
            The iteration record at the specified index, containing the complete
            state snapshot including mixture model, timing information, and
            pruner results for that iteration.

        Raises
        ------
        IndexError
            If the index is out of range for the current collection of logs.
        """
        if index >= len(self) or index < -len(self):
            raise IndexError(f"Index {index} out of range for container containing {len(self)} elements")
        return self._logs[index]
