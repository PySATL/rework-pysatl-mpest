"""Provides a configurable, iterative estimator for mixture models.

This module defines the `Pipeline` class, which orchestrates an iterative
estimation process by executing a sequence of processing steps. It allows for
the flexible construction of algorithms like Expectation-Maximization (EM) by
combining different steps (:class:`PipelineStep`), stopping criteria (:class:`Breakpointer`),
and component pruning strategies (:class:`Pruner`).
"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from collections.abc import Sequence
from copy import copy

import numpy as np
from numpy.typing import ArrayLike

from ...core import MixtureModel
from ...exceptions import NumericalStabilityError
from ...typings import DType
from ..base_estimator import BaseEstimator
from ._iteration_history import IterationRecord, IterationsHistory
from .breakpointer import Breakpointer
from .pipeline_state import PipelineState
from .pipeline_step import PipelineStep
from .pruner import Pruner


class Pipeline(BaseEstimator[DType]):
    """An estimator that fits a mixture model via a configurable iterative process.

    The pipeline executes a sequence of defined steps in a loop. After each full
    sequence of steps, pruning strategies are applied, and stopping conditions
    are checked. The loop continues until a breakpointer signals to stop.
    This allows for building complex, multi-stage estimation algorithms,
    such as variants of the EM algorithm.

    The core components used for configuration are:

    - :class:`.PipelineStep`
    - :class:`Breakpointer`
    - :class:`Pruner`

    Parameters
    ----------
    steps : Sequence[PipelineStep]
        An ordered sequence of steps to be executed in each iteration of the
        pipeline.
    breakpointers : Sequence[Breakpointer]
        A sequence of strategies that define the stopping conditions for the
        iterative process. This list cannot be empty.
    pruners : Sequence[Pruner] | None, optional
        A sequence of strategies for removing components from the mixture model
        during fitting. Defaults to None, meaning no pruning is performed.
    once_in_iterations: int, optional
        The iteration recording frequency. A value of `n` means recording occurs every
        `n` iterations. Defaults to 1 (record every iteration).

    Attributes
    ----------
    steps : list[PipelineStep]
        The ordered list of operations to be performed in each iteration.
    breakpointers : list[Breakpointer]
        The list of objects that determine when the fitting process should
        terminate.
    pruners : list[Pruner]
        The list of objects that may remove components from the mixture during
        the fitting process.
    history : IterationsHistory[DType]
        object that collects comprehensive information about each
        iteration of a :class:`Pipeline` estimator.
    Raises
    ------
    ValueError
        If the sequence of :attr:`steps` is empty or invalid (i.e., a step is
        followed by a step not listed in its :attr:`available_next_steps`), or if
        the sequence of :attr:`breakpointers` is empty.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        fit
    """

    def __init__(
        self,
        steps: Sequence[PipelineStep],
        breakpointers: Sequence[Breakpointer],
        pruners: Sequence[Pruner] | None = None,
        once_in_iterations: int = 1,
    ):
        self._validate_steps(list(steps))

        if not breakpointers:
            raise ValueError(
                "The 'breakpointers' list cannot be empty. "
                "At least one stopping criterion must be provided to prevent an infinite loop."
            )

        self.breakpointers = list(breakpointers)
        self.pruners = list(pruners) if pruners else []  # self.pruners will always be list
        self.steps = list(steps)
        self.history = IterationsHistory[DType](once_in_iterations)

    def _validate_steps(self, steps: list[PipelineStep]):
        """Validates the sequence of pipeline steps.

        Checks if each step in the pipeline can legally be followed by the next
        one, based on the :attr:`available_next_steps` property of each step. It also
        checks that the pipeline can be run in a loop (the last step must be
        compatible with the first step).

        Parameters
        ----------
        steps : list[PipelineStep]
            The sequence of steps to validate.

        Raises
        ------
        ValueError
            If the :attr:`steps` list is empty or if the pipeline configuration is
            invalid, meaning a step is followed by an incompatible one.
        """

        if not steps:
            raise ValueError("The 'steps' list cannot be empty for a Pipeline.")

        for i in range(-1, len(steps) - 1, 1):
            curr_step, next_step = steps[i], steps[i + 1]
            available_steps = tuple(curr_step.available_next_steps)

            if not isinstance(next_step, available_steps):
                raise ValueError(
                    f"Wrong pipeline configuration. Step '{curr_step}' have"
                    f"available next steps:'{curr_step.available_next_steps}', but got '{next_step}'"
                )

    def fit(self, X: ArrayLike, mixture: MixtureModel[DType]) -> MixtureModel[DType]:
        """Fits the mixture model to the data using the configured pipeline.

        This method initializes the pipeline's state and runs the main loop.
        The loop consists of executing all :attr:`steps` in order, followed by
        all :attr:`pruners`. This cycle repeats until any `breakpointers` indicate
        that the process should stop.

        Parameters
        ----------
        X : ArrayLike
            The input data sample.
        mixture : MixtureModel[DType]
            The initial mixture model to be fitted. An internal copy of this
            model will be modified throughout the process.

        Returns
        -------
        MixtureModel[DType]
            The fitted mixture model after the pipeline has converged or been
            stopped.
        """

        X = np.asarray(X, dtype=mixture.dtype)
        copied_mixture = copy(mixture)  # Copy to avoid modifying the original object
        state = PipelineState(X, None, None, copied_mixture, None)

        while True:
            # Updating the state before starting an iteration
            state.prev_mixture = copy(state.curr_mixture)

            # Performing steps
            for step in self.steps:
                result_state = step.run(state)
                # Log the error state before exiting
                if result_state.error:
                    if len(self.history) > 0:
                        self.history[-1].error = result_state.error
                    else:
                        self.history.save_record(
                            IterationRecord(
                                self.history._counter,
                                result_state.curr_mixture,
                                result_state.X,
                                result_state.H,
                                None,
                                result_state.error,
                            )
                        )

                    # Handle numerical stability errors by attempting a restart with higher precision
                    if isinstance(result_state.error, NumericalStabilityError):
                        new_dtype = np.promote_types(copied_mixture.dtype, np.float64).type
                        if new_dtype is not copied_mixture.dtype:
                            new_mixture = copied_mixture.astype(new_dtype)

                            msg = (
                                "Numerical stability issue detected. "
                                f"Restarting pipeline with higher precision ({new_dtype.__name__})."
                            )
                            warnings.warn(msg, UserWarning)
                            # Recursively call fit with the new, higher-precision model
                            return self.fit(X, new_mixture)

                    warnings.warn(
                        f"Pipeline fitting stopped prematurely due to an error in step "
                        f"'{step.__class__.__name__}': {state.error}",
                        RuntimeWarning,
                    )
                    return result_state.curr_mixture
                state = result_state

            # Pruning
            for pruner in self.pruners:
                state, removed_components_indices = pruner.prune(state)
                if removed_components_indices:
                    # Update optimization blocks in steps to drop blocks associated with removed components
                    for step in self.steps:
                        step.clear_after_prune(removed_components_indices)

                    # Update responsibility matrix H to match the new mixture size
                    if state.H is not None:
                        state.H = np.delete(state.H, removed_components_indices, axis=1)

            # Save iteration record
            self.history.save_record(
                IterationRecord(self.history._counter, state.curr_mixture, state.X, state.H, self.pruners, state.error)
            )

            # Checking stopping criteria
            if any(breakpointer.check(state) for breakpointer in self.breakpointers):
                break

        return state.curr_mixture
