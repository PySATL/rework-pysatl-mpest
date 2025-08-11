"""Provides a configurable, iterative estimator for mixture models.

This module defines the `Pipeline` class, which orchestrates an iterative
estimation process by executing a sequence of processing steps. It allows for
the flexible construction of algorithms like Expectation-Maximization (EM) by
combining different steps (`PipelineStep`), stopping criteria (`Breakpointer`),
and component pruning strategies (`Pruner`).
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from rework_pysatl_mpest.core.mixture import MixtureModel
from rework_pysatl_mpest.estimators.base_estimator import BaseEstimator
from rework_pysatl_mpest.estimators.iterative.breakpointer import Breakpointer
from rework_pysatl_mpest.estimators.iterative.pruner import Pruner
from rework_pysatl_mpest.estimators.iterative.pipeline_state import PipelineState
from rework_pysatl_mpest.estimators.iterative.pipeline_step import PipelineStep


class Pipeline(BaseEstimator):
    """An estimator that fits a mixture model via a configurable iterative process.

    The pipeline executes a sequence of defined steps in a loop. After each full
    sequence of steps, pruning strategies are applied, and stopping conditions
    (breakpointers) are checked. The loop continues until a breakpointer signals
    to stop. This allows for building complex, multi-stage estimation algorithms,
    such as variants of the EM algorithm.

    Attributes:
        _breakpointers (list[Breakpointer]): A list of objects that determine
            when the fitting process should terminate.
        _pruners (list[Pruner]): A list of objects that may remove components
            from the mixture during the fitting process.
        _steps (list[PipelineStep]): An ordered list of operations to be
            performed in each iteration.
        state (Optional[PipelineState]): The current state of the pipeline,
            containing the data, mixture models, and other relevant information.
            It is populated during the `fit` process.

    Args:
        breakpointers (list[Breakpointer]): A list of strategies that define
            the stopping conditions for the iterative process.
        pruners (list[Pruner]): A list of strategies for removing
            components from the mixture model during fitting.
        steps (list[PipelineStep]): An ordered sequence of steps to be
            executed in each iteration of the pipeline.

    Raises:
        ValueError: If the sequence of `steps` is invalid (i.e., a step
            is followed by a step not listed in its `available_next_steps`).

    """

    def __init__(self, breakpointers: list[Breakpointer], pruners: list[Pruner], steps: list[PipelineStep]) -> None:
        self._validate(steps)
        self._breakpointers = breakpointers
        self._pruners = pruners
        self._steps = steps
        self.state: Optional[PipelineState] = None

    def _validate(self, steps: list[PipelineStep]):
        """Validates the sequence of pipeline steps.

        Checks if each step in the pipeline can legally be followed by the next
        one, based on the `available_next_steps` property of each step.

        Args:
            steps (list[PipelineStep]): The sequence of steps to validate.

        Raises:
            ValueError: If the pipeline configuration is invalid, meaning a step
                is followed by an incompatible one.
        """

        prev_step, wrong_step = None, None
        for i, step in enumerate(steps):
            if i < len(steps):
                if type(steps[i + 1]) in step.available_next_steps:
                    continue
                prev_step, wrong_step = step, steps[i+1]
            else:
                if type(step) in steps[-1].available_next_steps:
                    continue
                prev_step, wrong_step = step, steps[-1]

        if prev_step and wrong_step:
            raise ValueError(f"Wrong pipeline configuration. Step '{type(prev_step)}' have"
                             f"available next steps:'{prev_step.available_next_steps}', but got '{type(wrong_step)}'")

    def fit(self, X: ArrayLike, mixture: MixtureModel) -> MixtureModel:
        """Fits the mixture model to the data using the configured pipeline.

        This method initializes the pipeline's state and runs the main loop.
        The loop consists of executing all `steps` in order, followed by
        all `pruners`. This cycle repeats until any `breakpointers` indicate
        that the process should stop.

        Args:
            X (ArrayLike): The input data sample.
            mixture (MixtureModel): The initial mixture model to be fitted.
                An internal copy of this model will be modified throughout
                the process.

        Returns:
            MixtureModel: The fitted mixture model after the pipeline has
                converged or been stopped.
        """

        X = np.asarray(X, dtype=np.float64)
        self.state = PipelineState(X, None, None, mixture, None)

        while not any([breakpointer.check(self.state) for breakpointer in self._breakpointers]):
            for step in self._steps:
                result_state = step.run(self.state)
                if result_state.error:
                    return result_state.curr_mixture
                self.state = result_state

            for pruner in self._pruners:
                self.state = pruner.prune(self.state)

        return self.state.curr_mixture
