"""Implements the Expectation-Conditional Maximization (ECM) algorithm.

This module provides the ``ECM`` class, which is a concrete
implementation of the :class:`~rework_pysatl_mpest.estimators.base_estimator.BaseEstimator`.
It uses a pipeline architecture (:class:`~rework_pysatl_mpest.estimators.iterative.pipeline.Pipeline`)
to fit the parameters of a mixture model to data.
"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence

from numpy.typing import ArrayLike

from ..core import MixtureModel
from ..optimizers import Optimizer
from ..typings import DType
from .base_estimator import BaseEstimator
from .iterative import (
    Breakpointer,
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    Pipeline,
    Pruner,
)
from .iterative._logger import IterationsHistory


class ECM(BaseEstimator[DType]):
    """An estimator that implements the Expectation-Conditional Maximization (ECM) algorithm.

    This class encapsulates the logic for the ECM algorithm, a variant of the
    classic Expectation-Maximization (EM) algorithm. It constructs and executes
    a :class:`~.Pipeline` consisting of an expectation step
    (:class:`~.ExpectationStep`) and a conditional maximization step
    (:class:`~.MaximizationStep`).

    The key feature of this ECM implementation is that the maximization (M-step)
    is partitioned into separate, smaller optimization problems—one for each
    component in the mixture. For each component, all of its optimizable
    (non-fixed) parameters are estimated simultaneously by maximizing the
    Q-function. This component-wise update simplifies the optimization process.

    The overall fitting process can be customized with stopping criteria,
    component pruning strategies, and a numerical optimizer.

    Parameters
    ----------
    breakpointers : Sequence[Breakpointer]
        A sequence of strategies that define the stopping conditions for the
        iterative process.
    pruners : Sequence[Pruner]
        A sequence of strategies for removing (pruning) components from the
        mixture model during fitting.
    optimizer : Optimizer
        A numerical optimizer instance used in the maximization step to find
        the optimal parameters.

    Attributes
    ----------
    breakpointers : list[Breakpointer]
        The list of objects that determine when the fitting process should
        terminate.
    pruners : list[Pruner]
        The list of objects that may remove components from the mixture during
        the fitting process.
    optimizer : Optimizer
        The numerical optimizer used for parameter estimation.
    logger : IterationsHistory | None
        An object that collects information about each iteration.
        This attribute is only available after the :meth:`fit` method has been called.
        Accessing it beforehand will raise an :class:`AttributeError`.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        fit
    """

    def __init__(self, breakpointers: Sequence[Breakpointer], pruners: Sequence[Pruner], optimizer: Optimizer) -> None:
        self.breakpointers = list(breakpointers)
        self.pruners = list(pruners)
        self.optimizer = optimizer
        self._logger: IterationsHistory[DType] | None = None

    @property
    def logger(self) -> IterationsHistory[DType]:
        """An object that collects information about each iteration.

        Raises
        ------
        AttributeError
            If accessed before the `fit` method has been called at least once.
        """

        if self._logger is None:
            raise AttributeError("Logger is not available. Call the 'fit' method first.")
        return self._logger

    def fit(self, X: ArrayLike, mixture: MixtureModel[DType], once_in_iterations: int = 1) -> MixtureModel[DType]:
        """Fits the mixture model to the data using the ECM algorithm.

        This method sets up and runs an iterative pipeline to estimate the
        parameters of a given mixture model based on the input data. At each
        iteration, it performs an E-step and an M-step. The process repeats
        until one of the stopping criteria is met.

        Parameters
        ----------
        X : ArrayLike
            The input dataset for fitting the model.
        mixture : MixtureModel[DType]
            The initial mixture model to be fitted.
        once_in_iterations : int, optional
            The logging frequency. A value of `n` means logging occurs every
            `n` iterations. Defaults to 1.

        Returns
        -------
        MixtureModel[DType]
            The mixture model with the estimated parameters.
        """

        blocks = []
        for i, comp in enumerate(mixture):
            block = OptimizationBlock(i, comp.params_to_optimize, MaximizationStrategy.QFUNCTION)
            blocks.append(block)

        pipeline: Pipeline[DType] = Pipeline(
            [ExpectationStep(), MaximizationStep(blocks, self.optimizer)],
            self.breakpointers,
            self.pruners,
            once_in_iterations,
        )
        result = pipeline.fit(X, mixture)
        self._logger = pipeline.logger
        return result
