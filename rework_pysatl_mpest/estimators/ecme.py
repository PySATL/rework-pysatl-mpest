"""Implements the Expectation-Conditional Maximization Either (ECME) algorithm.

This module provides the ``ECME`` class, a concrete implementation of the
:class:`~rework_pysatl_mpest.estimators.base_estimator.BaseEstimator`.
The ECME algorithm is an extension of the Expectation-Conditional Maximization (ECM)
algorithm that allows switching between maximizing the Q-function and directly
maximizing the Observed Data Likelihood (ODL) for different components.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from typing import Literal

from numpy.typing import ArrayLike

from rework_pysatl_mpest.core.mixture import MixtureModel
from rework_pysatl_mpest.estimators.iterative._logger import IterationsHistory
from rework_pysatl_mpest.estimators.iterative.pipeline import Pipeline
from rework_pysatl_mpest.estimators.iterative.steps.block import MaximizationStrategy, OptimizationBlock
from rework_pysatl_mpest.estimators.iterative.steps.expectation_step import ExpectationStep
from rework_pysatl_mpest.estimators.iterative.steps.maximization_step import MaximizationStep
from rework_pysatl_mpest.optimizers.optimizer import Optimizer

from ..typings import DType
from .base_estimator import BaseEstimator
from .iterative import (
    Breakpointer,
    Pruner,
)


class ECME(BaseEstimator[DType]):
    """An estimator that implements the Expectation-Conditional Maximization Either (ECME) algorithm.

    ECME is a generalized iterative algorithm for maximum likelihood estimation.
    Unlike standard ECM, which strictly maximizes the Q-function (conditional
    expectation of the complete-data log-likelihood) during the maximization steps,
    ECME allows replacing some or all of these steps with a direct maximization
    of the actual Observed Data Likelihood (ODL).

    This implementation provides flexibility by allowing the user to specify
    maximization strategies per component. Some components can be optimized via
    the Q-function, while others can be optimized via ODL within the same
    iterative process.

    Parameters
    ----------
    breakpointers : Sequence[Breakpointer[DType]]
        A sequence of strategies that define the stopping conditions for the
        iterative fitting process.
    pruners : Sequence[Pruner[DType]]
        A sequence of strategies for removing (pruning) components from the
        mixture model during fitting.
    optimizer : Optimizer[DType]
        The numerical optimizer used to solve the maximization problems in the
        M-step (or CM-steps).
    default_strategy : Literal["q-func", "odl"], optional
        The default maximization strategy to apply to components that are not
        explicitly assigned to a specific strategy in :meth:`fit`.
        Defaults to "odl".

    Attributes
    ----------
    breakpointers : list[Breakpointer[DType]]
        The list of stopping criteria used by the estimator.
    pruners : list[Pruner[DType]]
        The list of pruning strategies used by the estimator.
    optimizer : Optimizer[DType]
        The numerical optimizer instance.
    default_strategy_name : str
        The name of the default strategy ("q-func" or "odl").
    history : IterationsHistory[DType]
        The history of the fitting process, containing snapshots of the model
        and metrics for each iteration. Only available after :meth:`fit` is called.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        fit
    """

    def __init__(
        self,
        breakpointers: Sequence[Breakpointer[DType]],
        pruners: Sequence[Pruner[DType]],
        optimizer: Optimizer[DType],
        default_strategy: Literal["q-func", "odl"] = "odl",
    ):
        self.breakpointers = list(breakpointers)
        self.pruners = list(pruners)
        self.optimizer = optimizer

        if default_strategy not in ("odl", "q-func"):
            raise ValueError(f"Unknown default_strategy '{default_strategy}'. Expected 'odl' or 'q-func'.")

        self.default_strategy_name = default_strategy

        self._history: IterationsHistory[DType] | None = None

    @property
    def history(self) -> IterationsHistory[DType]:
        """IterationsHistory[DType]: The history of the last fitting process.

        Returns the history of iterations recorded during the last call to :meth:`fit`.

        Raises
        ------
        AttributeError
            If the :meth:`fit` method has not been called yet.
        """

        if self._history is None:
            raise AttributeError("History is not available. Call the 'fit' method first.")
        return self._history

    def _normalize_indices(self, indices: Sequence[int] | int | None) -> set[int]:
        """Normalizes the input indices into a set of integers.

        Parameters
        ----------
        indices : Sequence[int] | int | None
            The input indices, which can be a single integer, a sequence of
            integers, or None.

        Returns
        -------
        set[int]
            A set containing the indices. Returns an empty set if input is None.
        """

        if indices is None:
            return set()
        if isinstance(indices, int):
            return {indices}
        return set(indices)

    def _resolve_index(self, idx: int, n_components: int) -> int:
        """Resolves a component index, handling negative indexing.

        Parameters
        ----------
        idx : int
            The component index to resolve.
        n_components : int
            The total number of components in the mixture.

        Returns
        -------
        int
            The positive index in the range [0, n_components - 1].

        Raises
        ------
        ValueError
            If the resolved index is out of bounds.
        """

        original_idx = idx
        if idx < 0:
            idx += n_components

        if not (0 <= idx < n_components):
            raise ValueError(f"Component index {original_idx} is out of bounds for mixture size {n_components}.")
        return idx

    def fit(
        self,
        X: ArrayLike,
        mixture: MixtureModel[DType],
        q_indices_raw: Sequence[int] | int | None = None,
        odl_indices_raw: Sequence[int] | int | None = None,
        once_in_iterations: int = 1,
    ) -> MixtureModel[DType]:
        """Fits the mixture model to the data using the ECME algorithm.

        This method configures and executes an iterative pipeline. It assigns
        a maximization strategy to each component in the mixture:
        1. Components in ``q_indices_raw`` use Q-function maximization.
        2. Components in ``odl_indices_raw`` use Observed Data Likelihood maximization.
        3. All other components use the ``default_strategy`` defined in ``__init__``.

        Parameters
        ----------
        X : ArrayLike
            The input data sample to fit.
        mixture : MixtureModel[DType]
            The initial mixture model configuration.
        q_indices_raw : Sequence[int] | int | None, optional
            Indices of components that should be optimized by maximizing the
            Q-function. Can be a single integer, a sequence, or None.
            Supports negative indexing.
        odl_indices_raw : Sequence[int] | int | None, optional
            Indices of components that should be optimized by directly maximizing
            the Observed Data Likelihood (ODL). Can be a single integer, a
            sequence, or None. Supports negative indexing.
        once_in_iterations : int, optional
            The frequency of recording iterations in the history.
            Defaults to 1 (record every iteration).

        Returns
        -------
        MixtureModel[DType]
            The fitted mixture model with estimated parameters.

        Raises
        ------
        ValueError
            If a component index is out of bounds for the given mixture,
            or if the same index is present in both ``q_indices_raw`` and
            ``odl_indices_raw``.
        """

        n_components = mixture.n_components

        # 1. Resolve indices relative to the provided mixture
        q_indices = {self._resolve_index(i, n_components) for i in self._normalize_indices(q_indices_raw)}
        odl_indices = {self._resolve_index(i, n_components) for i in self._normalize_indices(odl_indices_raw)}

        # 2. Validation
        intersection = q_indices.intersection(odl_indices)
        if intersection:
            raise ValueError(f"Indices {intersection} specified for both Q-function and ODL strategies.")

        # 3. Build optimization blocks
        blocks = []

        # Map string config to Enum
        default_enum = (
            MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD
            if self.default_strategy_name == "odl"
            else MaximizationStrategy.QFUNCTION
        )

        for i, comp in enumerate(mixture):
            if i in q_indices:
                strategy = MaximizationStrategy.QFUNCTION
            elif i in odl_indices:
                strategy = MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD
            else:
                strategy = default_enum

            blocks.append(
                OptimizationBlock(
                    component_id=i, params_to_optimize=comp.params_to_optimize, maximization_strategy=strategy
                )
            )

        pipeline: Pipeline[DType] = Pipeline(
            [ExpectationStep(), MaximizationStep(blocks, self.optimizer)],
            self.breakpointers,
            self.pruners,
            once_in_iterations,
        )

        result = pipeline.fit(X, mixture)
        self._history = pipeline.history
        return result
