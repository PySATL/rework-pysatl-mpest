"""Expectation-Conditional Maximization Either (ECME) estimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from typing import Literal

from numpy.typing import ArrayLike

from ..core import MixtureModel
from ..optimizers import Optimizer
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
from .iterative._iteration_history import IterationsHistory


class ECME(BaseEstimator):
    """Expectation-Conditional Maximization Either (ECME) estimator.

    Extends ECM by allowing components to be optimized by either maximizing the
    Q-function or directly maximizing the Observed Data Likelihood (ODL).
    All free parameters are estimated, preserving fixed view parameters.

    Parameters
    ----------
    breakpointers : Sequence[Breakpointer]
        Stopping criteria for the iterative pipeline.
    pruners : Sequence[Pruner]
        Component pruning strategies applied during iterations.
    optimizer : Optimizer
        Numerical optimizer used in maximization steps.
    default_strategy : Literal["q-func", "odl"], optional
        Default strategy for components not specified in :meth:`fit`. Defaults to "odl".

    Attributes
    ----------
    breakpointers : list[Breakpointer]
        Configured stopping criteria.
    pruners : list[Pruner]
        Configured component pruning strategies.
    optimizer : Optimizer
        Numerical optimizer instance.
    default_strategy_name : str
        Name of the default strategy ("q-func" or "odl").
    history : IterationsHistory
        Recorded iteration snapshots, accessible after calling :meth:`fit`.

    Raises
    ------
    ValueError
        If ``default_strategy`` is not ``'odl'`` or ``'q-func'``.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        fit
    """

    def __init__(
        self,
        breakpointers: Sequence[Breakpointer],
        pruners: Sequence[Pruner],
        optimizer: Optimizer,
        default_strategy: Literal["q-func", "odl"] = "odl",
    ):
        self.breakpointers = list(breakpointers)
        self.pruners = list(pruners)
        self.optimizer = optimizer

        if default_strategy not in ("odl", "q-func"):
            raise ValueError(f"Unknown default_strategy '{default_strategy}'. Expected 'odl' or 'q-func'.")

        self.default_strategy_name = default_strategy

        self._history: IterationsHistory | None = None

    @property
    def history(self) -> IterationsHistory:
        """IterationsHistory: Recorded history of pipeline iterations after fitting.

        Raises
        ------
        AttributeError
            If accessed before ``fit`` has been called.
        """

        if self._history is None:
            raise AttributeError("History is not available. Call the 'fit' method first.")
        return self._history

    def _normalize_indices(self, indices: Sequence[int] | int | None) -> set[int]:
        if indices is None:
            return set()
        if isinstance(indices, int):
            return {indices}
        return set(indices)

    def _resolve_index(self, idx: int, n_components: int) -> int:
        original_idx = idx
        if idx < 0:
            idx += n_components

        if not (0 <= idx < n_components):
            raise ValueError(f"Component index {original_idx} is out of bounds for mixture size {n_components}.")
        return idx

    def fit(
        self,
        X: ArrayLike,
        mixture: MixtureModel,
        q_indices_raw: Sequence[int] | int | None = None,
        odl_indices_raw: Sequence[int] | int | None = None,
        once_in_iterations: int = 1,
    ) -> MixtureModel:
        """Fit mixture model parameters using the ECME algorithm.

        Parameters
        ----------
        X : ArrayLike
            Input data sample.
        mixture : MixtureModel
            Initial mixture model to fit.
        q_indices_raw : Sequence[int] | int | None, optional
            Component indices to optimize via Q-function maximization. Supports negative indexing.
        odl_indices_raw : Sequence[int] | int | None, optional
            Component indices to optimize via ODL maximization. Supports negative indexing.
        once_in_iterations : int, optional
            Iteration recording frequency in history. Defaults to 1.

        Returns
        -------
        MixtureModel
            Fitted mixture model with estimated parameters.

        Raises
        ------
        ValueError
            If a component index is out of bounds or present in both strategy index sets.
        """

        n_components = mixture.n_components

        q_indices = {self._resolve_index(i, n_components) for i in self._normalize_indices(q_indices_raw)}
        odl_indices = {self._resolve_index(i, n_components) for i in self._normalize_indices(odl_indices_raw)}

        intersection = q_indices.intersection(odl_indices)
        if intersection:
            raise ValueError(f"Indices {intersection} specified for both Q-function and ODL strategies.")

        blocks = []

        default_enum = (
            MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD
            if self.default_strategy_name == "odl"
            else MaximizationStrategy.QFUNCTION
        )

        for i in range(n_components):
            if i in q_indices:
                strategy = MaximizationStrategy.QFUNCTION
            elif i in odl_indices:
                strategy = MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD
            else:
                strategy = default_enum

            # No parameter names: every free parameter of a component is optimized, and the parameters
            # fixed with ``family.view(...)`` keep their values.
            blocks.append(OptimizationBlock(component_id=i, params_to_optimize=None, maximization_strategy=strategy))

        pipeline: Pipeline = Pipeline(
            [ExpectationStep(), MaximizationStep(blocks, self.optimizer)],
            self.breakpointers,
            self.pruners,
            once_in_iterations,
        )

        result = pipeline.fit(X, mixture)
        self._history = pipeline.history
        return result
