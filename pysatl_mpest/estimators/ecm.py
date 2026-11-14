"""Expectation-Conditional Maximization (ECM) estimator."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Mapping, Sequence

from numpy.typing import ArrayLike

from ..core import MixtureModel
from ..optimizers import Optimizer
from ._parameter_groups import component_block_chain, validate_parameter_groups
from .base_estimator import BaseEstimator
from .iterative import (
    Breakpointer,
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    Pipeline,
    Pruner,
    WeightsStep,
)
from .iterative._iteration_history import IterationsHistory


class ECM(BaseEstimator):
    """Expectation-Conditional Maximization (ECM) estimator.

    Maximizes the Q-function over ordered parameter groups supplied to :meth:`fit`.
    Without groups, jointly optimizes each component's free parameters, as in EM.

    Parameters
    ----------
    breakpointers : Sequence[Breakpointer]
        Stopping criteria for the iterative pipeline.
    pruners : Sequence[Pruner]
        Component pruning strategies applied during iterations.
    optimizer : Optimizer
        Numerical optimizer used in maximization steps.

    Attributes
    ----------
    breakpointers : list[Breakpointer]
        Configured stopping criteria.
    pruners : list[Pruner]
        Configured component pruning strategies.
    optimizer : Optimizer
        Numerical optimizer instance.
    history : IterationsHistory
        Recorded iteration snapshots, accessible after calling :meth:`fit`.

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
    ) -> None:
        self.breakpointers = list(breakpointers)
        self.pruners = list(pruners)
        self.optimizer = optimizer
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

    def fit(
        self,
        X: ArrayLike,
        mixture: MixtureModel,
        once_in_iterations: int = 1,
        *,
        parameter_groups: Mapping[int, Sequence[set[str]]] | None = None,
    ) -> MixtureModel:
        """Fit mixture model parameters using the ECM algorithm.

        Parameters
        ----------
        X : ArrayLike
            Input data sample.
        mixture : MixtureModel
            Initial mixture model to fit; not modified in-place.
        once_in_iterations : int, optional
            Iteration recording frequency in history. Defaults to 1.
        parameter_groups : Mapping[int, Sequence[set[str]]] | None, optional
            Component indices mapped to ordered, non-empty, disjoint groups covering
            all free parameters. Groups are optimized sequentially; omitted components
            use one joint step. Defaults to None.

        Returns
        -------
        MixtureModel
            Fitted mixture model with estimated parameters.

        Raises
        ------
        ValueError
            If a component index is invalid or its groups do not partition its
            free parameters into non-empty, disjoint sets.
        """

        validate_parameter_groups(mixture, parameter_groups)

        # With ``params_to_optimize=None`` every free parameter of a component is optimized, and
        # the parameters fixed with ``family.view(...)`` keep their values.
        blocks = [
            block
            for i in range(mixture.n_components)
            for block in component_block_chain(
                i,
                None if parameter_groups is None else parameter_groups.get(i),
                MaximizationStrategy.QFUNCTION,
            )
        ]

        pipeline: Pipeline = Pipeline(
            [ExpectationStep(), WeightsStep(), MaximizationStep(blocks, self.optimizer)],
            self.breakpointers,
            self.pruners,
            once_in_iterations,
        )
        result = pipeline.fit(X, mixture)
        self._history = pipeline.history
        return result
