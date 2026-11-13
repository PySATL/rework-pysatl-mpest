"""Expectation-Conditional Maximization Either (ECME) estimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Mapping, Sequence
from typing import Literal

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
    OptimizationBlock,
    Pipeline,
    Pruner,
    WeightsStep,
)
from .iterative._iteration_history import IterationsHistory


class ECME(BaseEstimator):
    """Expectation-Conditional Maximization Either (ECME) estimator.

    Extends ECM by allowing components to be optimized by either maximizing the
    Q-function or directly maximizing the Observed Data Likelihood (ODL).
    All free parameters are estimated, preserving fixed view parameters.

    The maximization blocks are ordered so that the components updated through
    the Q-function go first (in ascending component order), followed by the
    components updated through ODL (also ascending). Together with
    :class:`WeightsStep` and the sequential semantics of :class:`MaximizationStep`,
    this makes the observed-data log-likelihood non-decreasing across iterations
    when the expectation step is soft, no pruners remove components, and the
    optimizer never returns a point with a worse objective value than its
    starting point. :class:`~pysatl_mpest.optimizers.ScipyNelderMead` satisfies
    the optimizer condition; the base :class:`~pysatl_mpest.optimizers.Optimizer`
    contract does not promise it. The guarantee applies only to the block order
    built by this facade: a :class:`MaximizationStep` given an arbitrary block
    list executes it as given.

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

    def _validate_indices(self, indices: set[int], n_components: int) -> None:
        for idx in indices:
            if idx < 0:
                raise ValueError(f"Component index {idx} is negative; component indices must be non-negative.")
            if idx >= n_components:
                raise ValueError(f"Component index {idx} is out of bounds for mixture size {n_components}.")

    def fit(
        self,
        X: ArrayLike,
        mixture: MixtureModel,
        q_indices_raw: Sequence[int] | int | None = None,
        odl_indices_raw: Sequence[int] | int | None = None,
        once_in_iterations: int = 1,
        *,
        parameter_groups: Mapping[int, Sequence[set[str]]] | None = None,
    ) -> MixtureModel:
        """Fit mixture model parameters using the ECME algorithm.

        Parameters
        ----------
        X : ArrayLike
            Input data sample.
        mixture : MixtureModel
            Initial mixture model to fit; not modified in-place.
        q_indices_raw : Sequence[int] | int | None, optional
            Component indices to optimize via Q-function maximization. Must be non-negative.
        odl_indices_raw : Sequence[int] | int | None, optional
            Component indices to optimize via ODL maximization. Must be non-negative.
        once_in_iterations : int, optional
            Iteration recording frequency in history. Defaults to 1.
        parameter_groups : Mapping[int, Sequence[set[str]]] | None, optional
            Ordered partition of components' free parameters into conditional
            maximization steps. Keys are component indices; each value orders
            disjoint non-empty parameter groups whose union is exactly the
            component's free parameters. Every group becomes a separate
            maximization step computed at the parameter values produced by the
            previous steps of the same component, under the strategy assigned
            to that component. Components without an entry get a single step
            over all free parameters, which is also the behavior when the
            argument is omitted. Defaults to None.

        Returns
        -------
        MixtureModel
            Fitted mixture model with estimated parameters.

        Raises
        ------
        ValueError
            If a component index is negative, out of bounds, or present in both
            strategy index sets; or if a ``parameter_groups`` key is negative or
            out of bounds, a component's group list is empty, a group is empty,
            groups overlap, or the union of a component's groups differs from
            its free parameters (unknown, fixed, or uncovered parameter names).
        """

        n_components = mixture.n_components

        q_indices = self._normalize_indices(q_indices_raw)
        odl_indices = self._normalize_indices(odl_indices_raw)
        self._validate_indices(q_indices, n_components)
        self._validate_indices(odl_indices, n_components)

        intersection = q_indices.intersection(odl_indices)
        if intersection:
            raise ValueError(f"Indices {intersection} specified for both Q-function and ODL strategies.")

        validate_parameter_groups(mixture, parameter_groups)

        default_enum = (
            MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD
            if self.default_strategy_name == "odl"
            else MaximizationStrategy.QFUNCTION
        )

        def chain(component_id: int, strategy: MaximizationStrategy) -> list[OptimizationBlock]:
            groups = None if parameter_groups is None else parameter_groups.get(component_id)
            return component_block_chain(component_id, groups, strategy)

        # With ``params_to_optimize=None`` every free parameter of a component is optimized, and
        # the parameters fixed with ``family.view(...)`` keep their values.
        q_chains = [
            block
            for i in range(n_components)
            if i in q_indices or (i not in odl_indices and default_enum is MaximizationStrategy.QFUNCTION)
            for block in chain(i, MaximizationStrategy.QFUNCTION)
        ]
        odl_chains = [
            block
            for i in range(n_components)
            if i in odl_indices
            or (i not in q_indices and default_enum is MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD)
            for block in chain(i, MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD)
        ]

        pipeline: Pipeline = Pipeline(
            [ExpectationStep(), WeightsStep(), MaximizationStep([*q_chains, *odl_chains], self.optimizer)],
            self.breakpointers,
            self.pruners,
            once_in_iterations,
        )

        result = pipeline.fit(X, mixture)
        self._history = pipeline.history
        return result
