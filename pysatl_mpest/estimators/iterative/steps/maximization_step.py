"""Maximization step for iterative estimation pipelines."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import bisect
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import ClassVar

import numpy as np
from pysatl_core.families import ParametricFamilyDistribution

from ....core import _components
from ....optimizers import Optimizer
from .._strategies import StrategyFunction, moments_strategy, observed_data_likelihood_strategy, q_function_strategy
from ..pipeline_state import PipelineState
from ..pipeline_step import PipelineStep
from .block import MaximizationStrategy, OptimizationBlock


class MaximizationStep(PipelineStep):
    """Maximization step updating component parameters and mixture weights.

    Executes configured :class:`OptimizationBlock` tasks to update component parameters,
    then updates mixture weights from responsibilities.

    Parameters
    ----------
    blocks : Sequence[OptimizationBlock]
        Optimization tasks specifying components, parameters, and strategies.
    optimizer : Optimizer
        Numerical optimizer for strategies requiring numerical search.

    Attributes
    ----------
    blocks : list[OptimizationBlock]
        Configured optimization tasks.
    optimizer : Optimizer
        Numerical optimizer instance.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        run
    """

    _strategies: ClassVar[Mapping[MaximizationStrategy, StrategyFunction]] = MappingProxyType(
        {
            MaximizationStrategy.QFUNCTION: q_function_strategy,
            MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD: observed_data_likelihood_strategy,
            MaximizationStrategy.MOMENTS: moments_strategy,
        }
    )

    def __init__(self, blocks: Sequence[OptimizationBlock], optimizer: Optimizer):
        self.blocks = list(blocks)
        self.optimizer = optimizer

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        """list[type[PipelineStep]]: Steps permitted to follow MaximizationStep."""

        from pysatl_mpest.estimators.iterative.steps.expectation_step import ExpectationStep

        return [ExpectationStep]

    def _update_components_params(
        self, component: ParametricFamilyDistribution, params: dict[str, float]
    ) -> ParametricFamilyDistribution:
        return _components.with_parameters(component, **params)

    def run(self, state: PipelineState) -> PipelineState:
        """Execute the M-step, updating component parameters and weights.

        Strategy exceptions propagate directly. Parameters and weights update only
        after all strategy blocks succeed. Missing responsibilities set ``state.error``.

        Parameters
        ----------
        state : PipelineState
            Current state containing responsibility matrix H and current mixture.

        Returns
        -------
        PipelineState
            Updated state with modified mixture model.

        Raises
        ------
        ValueError
            If a block requests an unknown or fixed parameter.
        IndexError
            If a block's component index is out of bounds.
        NotImplementedError
            If a strategy is not implemented for the component's family.
        """

        if state.H is None:
            error = ValueError("Responsibility matrix H is not computed.")
            state.error = error
            return state

        results: list[tuple[int, dict[str, float]]] = []
        curr_mixture = state.curr_mixture

        for block in self.blocks:
            strategy = self._strategies[block.maximization_strategy]
            component_id, new_params = strategy(curr_mixture[block.component_id], state, block, self.optimizer)
            results.append((component_id, new_params))

        for component_id, params in results:
            curr_mixture[component_id] = self._update_components_params(curr_mixture[component_id], params)

        responsibilities_sum = np.sum(state.H, axis=0)
        new_weights = responsibilities_sum / state.X.shape[0]
        curr_mixture.log_weights = np.log(new_weights + 1e-30)

        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        """Updates optimization blocks after component pruning.

        Removes blocks associated with pruned components and shifts component_id
        for remaining blocks by the number of removed components before them.
        Surviving blocks are replaced with updated copies, so the block objects
        given to the constructor are not modified.

        Parameters
        ----------
        removed_components_indices : list[int]
            Indices of components that were removed.
        """
        if not removed_components_indices or not self.blocks:
            return

        removed = sorted(set(removed_components_indices))
        self.blocks = [
            replace(block, component_id=block.component_id - bisect.bisect_left(removed, block.component_id))
            for block in self.blocks
            if block.component_id not in removed
        ]
