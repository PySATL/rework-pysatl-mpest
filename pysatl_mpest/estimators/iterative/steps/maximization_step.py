"""Provides the Maximization-step for an iterative estimation pipeline.

This module defines the `MaximizationStep` class, a concrete implementation of
:class:`~pysatl_mpest.estimators.iterative.pipeline_step.PipelineStep`.
This step is responsible for performing the Maximization (M-step) in an
Expectation-Maximization (EM) like algorithm. It updates the parameters of the
mixture model components and their weights to maximize the expected
log-likelihood, using the responsibilities computed in a preceding
Expectation-step.
"""

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
    """A pipeline step that performs the Maximization (M-step).

    This step updates the parameters of each component in the mixture model,
    as well as the mixture weights, based on the responsibility matrix :attr:`H`
    calculated in the Expectation-step. The update process is configured
    through a sequence of :class:`OptimizationBlock` objects, each defining
    a specific optimization task.

    Parameters
    ----------
    blocks : Sequence[OptimizationBlock]
        A sequence of configuration blocks that define the optimization tasks.
        Each block specifies a component, its parameters to optimize, and the
        maximization strategy to use.
    optimizer : Optimizer
        A numerical optimizer instance used to find the optimal parameters
        when an analytical solution is not available for a given strategy.

    Attributes
    ----------
    blocks : list[OptimizationBlock]
        The list of optimization tasks to be performed.
    optimizer : Optimizer
        The numerical optimizer used for parameter estimation.

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
        """list[type[PipelineStep]]: Defines the valid subsequent steps.

        Specifies that a :class:`MaximizationStep` should typically be
        followed by an :class:`ExpectationStep` to complete one iteration of
        the EM algorithm.
        """

        from pysatl_mpest.estimators.iterative.steps.expectation_step import ExpectationStep

        return [ExpectationStep]

    def _update_components_params(
        self, component: ParametricFamilyDistribution, params: dict[str, float]
    ) -> ParametricFamilyDistribution:
        """Helper method to update parameters for a single component.

        Parameters
        ----------
        component : ParametricFamilyDistribution
            The component whose parameters are to be updated.
        params : dict[str, float]
            A dictionary mapping parameter names to their new optimized values.

        Returns
        -------
        ParametricFamilyDistribution
            A new component with the updated parameters.
        """

        return _components.with_parameters(component, **params)

    def run(self, state: PipelineState) -> PipelineState:
        """Executes the M-step.

        This method iterates through the configured optimization blocks,
        updates the parameters for each specified component using the
        appropriate strategy, and then recalculates the mixture weights based
        on the sum of responsibilities.

        Parameters
        ----------
        state : PipelineState
            The current state of the pipeline. Must contain the responsibility
            matrix :attr:`H` and the mixture model :attr:`curr_mixture`.

        Returns
        -------
        PipelineState
            The updated pipeline state with the modified :attr:`curr_mixture`. If the
            :attr:`H` matrix is not available in the input state, the state is
            returned with an error set, and no modifications are made.

        Raises
        ------
        ValueError
            If a block requests an unknown parameter or a parameter fixed by a
            family view.
        IndexError
            If a block's component index is at least the number of mixture components.
        NotImplementedError
            If the selected strategy is not implemented for the component's family
            and parametrization.

        Notes
        -----
        Strategy exceptions propagate directly to the caller. Component parameters
        and weights are updated only after every strategy has returned successfully.
        Missing responsibilities instead set :attr:`PipelineState.error`, which
        :meth:`Pipeline.fit` reports as a ``RuntimeWarning``.
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
