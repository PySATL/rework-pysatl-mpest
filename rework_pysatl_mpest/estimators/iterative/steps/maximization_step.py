"""Provides the Maximization-step for an iterative estimation pipeline.

This module defines the `MaximizationStep` class, a concrete implementation of
:class:`~rework_pysatl_mpest.estimators.iterative.pipeline_step.PipelineStep`.
This step is responsible for performing the Maximization (M-step) in an
Expectation-Maximization (EM) like algorithm. It updates the parameters of the
mixture model components and their weights to maximize the expected
log-likelihood, using the responsibilities computed in a preceding
Expectation-step.
"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Callable, ClassVar

import numpy as np

from ....distributions import ContinuousDistribution
from ....optimizers import Optimizer
from ....typings import DType
from .._strategies import q_function_strategy
from ..pipeline_state import PipelineState
from ..pipeline_step import PipelineStep
from .block import MaximizationStrategy, OptimizationBlock


class MaximizationStep(PipelineStep[DType]):
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

    _strategies: ClassVar[Mapping[MaximizationStrategy, Callable]] = MappingProxyType(
        {MaximizationStrategy.QFUNCTION: q_function_strategy}
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

        from rework_pysatl_mpest.estimators.iterative.steps.expectation_step import ExpectationStep

        return [ExpectationStep]

    def _update_components_params(self, component: ContinuousDistribution, params: dict[str, DType]):
        """Helper method to update parameters for a single component.

        Parameters
        ----------
        component : ContinuousDistribution
            The component whose parameters are to be updated.
        params : dict[str, DType]
            A dictionary mapping parameter names to their new optimized values.
        """

        param_names = list(params.keys())
        param_values = list(params.values())
        component.set_params_from_vector(param_names, param_values)

    def run(self, state: PipelineState[DType]) -> PipelineState[DType]:
        """Executes the M-step.

        This method iterates through the configured optimization blocks,
        updates the parameters for each specified component using the
        appropriate strategy, and then recalculates the mixture weights based
        on the sum of responsibilities.

        Parameters
        ----------
        state : PipelineState[DType]
            The current state of the pipeline. Must contain the responsibility
            matrix :attr:`H` and the mixture model :attr:`curr_mixture`.

        Returns
        -------
        PipelineState[DType]
            The updated pipeline state with the modified :attr:`curr_mixture`. If the
            :attr:`H` matrix is not available in the input state, the state is
            returned with an error set, and no modifications are made.
        """

        if state.H is None:
            error = ValueError("Responsibility matrix H is not computed.")
            state.error = error
            return state

        results = []
        curr_mixture = state.curr_mixture

        dtype = curr_mixture.dtype

        for block in self.blocks:
            strategy = self._strategies[block.maximization_strategy]
            component_id, new_params = strategy(curr_mixture[block.component_id], state, block, self.optimizer)
            if state.error:
                return state
            results.append((component_id, new_params))

        for result in results:
            component_id, params = result
            self._update_components_params(curr_mixture[component_id], params)

        responsibilities_sum = np.sum(state.H, axis=0)
        new_weights = responsibilities_sum / state.X.shape[0]
        curr_mixture.log_weights = np.log(new_weights + dtype(1e-30))

        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        """Updates optimization blocks after component pruning.

        This method removes optimization blocks associated with pruned components
        and updates the component_id in the remaining blocks to maintain consistency.

        Parameters
        ----------
        state : PipelineState
            The current pipeline state containing removed_components_indices
        removed_components_indices : list[int]
            Tracks which component indices were removed during pruning.
        """
        if len(removed_components_indices) == 0 or self.blocks is None:
            return
        removed_indices = set(removed_components_indices)

        self.blocks = [block for block in self.blocks if block.component_id not in removed_indices]

        old_to_new_mapping = {}
        new_component_id = 0
        for old_component_id in range(len(self.blocks) + len(removed_indices)):
            if old_component_id not in removed_indices:
                old_to_new_mapping[old_component_id] = new_component_id
                new_component_id += 1

        for block in self.blocks:
            block.component_id = old_to_new_mapping[block.component_id]
