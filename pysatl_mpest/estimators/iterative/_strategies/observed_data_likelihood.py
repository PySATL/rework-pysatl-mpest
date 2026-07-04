"""Provides strategy for the Maximization-step based on the observed data likelihood.

This module implements the logic for updating component parameters by directly
maximizing the log-likelihood of the observed data.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np

from ....distributions import ContinuousDistribution
from ....optimizers import Optimizer
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock


def observed_data_likelihood_strategy(
    component: ContinuousDistribution,
    state: PipelineState,
    block: OptimizationBlock,
    optimizer: Optimizer,
) -> tuple[int, dict[str, float]]:
    """Generic strategy that calculates optimized parameters by maximizing observed data log-likelihood.

    This function calculates the new parameters for a component by directly
    maximizing the log-likelihood of the mixture model:
        L(θ_q) = Σ ln( C_i + w_q * f_q(x_i | θ_q) )

    Where C_i is the fixed contribution of all other components (background)
    calculated based on the current state.

    Parameters
    ----------
    component : ContinuousDistribution
        The distribution component type/instance used for dispatch and parameter metadata.
    state : PipelineState
        The current state containing data X and current mixture parameters.
        (Note: Responsibilities H are not used in this strategy).
    block : OptimizationBlock
        Configuration defining which parameters to optimize (component_id and param names).
    optimizer : Optimizer
        Numerical optimizer instance.

    Returns
    -------
    tuple[int, dict[str, float]]
        Component ID and a dictionary of the optimized parameters.
    """

    X = state.X
    n_samples = X.shape[0]
    tol = np.finfo(np.float64).tiny

    component_id = block.component_id
    params_to_optimize = sorted(list(block.params_to_optimize.intersection(component.params_to_optimize)))

    weights = state.curr_mixture.weights
    target_weight = weights[component_id]

    background_term = np.zeros(n_samples, dtype=np.float64)
    for i, comp in enumerate(state.curr_mixture.components):
        if i != component_id:
            background_term += weights[i] * comp.pdf(X)

    def target(vector_params):
        temp_comp = component.clone_with_params(params_to_optimize, vector_params)
        mixture_pdf = np.maximum(background_term + target_weight * temp_comp.pdf(X), tol)
        return -np.sum(np.log(mixture_pdf))

    new_params = optimizer.minimize(target, component.get_params_vector(params_to_optimize))
    return component_id, dict(zip(params_to_optimize, new_params))
