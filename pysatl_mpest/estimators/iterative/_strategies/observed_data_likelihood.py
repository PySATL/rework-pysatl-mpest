"""Provides strategy for the Maximization-step based on the observed data likelihood.

This module implements the logic for updating component parameters by directly
maximizing the log-likelihood of the observed data.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from pysatl_core.families import ParametricFamilyDistribution
from scipy.special import logsumexp

from ....core import _components
from ....optimizers import Optimizer
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock


def observed_data_likelihood_strategy(
    component: ParametricFamilyDistribution,
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

    Everything is computed with log-densities, so the likelihood stays accurate where the
    densities themselves underflow to zero. The optimizer may try values that violate the
    constraints of the parametrization (for example a negative standard deviation), and the
    log-likelihood is zero where a point has zero density under every component. The target
    is ``+inf`` in both cases, so such values are never chosen.

    Only the parameters named by the block are optimized; the other parameters of the
    component, including those fixed with ``family.view(...)``, keep their values.

    Parameters
    ----------
    component : ParametricFamilyDistribution
        The distribution component whose parameters are to be optimized.
    state : PipelineState
        The current state containing data X and the current mixture, whose weights and other
        components define the background.
        (Note: Responsibilities H are not used in this strategy).
    block : OptimizationBlock
        Configuration defining which parameters to optimize (component_id and param names).
    optimizer : Optimizer
        Numerical optimizer instance.

    Returns
    -------
    tuple[int, dict[str, float]]
        Component ID and a dictionary of the optimized parameters.

    Raises
    ------
    ValueError
        If the block names parameters that are not free parameters of the component.
    """

    X = state.X
    component_id = block.component_id

    params_to_optimize = block.params_to_update(component)
    if not params_to_optimize:
        return component_id, {}

    mixture = state.curr_mixture
    log_weights = mixture.log_weights

    # Log of the contribution of the other components: the target component is masked out.
    joint = mixture.component_lpdf_matrix(X) + log_weights
    joint[:, component_id] = -np.inf
    log_background = logsumexp(joint, axis=1)

    log_target_weight = log_weights[component_id]
    current = _components.free_parameters(component)

    def target(vector_params: list[float]) -> float:
        try:
            candidate = _components.with_parameters(component, **dict(zip(params_to_optimize, vector_params)))
        except ValueError:
            return np.inf
        log_mixture = np.logaddexp(log_background, log_target_weight + _components.lpdf(candidate, X))
        return float(-np.sum(log_mixture))

    new_params = optimizer.minimize(target, [current[name] for name in params_to_optimize])
    return component_id, {name: float(value) for name, value in zip(params_to_optimize, new_params)}
