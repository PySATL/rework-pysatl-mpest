"""Provides strategy for the Maximization-step based on the observed data likelihood.

This module implements the logic for updating component parameters by directly
maximizing the log-likelihood of the observed data.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from copy import copy

import numpy as np

from rework_pysatl_mpest.estimators.iterative._strategies.utils import handle_numerical_overflow

from ....distributions import ContinuousDistribution
from ....optimizers import Optimizer
from ....typings import DType
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock

NUMERICAL_TOLERANCE = 1e-9


def observed_data_likelihood_strategy(
    component: ContinuousDistribution[DType],
    state: PipelineState[DType],
    block: OptimizationBlock,
    optimizer: Optimizer[DType],
) -> tuple[int, dict[str, DType]]:
    """Generic strategy that calculates optimized parameters by maximizing observed data log-likelihood.

    This function calculates the new parameters for a component by directly
    maximizing the log-likelihood of the mixture model:
        L(θ_q) = Σ ln( C_i + w_q * f_q(x_i | θ_q) )

    Where C_i is the fixed contribution of all other components (background)
    calculated based on the current state.

    Parameters
    ----------
    component : ContinuousDistribution[DType]
        The distribution component type/instance used for dispatch and parameter metadata.
    state : PipelineState[DType]
        The current state containing data X and current mixture parameters.
        (Note: Responsibilities H are not used in this strategy).
    block : OptimizationBlock
        Configuration defining which parameters to optimize (component_id and param names).
    optimizer : Optimizer[DType]
        Numerical optimizer instance.

    Returns
    -------
    tuple[int, dict[str, DType]]
        Component ID and a dictionary of the optimized parameters.
    """

    X = state.X
    n_samples = X.shape[0]
    dtype = component.dtype
    tol = np.finfo(dtype).tiny

    component_id = block.component_id
    params_to_optimize = sorted(list(block.params_to_optimize.intersection(component.params_to_optimize)))

    temp_mixture = copy(state.curr_mixture)
    target_comp = temp_mixture.components[component_id]
    background_term = np.zeros(n_samples, dtype=dtype)

    for i, comp in enumerate(state.curr_mixture.components):
        if i != component_id:
            background_term += temp_mixture.weights[i] * comp.pdf(X)

    def target(vector_params):
        target_comp.set_params_from_vector(params_to_optimize, vector_params)
        comp_pdf = target_comp.pdf(X)
        mixture_pdf = background_term + (temp_mixture.weights[component_id] * comp_pdf)
        mixture_pdf = np.maximum(mixture_pdf, tol)
        res = -np.sum(np.log(mixture_pdf))

        if np.isinf(res):
            handle_numerical_overflow(state, "Observed data likelihood optimization")

        return res

    new_params = optimizer.minimize(target, target_comp.get_params_vector(params_to_optimize))
    return component_id, dict(zip(params_to_optimize, new_params))
