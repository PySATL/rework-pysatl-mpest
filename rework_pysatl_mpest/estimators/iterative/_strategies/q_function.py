"""Provides strategies for the Maximization-step based on the Q-function.

This module implements the logic for updating component parameters by
maximizing the Q-function (the expected complete-data log-likelihood).
It uses `functools.singledispatch` to provide a generic, optimization-based
approach for any continuous distribution, as well as specialized, more
efficient analytical solutions for specific distribution types like the
`Exponential` distribution.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from copy import deepcopy
from functools import singledispatch

import numpy as np

from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.distributions.exponential import Exponential
from rework_pysatl_mpest.estimators.iterative.pipeline_state import PipelineState
from rework_pysatl_mpest.estimators.iterative.steps.block import OptimizationBlock
from rework_pysatl_mpest.optimizers.optimizer import Optimizer

NUMERICAL_TOLERANCE = 1e-9


@singledispatch
def q_function_strategy(
    component: ContinuousDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Generic M-step strategy that maximizes the Q-function numerically.

    This function serves as the default implementation for updating a
    component's parameters. It defines the negative Q-function as the target
    for a numerical optimizer and finds the parameter values that minimize it
    (and thus maximize the Q-function).

    This function is a dispatcher; specific, more efficient implementations can
    be registered for different distribution types.

    Parameters
    ----------
    component : ContinuousDistribution
        The distribution component whose parameters are to be optimized.
    state : PipelineState
        The current state of the pipeline, containing the data :attr:`X` and the
        responsibility matrix :attr:`H`.
    block : OptimizationBlock
        The configuration block defining which component and which of its
        parameters to optimize.
    optimizer : Optimizer
        The numerical optimizer instance used to perform the maximization.

    Returns
    -------
    tuple[int, dict[str, float]]
        A tuple containing the component's ID and a dictionary of the
        optimized parameter names and their new values.

    Raises
    ------
    ValueError
        If the responsibility matrix :attr:`H` has not been computed and set in
        the :attr:`state` object.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X, H_j = state.X, state.H[:, block.component_id]
    component_id = block.component_id

    params_to_optimize = list(component.params_to_optimize.intersection(block.params_to_optimize))
    temp_comp = deepcopy(component)

    def target(vector_params):
        temp_comp.set_params_from_vector(params_to_optimize, vector_params)
        return -temp_comp.q_function(X, H_j)

    new_params = optimizer.minimize(target, temp_comp.get_params_vector(params_to_optimize))
    return component_id, dict(zip(params_to_optimize, new_params))


@q_function_strategy.register(Exponential)
def _(
    component: Exponential, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Specialized M-step for the Exponential distribution using an analytical solution.

    This function provides a closed-form update for the parameters of an
    `Exponential` distribution, which is more efficient and precise than
    general-purpose numerical optimization. It calculates the new `loc` and
    `rate` parameters directly from the data and responsibilities.

    Parameters
    ----------
    component : Exponential
        The exponential distribution component to be optimized.
    state : PipelineState
        The current state of the pipeline, containing `X` and `H`.
    block : OptimizationBlock
        The configuration for the optimization task.
    optimizer : Optimizer
        This parameter is ignored by this specialized implementation.

    Returns
    -------
    tuple[int, dict[str, float]]
        A tuple containing the component ID and a dictionary of the
        analytically updated parameters. If the total responsibility for the
        component is negligible, an empty dictionary is returned, indicating
        no update was performed.

    Raises
    ------
    ValueError
        If the responsibility matrix `H` is not available in the `state`.

    Notes
    -----
    The analytical updates are as follows:
    - The new location `loc` is the minimum value in the dataset `X` among
      points with a non-negligible responsibility for this component.
    - The new rate `lambda` (or `rate`) is the reciprocal of the
      weighted average of `(X - loc)`.

    This implementation ignores the `optimizer` parameter as it does not
    require numerical optimization.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = component.params_to_optimize.intersection(block.params_to_optimize)
    new_params = {}

    N_j = np.sum(H_j).item()

    if N_j < NUMERICAL_TOLERANCE:
        return block.component_id, {}

    if Exponential.PARAM_LOC in params_to_optimize:
        relevant_X = X[H_j > NUMERICAL_TOLERANCE]
        if relevant_X.size > 0:
            new_params[Exponential.PARAM_LOC] = np.min(relevant_X).item()
        else:
            new_params[Exponential.PARAM_LOC] = component.loc

    if Exponential.PARAM_RATE in params_to_optimize:
        loc = new_params.get(Exponential.PARAM_LOC, component.loc)

        weighted_sum_X = np.dot(H_j, X).item()

        denominator = weighted_sum_X / N_j - loc
        if denominator > NUMERICAL_TOLERANCE:
            new_params[Exponential.PARAM_RATE] = 1.0 / denominator
        else:
            # If the weighted average is too close to loc,
            # leave rate unchanged to avoid infinity.
            new_params[Exponential.PARAM_RATE] = component.rate

    return block.component_id, new_params
