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

from ....distributions import ContinuousDistribution, Exponential, Normal, Weibull
from ....optimizers import Optimizer
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock

NUMERICAL_TOLERANCE = 1e-9


# ------------------------
# Base Q-function strategy
# ------------------------


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

    params_to_optimize = sorted(list(component.params_to_optimize.intersection(block.params_to_optimize)))
    temp_comp = deepcopy(component)

    def target(vector_params):
        temp_comp.set_params_from_vector(params_to_optimize, vector_params)
        lpdf_values = temp_comp.lpdf(X)
        safe_lpdf = np.where(H_j == 0, 0.0, lpdf_values)
        return -np.dot(H_j, safe_lpdf).item()

    new_params = optimizer.minimize(target, temp_comp.get_params_vector(params_to_optimize))
    return component_id, dict(zip(params_to_optimize, new_params))


# ---------------------------------
# Exponential distribution strategy
# ---------------------------------


@q_function_strategy.register(Exponential)
def _(
    component: Exponential, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Specialized Q-function parameter estimation strategy for
    the Exponential distribution using an analytical solution.

    This function provides a closed-form update for the parameters of an
    `Exponential` distribution, which is more efficient and precise than
    general-purpose numerical optimization. It calculates the new `loc` and
    `rate` parameters directly from the data and responsibilities.

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

    # If the component has negligible responsibility, do not update its parameters.
    if N_j < NUMERICAL_TOLERANCE:
        return block.component_id, {}

    # Update location (loc) if it's in the optimization block
    if Exponential.PARAM_LOC in params_to_optimize:
        relevant_X = X[H_j > NUMERICAL_TOLERANCE]
        if relevant_X.size > 0:
            new_params[Exponential.PARAM_LOC] = np.min(relevant_X).item()
        else:
            new_params[Exponential.PARAM_LOC] = component.loc

    # Update lambda (rate) if it's in the optimization block
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


# ----------------------------
# Normal distribution strategy
# ----------------------------


@q_function_strategy.register(Normal)
def _(
    component: Normal, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Specialized Q-function parameter estimation strategy for
    the normal distribution using an analytical solution.

    This function provides a closed-form update for the parameters of a
    `Normal` distribution, which is more efficient than numerical optimization.
    It calculates the new `loc` (mean) and `scale` (standard deviation)
    parameters directly from the data and responsibilities.

     Notes
    -----
    The analytical updates are as follows:
    - The new mean `loc` is the weighted average of the data `X`.
    - The new variance (`scale` squared) is the weighted average of the
      squared differences from the new mean.
    The weights are the responsibilities from the matrix `H`.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = component.params_to_optimize.intersection(block.params_to_optimize)
    new_params = {}

    N_j = np.sum(H_j).item()

    # If the component has negligible responsibility, do not update its parameters.
    if N_j < NUMERICAL_TOLERANCE:
        return block.component_id, {}

    # Update mean (loc) if it's in the optimization block
    if Normal.PARAM_LOC in params_to_optimize:
        weighted_sum_X = np.dot(H_j, X).item()
        new_mu = weighted_sum_X / N_j
        new_params[Normal.PARAM_LOC] = new_mu

    # Update std (scale) if it's in the optimization block
    if Normal.PARAM_SCALE in params_to_optimize:
        # Use the newly computed mean if available, otherwise use the existing one.
        mu = new_params.get(Normal.PARAM_LOC, component.loc)

        # Calculate the weighted variance
        weighted_sum_sq_diff = np.dot(H_j, (X - mu) ** 2).item()
        new_variance = weighted_sum_sq_diff / N_j

        if new_variance > NUMERICAL_TOLERANCE:
            new_params[Normal.PARAM_SCALE] = np.sqrt(new_variance)
        else:
            # If variance is too small, it can lead to instability.
            # Keep the old scale to prevent it from collapsing to zero.
            new_params[Normal.PARAM_SCALE] = component.scale

    return block.component_id, new_params


# ----------------------------
# Weibull distribution strategy
# ----------------------------


@q_function_strategy.register(Weibull)
def _(
    component: Weibull, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Specialized Q-function parameter estimation strategy for
    the Weibull distribution using an analytical solution.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = component.params_to_optimize.intersection(block.params_to_optimize)
    new_params = {}

    N_j = np.sum(H_j).item()

    # If the component has negligible responsibility, do not update its parameters.
    if N_j < NUMERICAL_TOLERANCE:
        return block.component_id, {}

    # ------------------------
    # Optimizing loc and shape
    # ------------------------

    params_for_numerical_opt = {Weibull.PARAM_SHAPE, Weibull.PARAM_LOC}.intersection(params_to_optimize)

    if params_for_numerical_opt:
        reduced_block = OptimizationBlock(block.component_id, params_for_numerical_opt, block.maximization_strategy)

        # The generic strategy will handle the optimization for the reduced set
        _, numerically_optimized_params = q_function_strategy.dispatch(ContinuousDistribution)(
            component, state, reduced_block, optimizer
        )
        new_params.update(numerically_optimized_params)

    # -----------------------------
    # Analytical solution for scale
    # -----------------------------

    final_shape = new_params.get(Weibull.PARAM_SHAPE, component.shape)
    final_loc = new_params.get(Weibull.PARAM_LOC, component.loc)

    if Weibull.PARAM_SCALE in params_to_optimize:
        # Ensure that data points are greater than the location parameter
        if np.any(final_loc >= X):
            new_params[Weibull.PARAM_SCALE] = component.scale
        else:
            X_minus_loc = X - final_loc

            # Use np.maximum to avoid taking powers of negative numbers if final_loc is slightly off
            safe_X_minus_loc = np.maximum(X_minus_loc, NUMERICAL_TOLERANCE)

            weighted_sum = np.dot(H_j, safe_X_minus_loc**final_shape)

            if weighted_sum > 0:
                new_scale = (weighted_sum / N_j) ** (1.0 / final_shape)
                new_params[Weibull.PARAM_SCALE] = new_scale
            else:
                new_params[Weibull.PARAM_SCALE] = component.scale

    return block.component_id, new_params
