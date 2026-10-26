"""Provides strategies for the Maximization-step based on the Q-function.

This module implements the logic for updating component parameters by
maximizing the Q-function (the expected complete-data log-likelihood).
It dispatches on the family and parametrization of the component to provide a generic,
optimization-based approach for any component, as well as specialized, more efficient
analytical solutions for specific ones: the `Normal` family with the ``meanStd``
parametrization and the `Exponential` family with the ``rate`` parametrization.
"""

__author__ = "Danil Totmyanin, Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName

from ....core import _components
from ....optimizers import Optimizer
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock
from ._dispatch import family_dispatch

NUMERICAL_TOLERANCE = 1e-9


# ------------------------
# Base Q-function strategy
# ------------------------


@family_dispatch
def q_function_strategy(
    component: ParametricFamilyDistribution,
    state: PipelineState,
    block: OptimizationBlock,
    optimizer: Optimizer,
) -> tuple[int, dict[str, float]]:
    """Generic M-step strategy that maximizes the Q-function numerically.

    This function serves as the default implementation for updating a
    component's parameters. It defines the negative Q-function as the target
    for a numerical optimizer and finds the parameter values that minimize it
    (and thus maximize the Q-function).

    Only the parameters named by the block are optimized; the other parameters of
    the component, including those fixed with ``family.view(...)``, keep their values.
    The optimizer may try values that violate the constraints of the parametrization
    (for example a negative standard deviation). The target is ``+inf`` at such points,
    so they are never chosen.

    This function is a dispatcher; specific, more efficient implementations can
    be registered for different families and parametrizations.

    Parameters
    ----------
    component : ParametricFamilyDistribution
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
        the :attr:`state` object, or if the block names parameters that are not free
        parameters of the component.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X, H_j = state.X, state.H[:, block.component_id]
    component_id = block.component_id

    params_to_optimize = block.params_to_update(component)
    if not params_to_optimize:
        return component_id, {}

    current = _components.free_parameters(component)

    def target(vector_params: list[float]) -> float:
        try:
            candidate = _components.with_parameters(component, **dict(zip(params_to_optimize, vector_params)))
        except ValueError:
            return np.inf
        lpdf_values = _components.lpdf(candidate, X)
        safe_lpdf = np.where(H_j == 0, 0.0, lpdf_values)
        return float(-np.dot(H_j, safe_lpdf))

    new_params = optimizer.minimize(target, [current[name] for name in params_to_optimize])
    return component_id, {name: float(value) for name, value in zip(params_to_optimize, new_params)}


# ---------------------------------
# Exponential distribution strategy
# ---------------------------------


@q_function_strategy.register(FamilyName.EXPONENTIAL, "rate")
def _(
    component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Specialized Q-function parameter estimation strategy for
    the Exponential distribution using an analytical solution.

    This function provides a closed-form update for the parameter of an
    `Exponential` distribution in the ``rate`` parametrization, which is more
    efficient and precise than general-purpose numerical optimization. It
    calculates the new ``lambda_`` directly from the data and responsibilities.

    Notes
    -----
    The analytical update is the inverse of the weighted mean of the data:
    ``lambda_`` = $1 / m_1$, where $m_1$ is the mean of ``X`` weighted by the
    responsibilities. If $m_1$ is too close to zero, ``lambda_`` is left unchanged.

    This implementation ignores the `optimizer` parameter as it does not
    require numerical optimization.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = block.params_to_update(component)
    new_params: dict[str, float] = {}

    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if N_j <= NUMERICAL_TOLERANCE:
        return block.component_id, {}

    # Update lambda (rate) if it's in the optimization block
    if "lambda_" in params_to_optimize:
        weighted_sum_X = np.dot(H_j, X)
        denominator = weighted_sum_X / N_j

        if denominator > NUMERICAL_TOLERANCE:
            new_params["lambda_"] = float(1.0 / denominator)
        else:
            # If the weighted average is too close to zero,
            # leave rate unchanged to avoid infinity.
            new_params["lambda_"] = _components.free_parameters(component)["lambda_"]

    return block.component_id, new_params


# ----------------------------
# Normal distribution strategy
# ----------------------------


@q_function_strategy.register(FamilyName.NORMAL, "meanStd")
def _(
    component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Specialized Q-function parameter estimation strategy for
    the normal distribution using an analytical solution.

    This function provides a closed-form update for the parameters of a
    `Normal` distribution in the ``meanStd`` parametrization, which is more
    efficient than numerical optimization. It calculates the new ``mu`` (mean)
    and ``sigma`` (standard deviation) directly from the data and responsibilities.

    Notes
    -----
    The analytical updates are as follows:

    - The new mean ``mu`` is the weighted average of the data `X`.
    - The new variance (``sigma`` squared) is the weighted average of the
      squared differences from the mean. If ``mu`` is not being optimized
      (for example, it is fixed with ``family.view(...)``), its current value is used.

    The weights are the responsibilities from the matrix `H`. If the new variance is
    too close to zero, ``sigma`` is left unchanged so that it does not collapse to zero.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = block.params_to_update(component)
    new_params: dict[str, float] = {}

    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if N_j <= NUMERICAL_TOLERANCE:
        return block.component_id, {}

    current = _components.all_parameters(component)

    # Update mean (mu) if it's in the optimization block
    if "mu" in params_to_optimize:
        weighted_sum_X = np.dot(H_j, X)
        new_params["mu"] = float(weighted_sum_X / N_j)

    # Update std (sigma) if it's in the optimization block
    if "sigma" in params_to_optimize:
        # Use the newly computed mean if available, otherwise use the existing one.
        mu = new_params.get("mu", current["mu"])

        # Calculate the weighted variance
        weighted_sum_sq_diff = np.dot(H_j, (X - mu) ** 2)

        new_variance = weighted_sum_sq_diff / N_j

        if new_variance > NUMERICAL_TOLERANCE:
            new_params["sigma"] = float(np.sqrt(new_variance))
        else:
            # If variance is too small, it can lead to instability.
            # Keep the old scale to prevent it from collapsing to zero.
            new_params["sigma"] = current["sigma"]

    return block.component_id, new_params
