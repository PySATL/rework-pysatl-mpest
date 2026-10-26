"""Provides strategies for the Maximization-step based on the Method of Moments.

This module implements the logic for updating component parameters by
equating theoretical moments of the distribution to the empirical weighted
moments of the data. It dispatches on the family and parametrization of the component to provide a generic
interface that can be specialized for specific distribution types.
"""

__author__ = "Aleksandra Ri"
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
# Base Moments strategy
# ------------------------


def _not_implemented(
    component: ParametricFamilyDistribution,
    state: PipelineState,
    block: OptimizationBlock,
    optimizer: Optimizer,
) -> tuple[int, dict[str, float]]:
    """Generic M-step strategy that uses the Method of Moments.

    This function is the default implementation of :data:`moments_strategy`. Since the
    Method of Moments typically requires analytical expressions for theoretical
    moments specific to each distribution, a generic implementation is not provided
    and it raises NotImplementedError.

    Parameters
    ----------
    component : ParametricFamilyDistribution
        The distribution component whose parameters are to be optimized.
    state : PipelineState
        The current state of the pipeline.
    block : OptimizationBlock
        The configuration block defining which component to optimize.
    optimizer : Optimizer
        The numerical optimizer (unused in this strategy).

    Raises
    ------
    NotImplementedError
        If a specialized moments strategy is not registered for the family and
        parametrization of the component.
    """
    raise NotImplementedError(
        f"Moments strategy is not implemented for the '{component.parametrization_name}' "
        f"parametrization of the '{component.family_name}' family."
    )


# ---------------------------------
# Exponential distribution strategy
# ---------------------------------


def _exponential_rate(
    component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """
    Update the parameter of a univariate Exponential component using the weighted first moment.

    This strategy performs a moment-based (closed-form) update of the Exponential
    distribution in the ``rate`` parametrization from the current responsibility
    matrix. The update is performed for the component associated with
    ``block.component_id`` and is limited to the parameters named by the
    optimization block (all free parameters if ``block.params_to_optimize``
    is ``None``). The theoretical mean of the distribution is equated to the
    responsibility-weighted mean of the observed data.

    If the total responsibility of the component is numerically negligible, the
    function returns without updating any parameters.

    Parameters
    ----------
    component : ParametricFamilyDistribution
        Exponential distribution component in the ``rate`` parametrization to be
        updated. The method may update ``lambda_`` depending on the block configuration.
    state : PipelineState
        Current pipeline state containing:

        - ``X`` : array-like
          Observations used to compute the estimates.
        - ``H`` : array-like, shape (n_samples, n_components)
          Responsibility matrix. Column ``block.component_id`` is used as
          weights for this component.
    block : OptimizationBlock
        Optimization block describing which component is being optimized and
        which parameters are allowed to change. The component index is taken
        from ``block.component_id``.
    optimizer : Optimizer
        Optimizer instance provided by the pipeline. It is not used directly by
        this analytical strategy but is included for API consistency.

    Returns
    -------
    component_id : int
        The identifier of the optimized component, equal to
        ``block.component_id``.
    new_params : dict[str, float]
        Dictionary of updated parameters for the component. Keys correspond to
        the parameter names (``"lambda_"``). If no update is performed (e.g., negligible
        responsibility), an empty dict is returned.

    Raises
    ------
    ValueError
        If ``state.H`` is ``None`` (i.e., the responsibility matrix has not been
        computed), or if the block names parameters that are not free parameters
        of the component.

    Notes
    -----
    - Let ``w_i = H[i, j]`` be responsibilities for component ``j`` and ``x_i``
      the observations.

      The mean of the distribution is $1 / \\lambda$, so the update is:

      .. math::

         N_j = \\sum_i w_i

         m_1 = \\frac{\\sum_i w_i x_i}{N_j}

         \\lambda_j = \\frac{1}{m_1}

    - If ``m_1`` is not greater than ``NUMERICAL_TOLERANCE``, the rate would be
      infinite or negative, which pysatl-core rejects, so ``lambda_`` is left
      unchanged.

    - If ``N_j`` is close to zero (within ``NUMERICAL_TOLERANCE``), the
      parameters are not updated.

    Examples
    --------
    Update the rate of component ``j`` using responsibilities::

        component_id, new_params = moments_strategy(exponential_component, state, block, optimizer)
        # new_params may contain {"lambda_": ...} depending on configuration.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = block.params_to_update(component)
    new_params: dict[str, float] = {}

    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if np.isclose(N_j, 0.0, atol=NUMERICAL_TOLERANCE):
        return block.component_id, {}

    weighted_sum_X = np.dot(H_j, X)

    m1 = weighted_sum_X / N_j

    # Update lambda (rate) if it's in the optimization block
    if "lambda_" in params_to_optimize:
        if m1 > NUMERICAL_TOLERANCE:
            new_params["lambda_"] = float(1.0 / m1)
        else:
            new_params["lambda_"] = _components.free_parameters(component)["lambda_"]

    return block.component_id, new_params


# ---------------------------------
# Normal distribution strategy
# ---------------------------------


def _normal_mean_std(
    component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """
    Update parameters of a univariate Normal component using weighted moments.

    This strategy performs a moment-based (closed-form) update of the Normal
    distribution parameters from the current responsibility matrix produced by
    the mixture/pipeline state. The update is performed for the component
    associated with ``block.component_id`` and is limited to the parameters named
    by the optimization block (all free parameters if ``block.params_to_optimize``
    is ``None``).

    The mean ``mu`` is updated as the responsibility-weighted mean of the observed
    data. The standard deviation ``sigma`` is updated as the square root of the
    responsibility-weighted second central moment (i.e., the weighted variance)
    around the updated (or fixed) ``mu``.

    If the total responsibility of the component is numerically negligible, the
    function returns without updating any parameters.

    Parameters
    ----------
    component : ParametricFamilyDistribution
        Normal distribution component in the ``meanStd`` parametrization to be
        updated. The method may update ``mu`` and/or ``sigma`` depending on the
        block configuration. A parameter fixed with ``family.view(...)`` keeps its
        value, and the spread is measured around it.
    state : PipelineState
        Current pipeline state containing:

        - ``X`` : array-like
          Observations used to compute moment estimates.
        - ``H`` : array-like, shape (n_samples, n_components)
          Responsibility matrix. Column ``block.component_id`` is used as
          weights for this component.
    block : OptimizationBlock
        Optimization block describing which component is being optimized and
        which parameters are allowed to change. The component index is taken
        from ``block.component_id``.
    optimizer : Optimizer
        Optimizer instance provided by the pipeline. It is not used directly by
        this moments-based strategy but is included for API consistency.

    Returns
    -------
    component_id : int
        The identifier of the optimized component, equal to
        ``block.component_id``.
    new_params : dict[str, float]
        Dictionary of updated parameters for the component. Keys correspond to
        Normal parameter names (e.g., ``"mu"``,
        ``"sigma"``). If no update is performed (e.g., negligible
        responsibility), an empty dict is returned.

    Raises
    ------
    ValueError
        If ``state.H`` is ``None`` (i.e., the responsibility matrix has not been
        computed), or if the block names parameters that are not free parameters
        of the component.

    Notes
    -----
    - Let ``w_i = H[i, j]`` be responsibilities for component ``j`` and ``x_i``
      the observations.

      The updates are:

      .. math::

         N_j = \\sum_i w_i

         \\mu_j = \\frac{\\sum_i w_i x_i}{N_j}

         \\sigma_j = \\sqrt{\\frac{\\sum_i w_i (x_i - \\mu_j)^2}{N_j}}

    - If the weighted variance is not greater than ``NUMERICAL_TOLERANCE`` (for
      example, all the data coincide with ``mu``), ``sigma`` is left
      unchanged so that it does not collapse to zero. This is the same rule as in
      the Q-function strategy.

    - If ``N_j`` is close to zero (within ``NUMERICAL_TOLERANCE``), the
      parameters are not updated.

    Examples
    --------
    Update both ``mu`` and ``sigma`` for component ``j`` using responsibilities::

        component_id, new_params = moments_strategy(normal_component, state, block, optimizer)
        # new_params may contain {"mu": ..., "sigma": ...} depending on configuration.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = block.params_to_update(component)
    new_params: dict[str, float] = {}

    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if np.isclose(N_j, 0.0, atol=NUMERICAL_TOLERANCE):
        return block.component_id, {}

    current = _components.all_parameters(component)

    if "mu" in params_to_optimize:
        new_loc = float(np.average(X, weights=H_j))
        new_params["mu"] = new_loc
    else:
        new_loc = current["mu"]

    if "sigma" in params_to_optimize:
        new_variance = np.average((X - new_loc) ** 2, weights=H_j)

        if new_variance > NUMERICAL_TOLERANCE:
            new_params["sigma"] = float(np.sqrt(new_variance))
        else:
            # The variance is degenerate: keep the old sigma to prevent it from collapsing to zero.
            new_params["sigma"] = current["sigma"]

    return block.component_id, new_params


# --------
# Dispatch
# --------

# The implementations are plain functions and are registered by a call, not by a decorator:
# the mutation testing tool does not mutate decorated functions.
moments_strategy = family_dispatch(_not_implemented)
moments_strategy.register(FamilyName.EXPONENTIAL, "rate")(_exponential_rate)
moments_strategy.register(FamilyName.NORMAL, "meanStd")(_normal_mean_std)
