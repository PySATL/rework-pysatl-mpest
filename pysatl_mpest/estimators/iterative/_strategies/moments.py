"""Provides strategies for the Maximization-step based on the Method of Moments.

This module implements the logic for updating component parameters by
equating theoretical moments of the distribution to the empirical weighted
moments of the data. It uses `functools.singledispatch` to provide a generic
interface that can be specialized for specific distribution types.
"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from functools import singledispatch

import numpy as np

from ....distributions import ContinuousDistribution, Exponential, Normal
from ....optimizers import Optimizer
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock

NUMERICAL_TOLERANCE = 1e-9


# ------------------------
# Base Moments strategy
# ------------------------


@singledispatch
def moments_strategy(
    component: ContinuousDistribution,
    state: PipelineState,
    block: OptimizationBlock,
    optimizer: Optimizer,
) -> tuple[int, dict[str, float]]:
    """Generic M-step strategy that uses the Method of Moments.

    This function serves as the base dispatcher. Since the Method of Moments
    typically requires analytical expressions for theoretical moments specific
    to each distribution, a generic implementation is not provided and raises
    NotImplementedError.

    Parameters
    ----------
    component : ContinuousDistribution
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
        If a specialized moments strategy is not registered for the given
        distribution type.
    """
    raise NotImplementedError(f"Moments strategy is not implemented for distribution '{component.name}'.")


# ---------------------------------
# Exponential distribution strategy
# ---------------------------------


@moments_strategy.register(Exponential)
def _(
    component: Exponential, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Specialized Moments parameter estimation strategy for the Exponential distribution
    using an analytical solution.

    This function provides a closed-form update for the parameters of an
    `Exponential` distribution using the Method of Moments. It equates the
    theoretical moments to the empirical weighted moments calculated from
    the data and responsibilities.

    Notes
    -----
    The analytical updates depend on the set of parameters to optimize:

    - **Both `loc` and `rate`**: Derived from the first weighted moment ($m_1$)
      and the second weighted moment ($m_2$). The variance is estimated as
      $Var = m_2 - m_1^2$.
      Then, ``rate`` = $1 / \\sqrt{Var}$ and ``loc`` = $m_1 - \\sqrt{Var}$.

    - **Only `rate`** (fixed `loc`): Derived from the first moment.
      ``rate`` = $1 / (m_1 - \\text{loc})$.

    - **Only `loc`** (fixed `rate`): Derived from the first moment.
      ``loc`` = $m_1 - (1 / \\text{rate})$.

    This implementation ignores the `optimizer` parameter as it does not
    require numerical optimization.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = component.params_to_optimize.intersection(block.params_to_optimize)
    new_params = {}

    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if np.isclose(N_j, 0.0, atol=NUMERICAL_TOLERANCE):
        return block.component_id, {}

    weighted_sum_X = np.dot(H_j, X)

    m1 = weighted_sum_X / N_j

    # Update lambda (rate) if it's in the optimization block
    if "lambda_" in params_to_optimize:
        if np.isclose(m1, 0.0, atol=NUMERICAL_TOLERANCE):
            new_params["lambda_"] = component.lambda_
        else:
            new_params["lambda_"] = float(1.0 / m1)

    return block.component_id, new_params


# ---------------------------------
# Normal distribution strategy
# ---------------------------------


@moments_strategy.register(Normal)
def _(
    component: Normal, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """
    Update parameters of a univariate Normal component using weighted moments.

    This strategy performs a moment-based (closed-form) update of the Normal
    distribution parameters from the current responsibility matrix produced by
    the mixture/pipeline state. The update is performed for the component
    associated with ``block.component_id`` and is limited to the intersection of
    parameters requested by both the component and the optimization block.

    The location is updated as the responsibility-weighted mean of the observed
    data. The scale is updated as the square root of the responsibility-weighted
    second central moment (i.e., the weighted standard deviation) around the
    updated (or fixed) location.

    If the total responsibility of the component is numerically negligible, the
    function returns without updating any parameters.

    Parameters
    ----------
    component : Normal
        Normal distribution component to be updated. The method may update
        ``component.loc`` and/or ``component.scale`` depending on
        ``component.params_to_optimize`` and the block configuration.
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
        computed).

    Notes
    -----
    - Let ``w_i = H[i, j]`` be responsibilities for component ``j`` and ``x_i``
      the observations.

      The updates are:

      .. math::

         N_j = \\sum_i w_i

         \\mu_j = \\frac{\\sum_i w_i x_i}{N_j}

         \\sigma_j = \\sqrt{\\frac{\\sum_i w_i (x_i - \\mu_j)^2}{N_j}}

    - The scale is lower-bounded by machine epsilon for the component np.float64 to
      avoid degeneracy:

      ``scale = max(scale, np.finfo(np.float64).eps)``.

    - If ``N_j`` is close to zero (within ``NUMERICAL_TOLERANCE``), the
      parameters are not updated.

    Examples
    --------
    Update both location and scale for component ``j`` using responsibilities::

        component_id, new_params = moments_strategy(normal_component, state, block, optimizer)
        # new_params may contain {"loc": ..., "scale": ...} depending on configuration.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    MIN_SCALE = np.finfo(np.float64).eps
    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = component.params_to_optimize.intersection(block.params_to_optimize)
    new_params: dict[str, float] = {}

    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if np.isclose(N_j, 0.0, atol=NUMERICAL_TOLERANCE):
        return block.component_id, {}

    if "mu" in params_to_optimize:
        new_loc = float(np.average(X, weights=H_j))
        new_params["mu"] = new_loc
    else:
        new_loc = component.mu

    if "sigma" in params_to_optimize:
        new_scale = np.sqrt(np.average((X - new_loc) ** 2, weights=H_j))
        new_params["sigma"] = float(new_scale)

        new_scale = max(new_scale, MIN_SCALE)

    return block.component_id, new_params
