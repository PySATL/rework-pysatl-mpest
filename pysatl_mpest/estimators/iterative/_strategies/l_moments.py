"""Provides strategies for the Maximization-step based on the Method of L-moments.

This module implements the logic for updating component parameters by
equating theoretical L-moments of the distribution to the empirical weighted
L-moments of the data. It uses `functools.singledispatch` to provide a generic
interface that can be specialized for specific distribution types.
"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from functools import singledispatch

import numpy as np

from ....distributions import ContinuousDistribution, Exponential
from ....optimizers import Optimizer
from ....typings import DType
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock
from .utils import handle_numerical_overflow

NUMERICAL_TOLERANCE = 1e-9


# ------------------------
# function to compute 1-st and 2-nd l-moments
# ------------------------
def compute_sample_lmoments(X: np.ndarray, H: np.ndarray, N_j: DType) -> tuple[float, float]:
    idx = np.argsort(X)
    X_sorted = X[idx]
    H_sorted = H[idx]

    l1 = np.sum(H_sorted * X_sorted) / N_j

    W_sum = np.cumsum(H_sorted)
    rank_weights = (W_sum - 0.5 * H_sorted) / N_j

    b1 = np.sum(H_sorted * X_sorted * rank_weights) / N_j
    l2 = 2 * b1 - l1

    return float(l1), float(l2)


# ------------------------
# Base L-moments strategy
# ------------------------


@singledispatch
def lmoments_strategy(
    component: ContinuousDistribution[DType],
    state: PipelineState[DType],
    block: OptimizationBlock,
    optimizer: Optimizer[DType],
) -> tuple[int, dict[str, DType]]:
    """Generic M-step strategy based on L-moments.

    Unlike the Q-function strategy, L-moments do not have a universal numerical
    fallback because the relationship between L-moments and distribution
    parameters is distribution-specific.

    This function serves as a dispatcher. Specific implementations must be
    registered for each distribution type.

    Raises
    ------
    NotImplementedError
        If no L-moment implementation exists for the given distribution type.
    ValueError
        If the responsibility matrix H is missing from the state.
    """
    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    raise NotImplementedError(f"L-moments strategy for the {component.name} distribution is not implemented.")


@lmoments_strategy.register(Exponential)
def _(
    component: Exponential[DType], state: PipelineState[DType], block: OptimizationBlock, optimizer: Optimizer[DType]
) -> tuple[int, dict[str, DType]]:
    """
    Update parameters of a univariate Exponential component using L-moments.

    This strategy performs a closed-form (analytical) update of the Exponential
    distribution parameters (location and rate) by equating weighted sample 
    L-moments to their theoretical definitions. The update is performed for 
    the component associated with ``block.component_id`` and adapts based on 
    which parameters are flagged for optimization.

    L-moments provide a robust alternative to maximum likelihood estimation (MLE), 
    especially in mixture models where components may have small effective 
    sample sizes or contain outliers.

    If the total responsibility of the component is numerically negligible, or 
    if the L-moments results in non-finite values, the function returns without 
    updating any parameters.

    Parameters
    ----------
    component : Exponential[DType]
        Exponential distribution component to be updated. The method may update
        ``component.loc`` and/or ``component.rate`` depending on
        ``component.params_to_optimize`` and the block configuration.
    state : PipelineState[DType]
        Current pipeline state containing:

        - ``X`` : array-like
        Observations used to compute L-moment estimates.
        - ``H`` : array-like, shape (n_samples, n_components)
        Responsibility matrix. Column ``block.component_id`` is used as
        weights for this component.
    block : OptimizationBlock
        Optimization block describing which component is being optimized and
        which parameters are allowed to change. The component index is taken
        from ``block.component_id``.
    optimizer : Optimizer[DType]
        Optimizer instance provided by the pipeline. It is not used directly by
        this analytical strategy but is included for API consistency.

    Returns
    -------
    component_id : int
        The identifier of the optimized component, equal to
        ``block.component_id``.
    new_params : dict[str, DType]
        Dictionary of updated parameters for the component. Keys correspond to
        Exponential parameter names (e.g., ``component.PARAM_LOC``,
        ``component.PARAM_RATE``). If no update is performed, an empty dict 
        is returned.

    Raises
    ------
    ValueError
        If ``state.H`` is ``None`` (i.e., the responsibility matrix has not been
        computed).

    Notes
    -----
    - Let :math:`l_1` be the first weighted sample L-moment (mean) and :math:`l_2` 
    be the second weighted sample L-moment (L-scale). The theoretical 
    L-moments for an Exponential distribution with location :math:`\\gamma` 
    and rate :math:`\\lambda` are:
    
    .. math::
        L_1 = \\gamma + \\frac{1}{\\lambda}, \\quad L_2 = \\frac{1}{2\\lambda}

    - **Case 1: Both Location and Rate are optimized**
    The parameters are recovered using both :math:`l_1` and :math:`l_2`:

    .. math::
        \\lambda = \\frac{1}{2l_2}, \\quad \\gamma = l_1 - \\frac{1}{\\lambda}

    - **Case 2: Rate is optimized, Location is fixed**
    The rate is estimated using the first L-moment and the fixed location:

    .. math::
        \\lambda = \\frac{1}{l_1 - \\gamma_{fixed}}

    - **Case 3: Location is optimized, Rate is fixed**
    The location is adjusted based on the first L-moment and the fixed rate:

    .. math::
        \\gamma = l_1 - \\frac{1}{\\lambda_{fixed}}

    - If the sum of responsibilities :math:`N_j` is close to zero (within 
    ``NUMERICAL_TOLERANCE``), the parameters are not updated.
    - Rates are checked against ``NUMERICAL_TOLERANCE`` to prevent division by zero.

    Examples
    --------
    Update location and rate for an Exponential component using L-moments::

        component_id, new_params = lmoments_strategy(exponential_component, state, block, optimizer)
        # new_params may contain {"loc": ..., "rate": ...}
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    dtype = component.dtype

    X = state.X
    H_j = state.H[:, block.component_id]
    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if np.isclose(N_j, 0.0, atol=NUMERICAL_TOLERANCE):
        return block.component_id, {}

    params_to_optimize = component.params_to_optimize.intersection(block.params_to_optimize)
    new_params = {}

    l1, l2 = compute_sample_lmoments(X, H_j, N_j)

    if np.isinf(l1):
        handle_numerical_overflow(state=state, context="Lmoments optimization")
        return block.component_id, {}

    if np.isinf(l2):
        handle_numerical_overflow(state=state, context="Lmoments optimization")
        return block.component_id, {}

    if component.PARAM_RATE in params_to_optimize and component.PARAM_LOC in params_to_optimize:
        # l2 = 1 / (2 * rate) => rate = 1 / (2 * l2)
        # l1 = loc + 1 / rate => loc = l1 - 1 / rate
        new_rate: DType | float = 1.0 / (2.0 * l2) if l2 > NUMERICAL_TOLERANCE else 1e-6
        new_loc: DType | float = l1 - (1.0 / new_rate)

        new_params[component.PARAM_RATE] = dtype(new_rate)
        new_params[component.PARAM_LOC] = dtype(new_loc)

    # Сценарий 2: Фиксированный loc, свободный rate
    elif component.PARAM_RATE in params_to_optimize:
        # loc фиксирован. Используем l1 для оценки rate (более эффективно, чем l2)
        # l1 = loc_fixed + 1 / rate => rate = 1 / (l1 - loc_fixed)
        diff = l1 - component.loc

        new_rate = component.rate if np.isclose(diff, 0.0, atol=1e-12) else 1.0 / diff

        new_params[component.PARAM_RATE] = dtype(new_rate)

    # Сценарий 3: Фиксированный rate, свободный loc
    elif component.PARAM_LOC in params_to_optimize:
        # rate фиксирован.
        # loc = l1 - 1 / rate_fixed

        new_loc = l1 - (1.0 / component.rate)

        if np.isinf(new_loc):
            handle_numerical_overflow(state=state, context="Lmoments optimization")

        new_params[component.PARAM_LOC] = dtype(new_loc)

    return block.component_id, new_params
