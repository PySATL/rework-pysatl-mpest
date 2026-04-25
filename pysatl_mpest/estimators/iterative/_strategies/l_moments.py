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

from ....distributions import ContinuousDistribution, Normal
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


@lmoments_strategy.register(Normal)
def _(
    component: Normal[DType], state: PipelineState[DType], block: OptimizationBlock, optimizer: Optimizer[DType]
) -> tuple[int, dict[str, DType]]:
    """
    Update parameters of a univariate Normal component using L-moments.

    This strategy performs a closed-form (analytical) update of the Normal
    distribution parameters by equating weighted sample L-moments to their
    theoretical definitions. The update is performed for the component
    associated with ``block.component_id`` and is limited to the intersection
    of parameters requested by the component and the optimization block.

    L-moments are often more robust to outliers and more efficient in small
    samples compared to traditional moments or numerical Maximum Likelihood
    Estimation (MLE) in certain mixture contexts.

    If the total responsibility of the component is numerically negligible,
    the function returns without updating any parameters.

    Parameters
    ----------
    component : Normal[DType]
        Normal distribution component to be updated. The method may update
        ``component.loc`` and/or ``component.scale`` depending on
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
        Normal parameter names (e.g., ``component.PARAM_LOC``,
        ``component.PARAM_SCALE``). If no update is performed, an empty dict
        is returned.

    Raises
    ------
    ValueError
        If ``state.H`` is ``None`` (i.e., the responsibility matrix has not been
        computed).

    Notes
    -----
    - Let :math:`l_1` be the weighted mean (first L-moment) and :math:`l_2`
    be the weighted L-scale (second L-moment).

    - **Location Update (:math:`\\mu$):**
    The location is directly equal to the first L-moment:

    .. math::

        \\mu = l_1

    - **Scale Update (:math:`\\sigma$):**
    The relationship between the standard deviation and the second L-moment
    for a Normal distribution is given by:

    .. math::

        \\sigma = l_2 \\sqrt{\\pi}

    - The scale is lower-bounded by machine epsilon for the component dtype to
    avoid degeneracy:

    .. math::

        \\sigma = \\max(\\sigma, \\text{np.finfo(dtype).eps})

    - If the sum of responsibilities :math:`N_j` is close to zero (within
    ``NUMERICAL_TOLERANCE``), the parameters are not updated.

    Examples
    --------
    Update both location and scale for a Normal component using L-moments::

        component_id, new_params = lmoments_strategy(normal_component, state, block, optimizer)
        # new_params may contain {"loc": ..., "scale": ...}
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    dtype = component.dtype
    MIN_SCALE = np.finfo(dtype).eps

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

    if component.PARAM_LOC in params_to_optimize:
        new_loc = l1
        new_params[component.PARAM_LOC] = dtype(new_loc)

    if component.PARAM_SCALE in params_to_optimize:
        new_scale = np.sqrt(np.pi) * l2
        new_params[component.PARAM_SCALE] = dtype(max(new_scale, MIN_SCALE))

    return block.component_id, new_params
