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
from scipy.special import gamma

from ....distributions import ContinuousDistribution, Weibull
from ....optimizers import Optimizer
from ....typings import DType
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock
from .utils import handle_numerical_overflow

NUMERICAL_TOLERANCE = 1e-9


# ------------------------
# function to compute 1-st and 2-nd l-moments
# ------------------------
def compute_sample_lmoments(X: np.ndarray, H: np.ndarray, N_j: DType, need_l3: bool = False) -> tuple[float, float, float]:
    idx = np.argsort(X)
    X_sorted = X[idx]
    H_sorted = H[idx]

    l1 = np.sum(H_sorted * X_sorted) / N_j

    W_sum = np.cumsum(H_sorted)
    rank_weights = (W_sum - 0.5 * H_sorted) / N_j

    b1 = np.sum(H_sorted * X_sorted * rank_weights) / N_j
    l2 = 2 * b1 - l1

    if not need_l3:
        return float(l1), float(l2), 0.0
    
    b2 = np.sum(H_sorted * X_sorted * (rank_weights**2)) / N_j
    
    l3 = 6 * b2 - 6 * b1 + l1

    return float(l1), float(l2), float(l3)


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


@lmoments_strategy.register(Weibull)
def _(
    component: Weibull[DType], state: PipelineState[DType], block: OptimizationBlock, optimizer: Optimizer[DType]
) -> tuple[int, dict[str, DType]]:
    """Update parameters of a univariate Weibull component using L-moments.

    This strategy performs a parameter update for the Weibull distribution by
    equating sample L-moments (computed from the current responsibility matrix)
    to their theoretical counterparts. The update is performed for the component
    associated with ``block.component_id`` and supports both two-parameter and
    three-parameter (shifted) Weibull configurations.

    The estimation uses a combination of analytical solutions and rational
    polynomial approximations. For the three-parameter case (where both location
    and shape are optimized), Hosking's approximation is used for the shape
    parameter. For the two-parameter case (fixed location), a logarithmic
    ratio is used to solve for shape and scale.

    If the total responsibility of the component is numerically negligible, or
    if the L-scale is degenerate, the function returns without updating any
    parameters.

    Parameters
    ----------
    component : Weibull[DType]
        Weibull distribution component to be updated. The method may update
        ``component.loc``, ``component.scale``, and/or ``component.shape``
        depending on ``component.params_to_optimize`` and the block configuration.
    state : PipelineState[DType]
        Current pipeline state containing:

        - ``X`` : array-like
        Observations used to compute sample L-moments.
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
        Weibull parameter names (e.g., ``component.PARAM_LOC``,
        ``component.PARAM_SCALE``, ``component.PARAM_SHAPE``). If no update
        is performed, an empty dict is returned.

    Raises
    ------
    ValueError
        If ``state.H`` is ``None`` (i.e., the responsibility matrix has not been
        computed) or if the computed L-skewness falls outside the valid range
        for approximation.

    Notes
    -----
    - Let :math:`L_1, L_2, L_3` be the first three sample L-moments and
    :math:`\\Gamma(\\cdot)` be the gamma function. The updates depend on the
    set of parameters being optimized:

    - **Location Update (:math:`\\gamma`):**
    The location is recovered by shifting the first L-moment by the
    theoretical mean of the non-shifted distribution:

    .. math::

        \\gamma = L_1 - \\lambda \\Gamma(1 + 1/k)

    - **Scale Update (:math:`\\lambda`):**
    - If the **location is optimized** (unknown), the scale is derived from
        the second L-moment (L-scale):

        .. math::

            \\lambda = \\frac{L_2}{\\Gamma(1 + 1/k)(1 - 2^{-1/k})}

    - If the **location is fixed**, the scale is updated using the
        relationship between the mean and the fixed location:

        .. math::

            \\lambda = \\frac{L_1 - \\gamma}{\\Gamma(1 + 1/k)}

    - **Shape Update (:math:`k`):**
    - If **location is optimized**, :math:`k` is found via L-skewness
        :math:`\\tau_3 = L_3 / L_2`:

        .. math::

            k \\approx \\frac{3.5208 - 2.0905\\tau_3 + 1.1370\\tau_3^2 - 1.4688\\tau_3^3}{1.0 + 5.6836\\tau_3}

    - If **location is fixed**, :math:`k` uses the ratio of :math:`L_1` and :math:`L_2`:

        .. math::

            k = -\\frac{\\ln(2)}{\\ln(1 - L_2 / (L_1 - \\gamma))}

    - The L-skewness :math:`\\tau_3` is clipped to :math:`[0.001, 0.499]` to ensure
    the stability of the polynomial approximation.
    - All scale results are lower-bounded by machine epsilon for the component
    dtype to avoid degeneracy.

    Examples
    --------
    Update all three parameters for component ``j``::

        component_id, new_params = lmoments_strategy(weibull_component, state, block, optimizer)
        # new_params may contain {"loc": ..., "scale": ..., "shape": ...}
        
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

    opt_loc = component.PARAM_LOC in params_to_optimize
    opt_scale = component.PARAM_SCALE in params_to_optimize
    opt_shape = component.PARAM_SHAPE in params_to_optimize

    l1, l2, l3 = compute_sample_lmoments(X, H_j, N_j, need_l3=(opt_shape and opt_loc))

    for val, name in [(l1, "l1"), (l2, "l2"), (l3, "l3")]:
        if not np.isfinite(val):
            handle_numerical_overflow(state=state, context=f"Lmoments optimization ({name})")
            return block.component_id, {}

    if l2 <= NUMERICAL_TOLERANCE:
        handle_numerical_overflow(state=state, context="Lmoments optimization (degenerate L-scale)")
        return block.component_id, {}

    new_params = {}

    if opt_shape:

        if opt_loc: 
            t3 = l3 / l2
            t3 = np.clip(t3, 0.001, 0.499)

            if not (0 < t3 < 0.5): raise ValueError("t3 must be in (0, 0.5)")

            new_shape = ((3.5208453 - 2.0905222 * t3 + 1.1370309 * t3**2 - 1.4688549 * t3**3) /
                                (1.0 + 5.6836423*t3))
            new_params[component.PARAM_SHAPE] = dtype(new_shape)

            gamma_part = gamma(1 + (1 / new_shape))

            new_scale = component.scale
            if opt_scale:
                new_scale = l2 / (gamma_part * (1 - 2**(-1 / new_shape)))

                new_params[component.PARAM_SCALE] = dtype(new_scale)
            
            new_params[component.PARAM_LOC] = dtype(l1 - new_scale * gamma_part)

        else:
            ratio = l2 / (l1 - component.loc)
            ratio = np.clip(ratio, 1e-6, 1 - 1e-6)

            new_shape = -np.log(2) / np.log(1 - ratio)
            new_params[component.PARAM_SHAPE] = dtype(new_shape)

            if opt_scale:
                new_scale = (l1 - component.loc) / gamma(1 + 1 / new_shape)
                new_params[component.PARAM_SCALE] = dtype(new_scale) 

    else:
        gamma_part = gamma(1 + 1/component.shape)
        
        if opt_loc and opt_scale:
        
            new_scale = l2 / (gamma_part * (1 - 2**(-1 / component.shape)))
            new_params[component.PARAM_SCALE] = dtype(new_scale)
            new_params[component.PARAM_LOC] = dtype(l1 - new_scale * gamma_part)
        
        elif opt_scale:
            
            new_params[component.PARAM_SCALE] = dtype((l1 - component.loc) / gamma_part)
            
        elif opt_loc:
    
            new_params[component.PARAM_LOC] = dtype(l1 - component.scale * gamma_part)
    
    for key, val in new_params.items():
        if not np.isfinite(val):
            handle_numerical_overflow(state=state, context=f"Lmoments optimization (result {key} overflow)")
            return block.component_id, {}

    return block.component_id, new_params
