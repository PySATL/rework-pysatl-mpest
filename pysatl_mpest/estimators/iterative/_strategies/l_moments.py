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

from ....distributions import ContinuousDistribution, Exponential, Normal
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
    """Specialized L-moments parameter estimation strategy for
    the Normal distribution using an analytical solution.

    This function provides a closed-form update for the parameters of a
    `Normal` distribution based on the first two sample L-moments (l1, l2).
    L-moments are often more robust to outliers and more efficient in small
    samples compared to traditional moments or numerical MLE in certain
    mixture contexts.

    The implementation calculates the sample L-moments using the responsibility
    matrix `H` from the E-step and maps them to the `loc` and `scale` parameters.

    Notes
    -----
    The analytical updates depend on which parameters are being optimized:
    - If both `loc` and `scale` are optimized:
        - `loc = l1`
        - `scale = sqrt(pi) * l2` (clipped to machine epsilon for numerical stability)
    - If only `loc` is optimized (`scale` is fixed):
        - `loc = l1`
    - If only `scale` is optimized (`loc` is fixed):
        - `scale = sqrt(pi) * l2` (clipped to machine epsilon)

    Where:
        - `l1` is the weighted mean (first L-moment).
        - `l2` is the weighted L-scale (second L-moment).

    This implementation ignores the `optimizer` parameter as it relies on
    direct analytical mapping.

    Raises
    ------
    ValueError
        If the responsibility matrix `H` has not been computed.
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