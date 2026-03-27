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
    """Specialized L-moments parameter estimation strategy for
    the Exponential distribution using an analytical solution.

    This function provides a closed-form update for the parameters of an
    `Exponential` distribution based on the first two sample L-moments (l1, l2).
    L-moments are often more robust to outliers and more efficient in small
    samples compared to traditional moments or numerical MLE in certain
    mixture contexts.

    The implementation calculates the sample L-moments using the responsibility
    matrix `H` from the E-step and maps them to the `loc` and `rate` parameters.

    Notes
    -----
    The analytical updates depend on which parameters are being optimized:
    - If both `loc` and `rate` are optimized:
        - `rate = 1 / (2 * l2)`
        - `loc = l1 - l2 * 2` (equivalent to `l1 - 1 / rate`)
    - If only `rate` is optimized (`loc` is fixed):
        - `rate = 1 / (l1 - loc_fixed)`
    - If only `loc` is optimized (`rate` is fixed):
        - `loc = l1 - 1 / rate_fixed`

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

    X = state.X
    H_j = state.H[:, block.component_id]
    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if np.isclose(N_j, 0.0, atol=NUMERICAL_TOLERANCE):
        return block.component_id, {}

    params_to_optimize = component.params_to_optimize.intersection(block.params_to_optimize)
    new_params = {}

    idx = np.argsort(X)

    X_sorted, H_sorted = X[idx], H_j[idx]
    W_sum = np.cumsum(H_sorted)

    l1 = np.sum(H_sorted * X_sorted) / N_j

    if np.isinf(l1):
        handle_numerical_overflow(state=state, context="Lmoments optimization")
        return block.component_id, {}

    rank_weights = (W_sum - 0.5 * H_sorted) / N_j
    b1 = np.sum(H_sorted * X_sorted * rank_weights) / N_j
    l2 = 2 * b1 - l1

    if np.isinf(l2):
        handle_numerical_overflow(state=state, context="Lmoments optimization")
        return block.component_id, {}

    if component.PARAM_RATE in params_to_optimize and component.PARAM_LOC in params_to_optimize:
        # l2 = 1 / (2 * rate) => rate = 1 / (2 * l2)
        # l1 = loc + 1 / rate => loc = l1 - 1 / rate
        new_rate = 1.0 / (2.0 * l2) if l2 > NUMERICAL_TOLERANCE else 1e-6
        new_loc = l1 - (1.0 / new_rate)

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
