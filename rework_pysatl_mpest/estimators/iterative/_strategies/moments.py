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

from ....distributions import ContinuousDistribution, Exponential
from ....optimizers import Optimizer
from ....typings import DType
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock
from .utils import handle_numerical_overflow

NUMERICAL_TOLERANCE = 1e-9


# ------------------------
# Base Moments strategy
# ------------------------


@singledispatch
def moments_strategy(
    component: ContinuousDistribution[DType],
    state: PipelineState[DType],
    block: OptimizationBlock,
    optimizer: Optimizer[DType],
) -> tuple[int, dict[str, DType]]:
    """Generic M-step strategy that uses the Method of Moments.

    This function serves as the base dispatcher. Since the Method of Moments
    typically requires analytical expressions for theoretical moments specific
    to each distribution, a generic implementation is not provided and raises
    NotImplementedError.

    Parameters
    ----------
    component : ContinuousDistribution[DType]
        The distribution component whose parameters are to be optimized.
    state : PipelineState
        The current state of the pipeline.
    block : OptimizationBlock
        The configuration block defining which component to optimize.
    optimizer : Optimizer[DType]
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
    component: Exponential[DType], state: PipelineState[DType], block: OptimizationBlock, optimizer: Optimizer[DType]
) -> tuple[int, dict[str, DType]]:
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

    dtype = component.dtype
    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = component.params_to_optimize.intersection(block.params_to_optimize)
    new_params = {}

    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if np.isclose(N_j, 0.0, atol=NUMERICAL_TOLERANCE):
        return block.component_id, {}

    weighted_sum_X = np.dot(H_j, X)
    if np.isinf(weighted_sum_X):
        handle_numerical_overflow(state, context="Moments optimization")
        return block.component_id, {}

    m1 = weighted_sum_X / N_j

    # Update both location (loc) and lambda (rate) if they are in the optimization block
    if Exponential.PARAM_LOC in params_to_optimize and Exponential.PARAM_RATE in params_to_optimize:
        weighted_sum_X2 = np.dot(H_j, X**2)
        if np.isinf(weighted_sum_X2):
            handle_numerical_overflow(state, context="Moments optimization")
            return block.component_id, {}

        m2 = weighted_sum_X2 / N_j

        variance = np.maximum(m2 - m1**2, dtype(NUMERICAL_TOLERANCE))

        std_dev = np.sqrt(variance)

        new_params[Exponential.PARAM_RATE] = dtype(1.0 / std_dev)
        new_params[Exponential.PARAM_LOC] = dtype(m1 - std_dev)

    # Update lambda (rate) if it's in the optimization block
    elif Exponential.PARAM_RATE in params_to_optimize:
        denominator = m1 - component.loc

        if np.isclose(denominator, 0.0, NUMERICAL_TOLERANCE):
            new_params[Exponential.PARAM_RATE] = component.rate
        else:
            new_params[Exponential.PARAM_RATE] = dtype(1.0 / denominator)

    # Update location (loc) if it's in the optimization block
    elif Exponential.PARAM_LOC in params_to_optimize:
        new_loc = m1 - (dtype(1.0) / component.rate)
        new_params[Exponential.PARAM_LOC] = dtype(new_loc)

    return block.component_id, new_params
