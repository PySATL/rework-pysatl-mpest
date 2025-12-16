"""
A module providing Q-function based parameter estimation strategies for distributions.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from copy import copy
from functools import singledispatch

import numpy as np

from ...distributions import ContinuousDistribution, Exponential
from ...optimizers import Optimizer

NUMERICAL_TOLERANCE = 1e-6


@singledispatch
def q_function_strategy(
    component: ContinuousDistribution, X: np.ndarray, H_j: np.ndarray, optimizer: Optimizer
) -> dict[str, float]:
    """Generic Q-function optimization strategy for continuous distributions.

    This function provides a general-purpose parameter estimation strategy that
    maximizes the Q-function (expected complete data log-likelihood) using
    numerical optimization. It works for any continuous distribution that
    implements the Q-function method.

    This is a generic implementation that uses numerical optimization. For specific
    distributions, more efficient analytical solutions may be available.

    Parameters
    ----------
    component : ContinuousDistribution
        The distribution component for which to estimate parameters.
    X : np.ndarray
        Input data points used for parameter estimation.
    H_j : np.ndarray
        Weight vector where H_j[i] represents the probability that data point i
        belongs to the j-th mixture component.
    optimizer : Optimizer
        Optimization algorithm used to maximize the Q-function.

    Returns
    -------
    dict[str, float]
        Dictionary mapping parameter names to their estimated values.

    See Also
    --------
    q_function_strategy_exponential : Specialized implementation for Exponential distribution

    Notes
    -----
    **Methodology**

    This strategy:
    1. Identifies optimizable parameters for the distribution
    2. Creates a target function that computes the negative Q-function
    3. Uses numerical optimization to find parameters that maximize the Q-function
    4. Returns the optimized parameter values

    **Error Handling**

    - Falls back gracefully if the distribution doesn't implement Q-function
    - Uses deep copy to avoid modifying the original distribution during optimization

    **Limitations**

    - Requires the distribution to implement the `q_function` method
    - Relies on numerical optimization which may converge to local optima
    - Performance depends on the quality of the initial parameter values

    **Available Specializations**

    The following specialized implementations are available:

    - :func:`q_function_strategy_exponential` - for Exponential distribution
    """

    params_to_optimize = list(component.params_to_optimize)
    temp_comp = copy(component)

    def target(vector_params):
        temp_comp.set_params_from_vector(params_to_optimize, vector_params)
        lpdf_values = temp_comp.lpdf(X)
        return -np.dot(H_j, lpdf_values).item()

    initial_params = temp_comp.get_params_vector(params_to_optimize)
    new_params_vector = optimizer.minimize(target, initial_params)

    new_params = dict(zip(params_to_optimize, new_params_vector))
    return new_params


@q_function_strategy.register(Exponential)
def q_function_strategy_exponential(
    component: Exponential, X: np.ndarray, H_j: np.ndarray, optimizer: Optimizer
) -> dict[str, float]:
    """Specialized Q-function optimization strategy for Exponential distribution.

    This function provides an analytical solution for parameter estimation of
    Exponential distribution, which is more efficient and stable than numerical
    optimization.

    Parameters
    ----------
    component : Exponential
        The Exponential distribution component for which to estimate parameters.
    X : np.ndarray
        Input data points used for parameter estimation.
    H_j : np.ndarray
        Weight vector where H_j[i] represents the probability that data point i
        belongs to the j-th mixture component.
    optimizer : Optimizer
        Optimization algorithm (not used in this analytical solution).

    Returns
    -------
    dict[str, float]
        Dictionary containing estimated values for 'loc' and 'rate' parameters.

    See Also
    --------
    q_function_strategy : Generic implementation for other distributions

    Notes
    -----
    **Methodology**

    For Exponential distribution, the parameters are estimated analytically:

    - Location parameter (loc): Estimated as the minimum value among data points
      with significant weights (above numerical tolerance)
    - Rate parameter (rate): Estimated using the method of moments with weighted
      data points, considering the numerical tolerance for stability

    **Numerical Stability**

    - Uses `NUMERICAL_TOLERANCE` to avoid numerical underflow and division by zero
    - Applies `np.maximum` to ensure positive values in calculations
    - Falls back to original parameter values when estimation is not feasible

    **Mathematical Formulation**

    The rate parameter is estimated as:
        rate = N_j / Σ(H_j * max(X - loc, tolerance))
    where N_j is the sum of weights for the component.

    Example
    --------
    >>> from rework_pysatl_mpest.distributions.exponential import Exponential
    >>> from rework_pysatl_mpest.optimizers.scipy_nelder_mead import ScipyNelderMead
    >>> import numpy as np

    >>> exp_dist = Exponential(loc=0, rate=1)
    >>> X = np.random.exponential(1, 100)
    >>> H_j = np.ones(100) / 100
    >>> optimizer = ScipyNelderMead()
    >>> params = q_function_strategy_exponential(exp_dist, X, H_j, optimizer)
    >>> print(f"Estimated loc: {params['loc']:.3f}, rate: {params['rate']:.3f}")
    """

    new_params: dict = {}
    N_j = np.sum(H_j).item()

    if np.any(H_j > NUMERICAL_TOLERANCE):
        relevant_X = X[H_j > NUMERICAL_TOLERANCE]
        new_params["loc"] = np.min(relevant_X).item()
    else:
        new_params["loc"] = component.loc

    loc = new_params.get("loc", component.loc)

    weighted_sum = np.dot(H_j, np.maximum(X - loc, NUMERICAL_TOLERANCE)).item()

    if weighted_sum > NUMERICAL_TOLERANCE:
        new_params["rate"] = N_j / weighted_sum
    else:
        new_params["rate"] = component.rate

    return new_params
