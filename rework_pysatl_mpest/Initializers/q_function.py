from copy import deepcopy
from functools import singledispatch

import numpy as np

from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.distributions.exponential import Exponential
from rework_pysatl_mpest.optimizers.optimizer import Optimizer

NUMERICAL_TOLERANCE = 0.33

@singledispatch
def q_function_strategy(
        component: ContinuousDistribution,
        X: np.ndarray,
        H_j: np.ndarray,
        optimizer: Optimizer
) -> dict[str, float]:
    params_to_optimize = sorted(list(component.params_to_optimize))
    temp_comp = deepcopy(component)

    def target(vector_params):
        temp_comp.set_params_from_vector(params_to_optimize, vector_params)
        return -temp_comp.q_function(X, H_j)

    initial_params = temp_comp.get_params_vector(params_to_optimize)
    new_params_vector = optimizer.minimize(target, initial_params)

    new_params = dict(zip(params_to_optimize, new_params_vector))
    return new_params


@q_function_strategy.register(Exponential)
def _(
        component: Exponential,
        X: np.ndarray,
        H_j: np.ndarray,
        optimizer: Optimizer
) -> dict[str, float]:
    new_params = {}
    N_j = np.sum(H_j).item()

    if Exponential.PARAM_LOC in component.params_to_optimize:
        if np.any(H_j > NUMERICAL_TOLERANCE):
            relevant_X = X[H_j > NUMERICAL_TOLERANCE]
            new_params[Exponential.PARAM_LOC] = np.min(relevant_X).item()
        else:
            new_params[Exponential.PARAM_LOC] = component.loc

    if Exponential.PARAM_RATE in component.params_to_optimize:
        loc = new_params.get(Exponential.PARAM_LOC, component.loc)

        weighted_sum = np.dot(H_j, np.maximum(X - loc, NUMERICAL_TOLERANCE)).item()

        if weighted_sum > NUMERICAL_TOLERANCE:
            new_params[Exponential.PARAM_RATE] = N_j / weighted_sum
        else:
            new_params[Exponential.PARAM_RATE] = component.rate

    return new_params
