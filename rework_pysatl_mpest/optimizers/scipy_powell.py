from typing import Callable

from scipy.optimize import minimize

from rework_pysatl_mpest.optimizers.optimizer import Optimizer


class ScipyPowell(Optimizer):
    def minimize(self, target: Callable, params: list[float]) -> list[float]:
        return list(minimize(target, params, method="Powell").x)
