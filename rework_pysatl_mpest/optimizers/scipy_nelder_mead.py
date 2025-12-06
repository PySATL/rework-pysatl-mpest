"""A module that provides a Nelder-Mead optimizer using the SciPy library."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable

from scipy.optimize import minimize

from .optimizer import Optimizer


class ScipyNelderMead(Optimizer):
    """An optimizer that uses the Nelder-Mead simplex algorithm from SciPy.

    This class serves as a wrapper for the `scipy.optimize.minimize` function,
    specifically configured to use the 'Nelder-Mead' method. The Nelder-Mead
    algorithm is a direct search method that does not require gradient
    information, making it suitable for non-differentiable or noisy objective
    functions.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        minimize
    """

    def minimize(self, target: Callable, params: list[float]) -> list[float]:
        """Minimizes a target function using the Nelder-Mead algorithm.

        This method leverages the `scipy.optimize.minimize` function to find
        the parameters that minimize the provided objective function.

        Parameters
        ----------
        target : Callable
            The objective function to minimize. It must be a callable that
            accepts a list or NumPy array of parameters and returns a single
            scalar value.
        params : list[float]
            A list of initial values for the parameters that serves as the
            starting point for the optimization.

        Returns
        -------
        list[float]
            A list containing the set of parameters that minimizes the target
            function, as found by the Nelder-Mead algorithm.
        """

        return list(minimize(target, params, method="Nelder-Mead").x)
