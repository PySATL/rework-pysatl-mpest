"""A module that provides a Powell optimizer using the SciPy library."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from ..typings import FloatingType
from .optimizer import Optimizer


class ScipyPowell[FloatT: FloatingType](Optimizer[FloatT]):
    """An optimizer that uses Powell's conjugate direction method from SciPy.

    This class serves as a wrapper for the `scipy.optimize.minimize` function,
    specifically configured to use the 'Powell' method. Powell's method is a
    conjugate direction algorithm that does not require the computation of
    gradients, making it suitable for objective functions where derivatives are
    not available or are difficult to compute.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        minimize
    """

    def minimize(self, target: Callable, params: list[FloatT]) -> list[FloatT]:
        """Minimizes a target function using Powell's method.

        This method leverages the `scipy.optimize.minimize` function to find
        the parameters that minimize the provided objective function.

        Parameters
        ----------
        target : Callable
            The objective function to minimize. It must be a callable that
            accepts a list or NumPy array of parameters and returns a single
            scalar value.
        params : list[FloatT]
            A list of initial values for the parameters that serves as the
            starting point for the optimization.

        Returns
        -------
        list[FloatT]
            A list containing the set of parameters that minimizes the target
            function, as found by Powell's method.
        """

        dtype = params[0].dtype

        return list(np.asarray(minimize(target, params, method="Powell").x, dtype=dtype))
