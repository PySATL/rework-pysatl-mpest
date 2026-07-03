"""Mock implementation for Optimizer."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

from pysatl_mpest.optimizers.optimizer import Optimizer


class MockOptimizer(Optimizer):
    """A minimal mock implementation of Optimizer.

    This mock simply returns the input parameters without performing
    any actual optimization.
    """

    def minimize(self, target: Callable, params: list[float]) -> list[float]:
        """Returns the input parameters unchanged.

        Parameters
        ----------
        target : Callable
            The objective function to minimize (ignored).
        params : list[float]
            A list of initial values for the parameters.

        Returns
        -------
        list[float]
            The unchanged input parameters.
        """

        return params
