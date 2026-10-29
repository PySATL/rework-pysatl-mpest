"""Abstract base class for numerical optimizers."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from collections.abc import Callable


class Optimizer(ABC):
    """Abstract base class for numerical optimizers.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        minimize
    """

    @abstractmethod
    def minimize(self, target: Callable, params: list[float]) -> list[float]:
        """Find parameters that minimize a scalar objective function.

        Parameters
        ----------
        target : Callable
            Objective function accepting parameter values and returning a scalar.
        params : list[float]
            Initial parameter guess.

        Returns
        -------
        list[float]
            Optimized parameter values.
        """
