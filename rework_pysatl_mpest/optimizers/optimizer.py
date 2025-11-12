"""A module that provides an abstract base class for optimizers.

This module defines the :class:`Optimizer` abstract class, which serves as a contract
for implementing various optimization algorithms. Any concrete optimizer should
inherit from this class and implement the :meth:`minimize` method.
"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from typing import Callable

from ..typings import DType


class Optimizer(ABC):
    """Abstract base class for numerical optimizers.

    This class defines the standard interface for all optimizer implementations.
    Its primary purpose is to find the parameters that minimize a given
    objective function, starting from an initial guess.

    Subclasses must implement the :meth:`minimize` method to provide a specific
    optimization algorithm.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        minimize
    """

    @abstractmethod
    def minimize(self, target: Callable, params: list[DType]) -> list[DType]:
        """Finds the parameters that minimize a target function.

        This abstract method must be implemented by subclasses to perform the
        actual optimization.

        Parameters
        ----------
        target : Callable
            The objective function to minimize. It must be a callable that
            accepts a list or NumPy array of parameters and returns a single
            scalar value.
        params : list[DType]
            A list of initial values for the parameters to be optimized. This
            serves as the starting point for the optimization algorithm.

        Returns
        -------
        list[Dtype]
            A list containing the set of parameters that minimizes the
            target function.
        """
