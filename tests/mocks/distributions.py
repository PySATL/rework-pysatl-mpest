"""
Mock distributions for testing purposes.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pysatl_core.types import CharacteristicName


class MockContinuousDistribution:
    """
    Mock continuous distribution for testing.

    Parameters
    ----------
    param1 : float, default=0.0
        A dummy parameter for testing.
    """

    def __init__(self, param1: float = 0.0) -> None:
        self.family_name = "MockFamily"
        self.parametrization_name = "MockParam"
        self.parametrization = param1

    def calculate_characteristic(self, name: CharacteristicName, X: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate a characteristic of the distribution.

        Parameters
        ----------
        name : CharacteristicName
            The name of the characteristic to calculate.
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[np.float64]
            The calculated characteristic.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if name == CharacteristicName.LPDF:
            return np.zeros_like(X_arr)
        raise NotImplementedError

    def sample(self, n: int, random_state: np.random.Generator | None = None) -> NDArray[np.float64]:
        """
        Generate random samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        random_state : np.random.Generator or None, default=None
            Random number generator instance.

        Returns
        -------
        NDArray[np.float64]
            Generated samples.
        """
        if random_state is None:
            random_state = np.random.default_rng()
        return random_state.normal(size=n)


class MockInfLpdfContinuousDistribution(MockContinuousDistribution):
    """
    Mock continuous distribution with infinite log-PDF for testing.

    Parameters
    ----------
    param1 : float, default=0.0
        A dummy parameter for testing.
    """

    def calculate_characteristic(self, name: CharacteristicName, X: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate a characteristic of the distribution.

        Parameters
        ----------
        name : CharacteristicName
            The name of the characteristic to calculate.
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[np.float64]
            The calculated characteristic, returning -inf for LPDF.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if name == CharacteristicName.LPDF:
            return np.full_like(X_arr, -np.inf)
        raise NotImplementedError
