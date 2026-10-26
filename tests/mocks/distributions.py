"""
Mock distributions for testing purposes.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pysatl_core.types import CharacteristicName


class MockFamily:
    """
    Stands in for a pysatl-core family: it is a plain family, not a view with fixed parameters.
    """


@dataclass(frozen=True)
class MockParametrization:
    """
    Stands in for a pysatl-core parametrization; the family is reachable through ``__family__``.

    Parameters
    ----------
    param1 : float
        A dummy parameter for testing.
    """

    param1: float
    __family__: ClassVar[MockFamily] = MockFamily()


class MockContinuousDistribution:
    """
    Mock continuous distribution for testing.

    It has the attributes that the internal component helpers read from a pysatl-core distribution.
    Its quantile function is the identity, so sampling yields uniform values on [0, 1).

    Parameters
    ----------
    param1 : float, default=0.0
        A dummy parameter for testing.
    """

    def __init__(self, param1: float = 0.0) -> None:
        self.family_name = "MockFamily"
        self.parametrization_name = "MockParam"
        self.parametrization = MockParametrization(param1)
        self.parameters = {"param1": param1}

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
        if name == CharacteristicName.PPF:
            return X_arr
        raise NotImplementedError


class MockConstantLpdfContinuousDistribution(MockContinuousDistribution):
    """
    Mock continuous distribution with a constant log-PDF for testing.

    Parameters
    ----------
    log_density : float
        The value returned as the log-PDF at every point.
    """

    def __init__(self, log_density: float) -> None:
        super().__init__(param1=log_density)
        self.log_density = log_density

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
            The calculated characteristic, returning ``log_density`` for LPDF.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if name == CharacteristicName.LPDF:
            return np.full_like(X_arr, self.log_density)
        raise NotImplementedError


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
