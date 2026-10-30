"""Mock distributions for testing purposes."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pysatl_core.types import CharacteristicName


class MockFamily:
    """Stands in for a pysatl-core family: plain family, not a view."""


@dataclass(frozen=True)
class MockParametrization:
    param1: float
    __family__: ClassVar[MockFamily] = MockFamily()


class MockContinuousDistribution:
    """Mock continuous distribution whose quantile function is identity (uniform [0, 1) sampling)."""

    def __init__(self, param1: float = 0.0) -> None:
        self.family_name = "MockFamily"
        self.parametrization_name = "MockParam"
        self.parametrization = MockParametrization(param1)
        self.parameters = {"param1": param1}

    def calculate_characteristic(self, name: CharacteristicName, X: ArrayLike) -> NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        if name == CharacteristicName.LPDF:
            return np.zeros_like(X_arr)
        if name == CharacteristicName.PPF:
            return X_arr
        raise NotImplementedError


class MockConstantLpdfContinuousDistribution(MockContinuousDistribution):
    """Mock distribution with constant log-PDF."""

    def __init__(self, log_density: float) -> None:
        super().__init__(param1=log_density)
        self.log_density = log_density

    def calculate_characteristic(self, name: CharacteristicName, X: ArrayLike) -> NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        if name == CharacteristicName.LPDF:
            return np.full_like(X_arr, self.log_density)
        raise NotImplementedError


class MockInfLpdfContinuousDistribution(MockContinuousDistribution):
    """Mock distribution with negative infinite log-PDF."""

    def calculate_characteristic(self, name: CharacteristicName, X: ArrayLike) -> NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        if name == CharacteristicName.LPDF:
            return np.full_like(X_arr, -np.inf)
        raise NotImplementedError
