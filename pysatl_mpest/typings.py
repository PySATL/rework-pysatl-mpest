"""A module that provides custom types for the project."""

__author__ = "Aleksandra Ri, Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy._typing import ArrayLike

DType = TypeVar("DType", bound=np.floating)


@runtime_checkable
class HardClusterizer(Protocol):
    def fit_predict(self, X: ArrayLike, y: Any = None) -> np.ndarray: ...


@runtime_checkable
class SoftClusterizer(Protocol):
    def fit_transform(self, X: ArrayLike, y: Any = None) -> np.ndarray: ...


Clusterizer = HardClusterizer | SoftClusterizer
