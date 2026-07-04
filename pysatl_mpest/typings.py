"""A module that provides custom types for the project."""

__author__ = "Aleksandra Ri, Viktor Khanukaev, Totmyanin Danil"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy._typing import ArrayLike

Scalar = int | float | np.floating | np.integer
BoolScalar = bool | np.bool_

type UnivariateFloatArray = np.ndarray[tuple[int], np.dtype[np.float64]]
type MultivariateFloatArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

type UnivariateIntArray = np.ndarray[tuple[int], np.dtype[np.integer]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]

type BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]


@runtime_checkable
class HardClusterizer(Protocol):
    def fit_predict(self, X: ArrayLike, y: Any = None) -> np.ndarray: ...


@runtime_checkable
class SoftClusterizer(Protocol):
    def fit_transform(self, X: ArrayLike, y: Any = None) -> np.ndarray: ...


Clusterizer = HardClusterizer | SoftClusterizer
