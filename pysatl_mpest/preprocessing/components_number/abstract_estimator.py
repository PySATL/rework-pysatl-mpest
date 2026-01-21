"""Module which contains abstract class for methods estimating number of components in mixture"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

import numpy as np


class AComponentsNumber(ABC):
    """Abstract class for methods estimating number of components in mixture"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name getter"""

    @abstractmethod
    def estimate(self, X: np.ndarray) -> int:
        """The function for estimating number of components"""
