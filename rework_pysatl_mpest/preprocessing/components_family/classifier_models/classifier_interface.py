"""Module which contains interface of the classifier model"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

import numpy as np


class ClassifierInterface(ABC):
    """Class representing an interface for classification models"""

    def __init__(self) -> None:
        self.is_fitted = False

    @abstractmethod
    def _load_model(self, model_path: str) -> None:
        """An abstract method for implementing model loading"""

    @abstractmethod
    def predict(self, criterions: dict[str, float]) -> np.ndarray:
        """Abstract method for implementing a model prediction"""

    def load_model(self, model_path: str) -> None:
        self._load_model(model_path)
        self.is_fitted = True
