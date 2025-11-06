"""Module which contains interface of the classifier model"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

import numpy as np


class IClassifier(ABC):
    """Class representing an interface for classification models"""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """A property indicating whether the model has been trained"""

    @abstractmethod
    def predict(self, criterions: dict[str, float]) -> np.ndarray:
        """Abstract method for implementing a model prediction"""

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """An abstract method for implementing model loading"""
