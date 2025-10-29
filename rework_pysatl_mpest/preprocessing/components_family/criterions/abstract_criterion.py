"""Module which contains abstract classes of mixture classifier criterions"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import find_peaks


class ASampleRecognitionCriterion(ABC):
    """Abstract class of sample feature for mixture classifier"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name getter"""

    @abstractmethod
    def score(self, X: np.ndarray) -> float:
        """Function evaluating sample feature for a mixture classifier"""


class APeaksRecognitionCriterion(ABC):
    """Abstract class of peaks feature for mixture classifier"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name getter"""

    @staticmethod
    def _get_peaks(hist: np.ndarray, is_valleys: bool = False) -> list[np.ndarray]:
        hist_prep = np.concatenate((np.zeros(1), hist, np.zeros(1)))
        if not is_valleys:
            peaks, _ = find_peaks(hist_prep)
            return [hist_prep, peaks]

        valleys, _ = find_peaks(-hist_prep)
        return [hist_prep, valleys]

    @abstractmethod
    def score(self, hist: np.ndarray) -> float:
        """Function evaluating peaks feature for a mixture classifier"""


class AHistRecognitionCriterion(ABC):
    """Abstract class of hist feature for mixture classifier"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name getter"""

    @abstractmethod
    def score(self, hist: np.ndarray) -> float:
        """Function evaluating hist feature for a mixture classifier"""
