"""Module which contains histogram features for a mixture classifier"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from rework_pysatl_mpest.preprocessing.components_family.criterions.abstract_criterion import (
    AHistClassifierCriterion,
)
from scipy.ndimage import sobel
from scipy.spatial.distance import jensenshannon


class CHistEnergy(AHistClassifierCriterion):
    @property
    def name(self) -> str:
        return "Hist Energy Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)
        return np.sum(hist**2)


class CHistEntropy(AHistClassifierCriterion):
    def __init__(self, noise: float = 10**-12) -> None:
        self.noise = noise

    @property
    def name(self) -> str:
        return "Hist Entropy Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)
        return -np.sum(hist * np.log2(hist + self.noise))


class CHistFlat(AHistClassifierCriterion):
    def __init__(self, rate: float = 0.05) -> None:
        self.rate = rate

    @property
    def name(self) -> str:
        return "Hist Flatness Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)
        return np.mean(np.abs(np.diff(hist)) < self.rate)


class CHistLength(AHistClassifierCriterion):
    @property
    def name(self) -> str:
        return "Hist Length Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)
        return np.sum(np.abs(np.diff(hist)))


class CHistUniform(AHistClassifierCriterion):
    @property
    def name(self) -> str:
        return "Hist Uniform Criterion"

    def score(self, hist: np.ndarray) -> float:
        n = len(hist)
        hist = hist / np.sum(hist)

        uniform = np.ones(n) / n
        return jensenshannon(hist, uniform)


class CSobelCount(AHistClassifierCriterion):
    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "Sobel Count Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        sob = sobel(hist)
        return np.mean(np.abs(sob) > np.max(np.abs(sob)) * self.threshold)


class CSobelMax(AHistClassifierCriterion):
    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "Sobel Max Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        sob = sobel(hist)
        return np.max(np.abs(sob))


class CSobelMean(AHistClassifierCriterion):
    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "Sobel Mean Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        sob = sobel(hist)
        return np.mean(np.abs(sob))


class CSobelMin(AHistClassifierCriterion):
    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "Sobel Min Criterion"

    def score(self, hist: np.ndarray) -> float:
        hist = hist / np.sum(hist)

        sob = sobel(hist)
        return np.min(np.abs(sob))
