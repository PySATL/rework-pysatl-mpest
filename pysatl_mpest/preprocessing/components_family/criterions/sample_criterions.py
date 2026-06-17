"""Module which contains sample features for a mixture classifier"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from pysatl_mpest.preprocessing.components_family.criterions.abstract_criterion import (
    ASampleClassifierCriterion,
)
from scipy.stats import iqr, kurtosis, skew, zscore


class CKurt(ASampleClassifierCriterion):
    @property
    def name(self) -> str:
        return "Kurtosis Criterion"

    def score(self, X: np.ndarray) -> float:
        result = kurtosis(X)
        return kurtosis(X) if not np.isnan(result) else 0


class CNegativeValue(ASampleClassifierCriterion):
    @property
    def name(self) -> str:
        return "Negative Value Criterion"

    def score(self, X: np.ndarray) -> float:
        return float(np.min(X) < 0)


class CIqr(ASampleClassifierCriterion):
    @property
    def name(self) -> str:
        return "IQR Criterion"

    def score(self, X: np.ndarray) -> float:
        return iqr(X) / np.median(X)


class CKurtMoors(ASampleClassifierCriterion):
    @property
    def name(self) -> str:
        return "Kurtosis Moors Criterion"

    def score(self, X: np.ndarray) -> float:
        p1, p2, p3, p4, p5, p6 = np.percentile(X, [12.5, 25, 37.5, 62.5, 75, 87.5])
        result = ((p6 - p2) + (p5 - p3)) / (p4 - p2)
        return result if not np.isnan(result) else 0


class CLogRatio(ASampleClassifierCriterion):
    @property
    def name(self) -> str:
        return "Log Extreme Ratio Criterion"

    def score(self, X: np.ndarray) -> float:
        xmin, xmax = np.min(X), np.max(X)
        median = np.median(X)
        result = np.log((xmax - median) / (median - xmin))
        return result if not np.isnan(result) else 0


class COutlierFraction(ASampleClassifierCriterion):
    def __init__(self, k: float = 3) -> None:
        self.k = k

    @property
    def name(self) -> str:
        return "Outlier Fraction Criterion"

    def score(self, X: np.ndarray) -> float:
        mu, sigma = np.mean(X), np.std(X)
        return np.mean(np.abs(X - mu) > self.k * sigma)


class CRange(ASampleClassifierCriterion):
    @property
    def name(self) -> str:
        return "Range Criterion"

    def score(self, X: np.ndarray) -> float:
        return np.ptp(X)


class CSkew(ASampleClassifierCriterion):
    @property
    def name(self) -> str:
        return "Skewness Criterion"

    def score(self, X: np.ndarray) -> float:
        result = skew(X)
        return result if not np.isnan(result) else 0


class CSkewBowley(ASampleClassifierCriterion):
    @property
    def name(self) -> str:
        return "Skewness Bowley Criterion"

    def score(self, X: np.ndarray) -> float:
        p25, p50, p75 = np.percentile(X, [25, 50, 75])
        result = (p75 + p25 - 2 * p50) / (p75 - p25)
        return result if not np.isnan(result) else 0


class CSpacingGap(ASampleClassifierCriterion):
    def __init__(self, rate: float = 5) -> None:
        self.rate = rate

    @property
    def name(self) -> str:
        return "Spacing Gap Criterion"

    def score(self, X: np.ndarray) -> float:
        diff = np.diff(np.sort(X))
        dmedian = np.median(diff)
        return np.mean(diff > self.rate * dmedian)


class CSpacingGini(ASampleClassifierCriterion):
    @property
    def name(self) -> str:
        return "Spacing Gini Criterion"

    def score(self, X: np.ndarray) -> float:
        diff = np.diff(np.sort(X))
        diff = np.sort(diff)
        n = len(diff)
        index = np.arange(1, n + 1)
        result = np.sum((2 * index - n - 1) * diff) / (np.sum(diff) * n)
        return result if not np.isnan(result) else 0


class CMaxZscore(ASampleClassifierCriterion):
    @property
    def name(self) -> str:
        return "Z-Score Criterion"

    def score(self, X: np.ndarray) -> float:
        result = np.max(np.abs(zscore(X)))
        return result if not np.isnan(result) else 0


class CBootKurt(ASampleClassifierCriterion):
    def __init__(self, n_boot: int = 200, state: int | None = None) -> None:
        self.n_boot = n_boot
        self.state = state

    @property
    def name(self) -> str:
        return "Bootstrap Kurtosis Criterion"

    def score(self, X: np.ndarray) -> float:
        np.random.seed(self.state)

        n = len(X)
        means = [kurtosis(np.random.choice(X, size=n, replace=True)) for _ in range(self.n_boot)]
        result = np.var(means)
        return result if not np.isnan(result) else 0


class CHillAbs(ASampleClassifierCriterion):
    @property
    def name(self) -> str:
        return "Hill Abs Criterion"

    def score(self, X: np.ndarray) -> float:
        X = np.sort(np.abs(X))
        k = int(len(X) ** 0.5)

        x_tail = X[-k:]
        x_min = X[-k - 1]
        return (1 / k) * np.sum(np.log(x_tail) - np.log(x_min))
