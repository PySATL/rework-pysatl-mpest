"""Module which contains sample features for a mixture classifier"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from rework_pysatl_mpest.preprocessing.components_family.criterions.abstract_criterion import (
    ASampleRecognitionCriterion,
)
from scipy.stats import gmean, iqr, kurtosis, median_abs_deviation, skew, zscore


class CMedian(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Median Criterion"

    def score(self, X: np.ndarray) -> float:
        return np.median(X)


class CMean(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Mean Criterion"

    def score(self, X: np.ndarray) -> float:
        return np.mean(X)


class CMad(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "MAD Criterion"

    def score(self, X: np.ndarray) -> float:
        return median_abs_deviation(X)


class CNegativeValue(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Negative Value Criterion"

    def score(self, X: np.ndarray) -> float:
        return float(np.min(X) < 0)


class CKurt(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Kurtosis Criterion"

    def score(self, X: np.ndarray) -> float:
        result = kurtosis(X)
        return kurtosis(X) if not np.isnan(result) else 0


class CIqr(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "IQR Criterion"

    def score(self, X: np.ndarray) -> float:
        return iqr(X) / np.median(X)


class CKurtMoors(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Kurtosis Moors Criterion"

    def score(self, X: np.ndarray) -> float:
        p1, p2, p3, p4, p5, p6 = np.percentile(X, [12.5, 25, 37.5, 62.5, 75, 87.5])
        result = ((p6 - p2) + (p5 - p3)) / (p4 - p2)
        return result if not np.isnan(result) else 0


class CLogRatio(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Log Extreme Ratio Criterion"

    def score(self, X: np.ndarray) -> float:
        xmin, xmax = np.min(X), np.max(X)
        median = np.median(X)
        result = np.log((xmax - median) / (median - xmin))
        return result if not np.isnan(result) else 0


class COutlierFraction(ASampleRecognitionCriterion):
    def __init__(self, k: float = 3) -> None:
        self.k = k

    @property
    def name(self) -> str:
        return "Outlier Fraction Criterion"

    def score(self, X: np.ndarray) -> float:
        mu, sigma = np.mean(X), np.std(X)
        return np.mean(np.abs(X - mu) > self.k * sigma)


class CRange(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Range Criterion"

    def score(self, X: np.ndarray) -> float:
        return np.ptp(X)


class CSkew(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Skewness Criterion"

    def score(self, X: np.ndarray) -> float:
        result = skew(X)
        return result if not np.isnan(result) else 0


class CSkewBowley(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Skewness Bowley Criterion"

    def score(self, X: np.ndarray) -> float:
        p25, p50, p75 = np.percentile(X, [25, 50, 75])
        result = (p75 + p25 - 2 * p50) / (p75 - p25)
        return result if not np.isnan(result) else 0


class CSpacingGap(ASampleRecognitionCriterion):
    def __init__(self, rate: float = 5) -> None:
        self.rate = rate

    @property
    def name(self) -> str:
        return "Spacing Gap Criterion"

    def score(self, X: np.ndarray) -> float:
        diff = np.diff(np.sort(X))
        dmedian = np.median(diff)
        return np.mean(diff > self.rate * dmedian)


class CSpacingGini(ASampleRecognitionCriterion):
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


class CSpacingVar(ASampleRecognitionCriterion):
    def __init__(self, n_boot: int = 200) -> None:
        self.n_boot = n_boot

    @property
    def name(self) -> str:
        return "Spacing Var Criterion"

    def score(self, X: np.ndarray) -> float:
        diff = np.diff(np.sort(X))
        return np.var(diff)


class CStd(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Standard Deviation Criterion"

    def score(self, X: np.ndarray) -> float:
        return np.std(X)


class CMaxZscore(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Z-Score Criterion"

    def score(self, X: np.ndarray) -> float:
        result = np.max(np.abs(zscore(X)))
        return result if not np.isnan(result) else 0


class CPercentileMedian(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Percentile Median Criterion"

    def score(self, X: np.ndarray) -> float:
        result = (np.percentile(X, 75) - np.median(X)) / (np.median(X) - np.percentile(X, 25))
        return result if not np.isnan(result) else 0


class CPercentileRange(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Percentile Range Criterion"

    def score(self, X: np.ndarray) -> float:
        return np.percentile(X, 95) - np.percentile(X, 5)


class CPercentileTail(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Percentile Tail Criterion"

    def score(self, X: np.ndarray) -> float:
        median = np.median(X)
        return (np.percentile(X, 99) - median) - (median - np.percentile(X, 1))


class CPercentileExtreme(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Percentile Extreme Criterion"

    def score(self, X: np.ndarray) -> float:
        median = np.median(X)
        return (np.percentile(X, 99.9) - median) - (np.percentile(X, 99) - median)


class CBootKurt(ASampleRecognitionCriterion):
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


class CBootMean(ASampleRecognitionCriterion):
    def __init__(self, n_boot: int = 200, state: int | None = None) -> None:
        self.n_boot = n_boot
        self.state = state

    @property
    def name(self) -> str:
        return "Bootstrap Mean Criterion"

    def score(self, X: np.ndarray) -> float:
        np.random.seed(self.state)

        n = len(X)
        means = [np.mean(np.random.choice(X, size=n, replace=True)) for _ in range(self.n_boot)]
        return np.var(means)


class CBootVar(ASampleRecognitionCriterion):
    def __init__(self, n_boot: int = 200, state: int | None = None) -> None:
        self.n_boot = n_boot
        self.state = state

    @property
    def name(self) -> str:
        return "Bootstrap Var Criterion"

    def score(self, X: np.ndarray) -> float:
        np.random.seed(self.state)

        n = len(X)
        means = [np.var(np.random.choice(X, size=n, replace=True)) for _ in range(self.n_boot)]
        return np.var(means)


class CGmean(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Geometric Mean Criterion"

    def score(self, X: np.ndarray) -> float:
        return gmean(np.abs(X))


class CHillAbs(ASampleRecognitionCriterion):
    @property
    def name(self) -> str:
        return "Hill Abs Criterion"

    def score(self, X: np.ndarray) -> float:
        X = np.sort(np.abs(X))
        k = int(len(X) ** 0.5)

        x_tail = X[-k:]
        x_min = X[-k - 1]
        return (1 / k) * np.sum(np.log(x_tail) - np.log(x_min))
