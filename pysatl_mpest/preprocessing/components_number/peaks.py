"""Module which contains Peaks Method"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from math import ceil

import numpy as np
from rework_pysatl_mpest.preprocessing.components_number.abstract_estimator import AComponentsNumber
from scipy.signal import find_peaks


class Peaks(AComponentsNumber):
    """Peaks Method with Empirical Density"""

    @property
    def name(self) -> str:
        return "Peaks"

    def estimate(self, X: np.ndarray) -> int:
        """
        Doanes fromula to determinate numbers of bins
        #  nbins = 1 + log2(n) + log2(1 + |skewness| / sg1)
        #  sg1 = √(6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0)))
        """

        n = X.size
        sg1 = np.sqrt(6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0)))
        skew = np.mean(((X - np.mean(X)) / np.std(X)) ** 3)

        nbins = ceil(1 + np.log2(n) + np.log2(1 + abs(skew) / sg1))

        #  Emperical Density
        hist = np.histogram(X, bins=nbins, density=True)[0]
        hist = np.concatenate((np.zeros(1), hist, np.zeros(1)))

        #  Peaks counting
        peaks, _ = find_peaks(hist)
        peaks_count = len(peaks)
        return peaks_count
