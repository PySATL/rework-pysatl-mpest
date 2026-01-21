"""Module which contains collector of a vector of criterions for mixture classifier"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from math import ceil

import numpy as np
from rework_pysatl_mpest.preprocessing.components_family.criterions import base_criterions
from rework_pysatl_mpest.preprocessing.components_family.criterions.abstract_criterion import (
    AHistClassifierCriterion,
    APeaksClassifierCriterion,
    ASampleClassifierCriterion,
)
from scipy.stats import iqr


class MixtureClassifierCriterions:
    """
    MixtureClassifierCriterions

    Parameters
    ----------
    :criterions list[ASampleClassifierCriterion | APeaksClassifierCriterion | AHistClassifierCriterion]

    — List of criterions for the mixture classifiers
    """

    def __init__(
        self,
        criterions: list[
            ASampleClassifierCriterion | APeaksClassifierCriterion | AHistClassifierCriterion
        ] = base_criterions,
    ) -> None:
        self.criterions = criterions

    @staticmethod
    def _get_hist(X: np.ndarray) -> np.ndarray:
        """A function for constructing a histogram with constraints"""
        n = X.size
        bmin = 20
        bmax = 150

        h = 1 * iqr(X) * n ** (-1 / 3)
        bins = ceil((X.max() - X.min()) / h) if h > 0 else bmin
        nbins = max(bmin, min(bins, bmax))

        hist = np.histogram(X, bins=nbins, density=True)[0]

        return hist

    @staticmethod
    def _get_criterion(
        X: np.ndarray,
        hist: np.ndarray,
        criterion: (ASampleClassifierCriterion | APeaksClassifierCriterion | AHistClassifierCriterion),
    ) -> float:
        """Function for obtaining a single criterion based on a sample"""

        warnings.filterwarnings("ignore")

        if isinstance(criterion, ASampleClassifierCriterion):
            return criterion.score(X)

        return criterion.score(hist)

    def get_criterions(self, X: np.ndarray) -> dict[str, float]:
        """Function for evaluating a feature vector based on a sample"""

        hist_list = self._get_hist(X)
        return dict([(criterion.name, self._get_criterion(X, hist_list, criterion)) for criterion in self.criterions])
