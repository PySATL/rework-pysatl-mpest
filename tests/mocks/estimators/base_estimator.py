"""Mock implementation for BaseEstimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from copy import copy

from numpy.typing import ArrayLike
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.base_estimator import BaseEstimator


class MockBaseEstimator(BaseEstimator):
    """Minimal mock returning an unmodified copy of the mixture."""

    def fit(self, X: ArrayLike, mixture: MixtureModel) -> MixtureModel:
        return copy(mixture)
