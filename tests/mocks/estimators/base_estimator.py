"""Mock implementation for BaseEstimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from copy import copy

from numpy.typing import ArrayLike
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.base_estimator import BaseEstimator


class MockBaseEstimator(BaseEstimator):
    """A minimal mock implementation of BaseEstimator.

    This mock simply returns a copy of the input mixture model without performing
    any actual estimation. It is used to verify the abstract interface signature.
    """

    def fit(self, X: ArrayLike, mixture: MixtureModel) -> MixtureModel:
        """Returns a copy of the mixture model without modifications.

        Parameters
        ----------
        X : ArrayLike
            The input dataset (ignored).
        mixture : MixtureModel
            The initial mixture model.

        Returns
        -------
        MixtureModel
            A copy of the input mixture model.
        """

        return copy(mixture)
