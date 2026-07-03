"""Mock implementation for BaseEstimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from copy import copy

from numpy.typing import ArrayLike
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.base_estimator import BaseEstimator
from pysatl_mpest.typings import FloatingType


class MockBaseEstimator[FloatT: FloatingType](BaseEstimator[FloatT]):
    """A minimal mock implementation of BaseEstimator.

    This mock simply returns a copy of the input mixture model without performing
    any actual estimation. It is used to verify the abstract interface signature.
    """

    def fit(self, X: ArrayLike, mixture: MixtureModel[FloatT]) -> MixtureModel[FloatT]:
        """Returns a copy of the mixture model without modifications.

        Parameters
        ----------
        X : ArrayLike
            The input dataset (ignored).
        mixture : MixtureModel[FloatT]
            The initial mixture model.

        Returns
        -------
        MixtureModel[FloatT]
            A copy of the input mixture model.
        """

        return copy(mixture)
