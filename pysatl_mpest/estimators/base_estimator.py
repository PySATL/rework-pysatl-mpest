"""A module that provides an abstract class for implementing custom estimators."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import Generic

from numpy.typing import ArrayLike

from ..core import MixtureModel
from ..typings import DType


class BaseEstimator(ABC, Generic[DType]):
    """Abstract class for a mixture model parameter estimator.

    This class defines the interface for all estimator algorithms. Estimators are responsible for
    fitting the parameters of a :class:`~pysatl_mpest.core.MixtureModel` to a given dataset.

    Methods
    -------
    **Abstract methods**

    .. autosummary::
        :toctree: generated/

        fit

    Notes
    -----
    **Implementation Requirements**

    Subclasses must implement the abstract method :meth:`fit` to provide a specific
    estimation strategy.
    """

    @abstractmethod
    def fit(self, X: ArrayLike, mixture: MixtureModel[DType]) -> MixtureModel[DType]:
        """Fits the mixture model to the provided data.

        This method estimates the parameters of the model's components and their
        corresponding weights based on the input data sample.

        Parameters
        ----------
        X : ArrayLike
            The input data sample for fitting the model.
        mixture : MixtureModel[DType]
            The initial mixture model to be fitted.

        Returns
        -------
        MixtureModel[DType]
            The mixture model with estimated parameters.
        """
