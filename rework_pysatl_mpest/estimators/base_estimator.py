"""A module that provides an abstract class for implementing custom estimators."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

from rework_pysatl_mpest.core.mixture import MixtureModel


class BaseEstimator(ABC):
    """Abstract class of parameter estimator for a mixture of distributions.

    .. rubric:: Implementation Requirements

    Subclasses must:
        1. Implement abstract method `~fit` for estimating mixture parameters.
    """

    @abstractmethod
    def fit(self, X: ArrayLike, mixture: MixtureModel) -> MixtureModel:
        """Fits the mixture model to the provided data.

        This method takes an input data sample and a mixture model, then
        estimates the parameters of the model's components and their weights
        based on the data.

        Args:
            X (ArrayLike): The input data sample for fitting the model.
            mixture (MixtureModel): The initial mixture model to be fitted.

        Returns:
            MixtureModel: The mixture model with estimated parameters.
        """
