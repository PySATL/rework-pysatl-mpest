"""Module which contains method for initial estimation of mixture components family based on mixture classifier"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from rework_pysatl_mpest.preprocessing.components_family.mixture_classifiers.mixture_classifier import (
    MixtureClassifierModel,
)
from rework_pysatl_mpest.preprocessing.components_number.abstract_estimator import AComponentsNumber
from rework_pysatl_mpest.preprocessing.utils import Distribution


class ComponentsFamily:
    """
    ComponentsFamily

    Parameters
    ----------
    :model: MixtureClassifierModel  — Mixture Classifier Model
    :top_k: int                     — Top k most likely mixtures
    :components_number: int | None  — Method for estimating number of components
    :random_state: int | None       — Determines random generation for some criterions
    """

    def __init__(
        self,
        recognition_model: MixtureClassifierModel,
        top_k: int,
        components_number: AComponentsNumber | None = None,
        state: int | None = None,
    ) -> None:
        self.model = recognition_model
        self.top_k = top_k
        self.components_number = components_number
        self.state = state

    def predict(self, X: np.ndarray, k: int | list[int] | None = None) -> list[list[Distribution]]:
        """
        Function for evaluating the top k most probable configurations

        Parameters
        ----------
        :X: np.ndarray             — Sample Data
        :k: int | list[int] | None —  The set number of components of the mixture

        k is a specific number, or a number in a specified range, or None
        (to determine the number of components using a specified method,
        or if no method is specified, to use the entire range from 1 to 10 components)

        Returns
        ----------
        list[list[Distribution]]

        — List of mixture configurations using distribution classes for further work with the mixture
        """

        def __get_components_n(k: None | int | list[int]) -> list[int]:
            """Function that defines the boundaries of the possible number of mixture components"""
            upper_bound = 10

            if isinstance(k, int):
                return [k]

            if isinstance(k, list):
                return k

            if isinstance(self.components_number, AComponentsNumber):
                comp_k = self.components_number.estimate(X)
                return [max(comp_k - 1, 1), comp_k, min(comp_k + 1, upper_bound)]

            return [i for i in range(1, upper_bound + 1)]

        np.random.seed(self.state)

        n = __get_components_n(k)
        prob = self.model.predict(X)
        result: list[list[Distribution]] = []

        for i in np.argsort(prob)[::-1]:
            if len(result) == self.top_k:
                break

            components = self.model.transform(i)
            if len(components) not in n:
                continue

            result.append(components)

        return result
