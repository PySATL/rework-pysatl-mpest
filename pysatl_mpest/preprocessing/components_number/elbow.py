"""Module which contains Elbow Method"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from kneed import KneeLocator
from pysatl_mpest.preprocessing.components_number.abstract_estimator import AComponentsNumber
from sklearn.cluster import KMeans


class Elbow(AComponentsNumber):
    """
    Elbow method with KMeans++

    Parameters
    -----
    :kmax:          int                       — Assumed maximum number of components
    :k_init:        int         default: 1    — Number of times the KMeans is run
    :k_max_iter:    int         default: 300  — Maximum number of iterations in KMeans
    :random_state:  int | None  default: None — Determines random generation for KMeans
    """

    def __init__(
        self,
        kmax: int,
        k_init: int = 1,
        k_max_iter: int = 300,
        random_state: int | None = None,
    ) -> None:
        self.kmax = kmax
        self.k_init = k_init
        self.k_max_iter = k_max_iter
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "Elbow"

    def estimate(self, X: np.ndarray) -> int:
        X = X.reshape(-1, 1)
        k_range = range(1, self.kmax + 2)
        wcss = []

        for k in k_range:
            kmeans_elbow = KMeans(
                max_iter=self.k_max_iter,
                n_clusters=k,
                init="k-means++",
                n_init=self.k_init,
                random_state=self.random_state,
            ).fit(X)
            wcss.append(kmeans_elbow.inertia_)

        knee = KneeLocator(k_range, wcss, curve="convex", direction="decreasing").elbow

        return knee
