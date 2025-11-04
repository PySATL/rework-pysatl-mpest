"""
A module providing a cluster-based initializer for mixture models.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from rework_pysatl_mpest.core.mixture import MixtureModel
from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.initializers._estimation_strategies.q_function import q_function_strategy
from rework_pysatl_mpest.initializers.cluster_match_strategy import (
    match_clusters_for_models_akaike,
    match_clusters_for_models_log_likelihood,
)
from rework_pysatl_mpest.initializers.initializer import Initializer
from rework_pysatl_mpest.initializers.strategies import ClusterMatchStrategy, EstimationStrategy
from rework_pysatl_mpest.optimizers import Optimizer
from rework_pysatl_mpest.optimizers.scipy_nelder_mead import ScipyNelderMead


class ClusterizeInitializer(Initializer):
    """Cluster-based initializer for mixture model parameters.

    This initializer uses clustering algorithms to partition the data and then
    estimates initial parameters for mixture components based on the clustering results.
    Supports both hard clustering (crisp assignments) and soft clustering (fuzzy assignments).
    For homogeneous mixtures fast (is_accurate = False) initialization is recommended.

    Attributes
    ----------
    MIN_SAMPLES : int
        Minimum number of samples required for a cluster to be considered valid.
        Mapping of cluster matching strategies to their implementation functions.
    n_components : Optional[int]
        Number of mixture components to initialize.
    cluster_match_strategy : ClusterMatchStrategy
        Strategy for matching clusters to distribution models.
    estimation_strategies : list[EstimationStrategy]
        List of estimation strategies for each distribution model.
    models : list[ContinuousDistribution]
        List of distribution models to initialize.

    Parameters
    ----------
    is_accurate : bool
        If True, uses accurate initialization with optimal cluster-model matching.
        If False, uses fast initialization with direct cluster assignments.
    is_soft : bool
        If True, uses soft clustering (fuzzy assignments).
        If False, uses hard clustering (crisp assignments).
    clusterizer : Any
        The clustering algorithm instance. Must have `fit_transform` method for soft
        clustering or `fit_predict` method for hard clustering.

    Methods
    -------
    perform(X, dists, cluster_match_strategy, estimation_strategies, optimizer)
        Performs cluster-based initialization of mixture model parameters.

    Notes
    -----
    **Supported Clustering Types**

    - Soft clustering: Requires clusterizer with `fit_transform` method that returns
      a weight matrix where each element represents the probability of a data point
      belonging to a cluster.
    - Hard clustering: Requires clusterizer with `fit_predict` method that returns
      cluster labels for each data point.

    **Initialization Modes**

    - Accurate mode: Uses optimal cluster-model matching based on the specified strategy
      (likelihood or AIC e.t.c). Evaluates multiple assignments to find the best fit.
    - Fast mode: Directly assigns each cluster to a model in order, providing faster
      but potentially suboptimal initialization.

    **Error Handling**

    - Validates clusterizer compatibility with the specified clustering type.
    - Handles outliers in hard clustering by distributing weights evenly.
    - Falls back to fast initialization if accurate initialization fails.
    """

    MIN_SAMPLES = 10
    _estimation_strategies: ClassVar[Mapping[EstimationStrategy, Callable]] = MappingProxyType(
        {EstimationStrategy.QFUNCTION: q_function_strategy}
    )
    _cluster_match_strategies: ClassVar[Mapping[ClusterMatchStrategy, Callable]] = MappingProxyType(
        {
            ClusterMatchStrategy.LIKELIHOOD: match_clusters_for_models_log_likelihood,
            ClusterMatchStrategy.AKAIKE: match_clusters_for_models_akaike,
        }
    )

    def __init__(self, is_accurate: bool, is_soft: bool, clusterizer: Any):
        """Initializes the cluster-based initializer.

        Parameters
        ----------
        is_accurate : bool
            If True, uses accurate initialization with optimal cluster-model matching.
            If False, uses fast initialization with direct cluster assignments.
        is_soft : bool
            If True, uses soft clustering (fuzzy assignments).
            If False, uses hard clustering (crisp assignments).
        clusterizer : Any
            The clustering algorithm instance. Must have appropriate methods for
            the specified clustering type.
        """
        self.is_soft = is_soft
        self.is_accurate = is_accurate
        self.clusterizer = clusterizer
        self.n_components: int | None = None
        self.cluster_match_strategy: ClusterMatchStrategy = ClusterMatchStrategy.LIKELIHOOD
        self.estimation_strategies: list[EstimationStrategy] = []
        self.models: list[ContinuousDistribution] = []

    def _clusterize(self, X: np.ndarray, clusterizer: Any) -> np.ndarray:
        """Performs clustering on the input data and returns weight matrix.

        Parameters
        ----------
        X : np.ndarray
            Input data points to cluster.
        clusterizer : Any
            The clustering algorithm instance.

        Returns
        -------
        H : np.ndarray
            Weight matrix where H[i, k] represents the probability that data point i
            belongs to cluster k.

        Raises
        ------
        ValueError
            If the clusterizer doesn't have the required method for the specified
            clustering type, or if clustering fails.
        """
        if self.is_soft and hasattr(clusterizer, "fit_transform"):
            try:
                H = clusterizer.fit_transform(X)
                return H
            except Exception as e:
                raise ValueError(f"Fuzzy clusterizer failed: {e}")

        elif not self.is_soft and hasattr(clusterizer, "fit_predict"):
            try:
                labels = clusterizer.fit_predict(X)
                unique_labels = np.unique(labels)
                valid_labels = unique_labels[unique_labels != -1]
                n_clusters = len(valid_labels)
                if n_clusters == 0:
                    n_clusters = 1
                    valid_labels = np.array([0])
                    labels = np.zeros_like(labels)
                H = np.zeros((len(X), n_clusters))
                if np.any(labels == -1):
                    outlier_mask = labels == -1
                    non_outlier_mask = ~outlier_mask
                    H[outlier_mask, :] = 1.0 / n_clusters
                    for idx in np.where(non_outlier_mask)[0]:
                        label = labels[idx]
                        cluster_idx = np.where(valid_labels == label)[0][0]
                        H[idx, cluster_idx] = 1.0
                else:
                    for i, label in enumerate(labels):
                        cluster_idx = np.where(valid_labels == label)[0][0]
                        H[i, cluster_idx] = 1.0
                return H

            except Exception as e:
                raise ValueError(f"Hard clusterizer failed: {e}")
        else:
            raise ValueError("Clusterizer doesn't have required method")

    def _accurate_init(
        self, X: np.ndarray, H: np.ndarray, optimizer: Optimizer = ScipyNelderMead()
    ) -> tuple[list[ContinuousDistribution], list[float]]:
        """Performs accurate initialization with optimal cluster-model matching.

        Parameters
        ----------
        X : np.ndarray
            Input data points.
        H : np.ndarray
            Weight matrix from clustering.
        optimizer : Optimizer
            Optimizer that will be used in estimation strategies.
            By default, ScipyNelderMead.

        Returns
        -------
        tuple[list[ContinuousDistribution], list[float]]
            A tuple containing:
            - List of initialized distribution models
            - List of component weights

        Raises
        ------
        ValueError
            If n_components is not set or if the number of estimation strategies
            doesn't match the number of models.
        """
        if self.n_components is None:
            raise ValueError("n_components must be set before calling _accurate_init")

        if len(self.estimation_strategies) != len(self.models):
            raise ValueError("Count of models must match count of estimation strategies")

        cluster_match_func = self._cluster_match_strategies[self.cluster_match_strategy]

        estimation_funcs = [self._estimation_strategies[strategy] for strategy in self.estimation_strategies]

        distributions, params, weights = cluster_match_func(self.models, X, H, estimation_funcs)
        if not np.all(params):
            return self._fast_init(X, H, optimizer)

        new_distributions = []
        for i, dist in enumerate(distributions):
            params_names = params[i].keys()
            params_values = params[i].values()
            dist.set_params_from_vector(params_names, params_values)
            new_distributions.append(dist)

        return new_distributions, weights

    def _fast_init(
        self, X: np.ndarray, H: np.ndarray, optimizer: Optimizer = ScipyNelderMead()
    ) -> tuple[list[ContinuousDistribution], list[float]]:
        """Performs fast initialization with direct cluster assignments.

        Parameters
        ----------
        X : np.ndarray
            Input data points.
        H : np.ndarray
            Weight matrix from clustering.
        optimizer : Optimizer
            Optimizer that will be used in estimation strategies.
            By default, ScipyNelderMead.

        Returns
        -------
        tuple[list[ContinuousDistribution], list[float]]
            A tuple containing:
            - List of initialized distribution models
            - List of component weights

        Raises
        ------
        ValueError
            If n_components is not set.
        """
        if self.n_components is None:
            raise ValueError("n_components must be set before calling _fast_init")

        distributions: list[ContinuousDistribution] = []
        weights: list[float] = []
        estimation_funcs = [self._estimation_strategies[strategy] for strategy in self.estimation_strategies]

        for k in range(self.n_components):
            model = self.models[k]
            H_k = H[:, k]
            params = estimation_funcs[k](model, X, H_k, optimizer)
            params_names = params.keys()
            params_values = params.values()
            model.set_params_from_vector(params_names, params_values)
            weight = np.sum(H, axis=0)[k] / len(X)

            distributions.append(model)
            weights.append(float(weight))

        return distributions, weights

    def perform(
        self,
        X: ArrayLike,
        dists: list[ContinuousDistribution],
        cluster_match_strategy: ClusterMatchStrategy,
        estimation_strategies: list[EstimationStrategy],
        optimizer: Optimizer = ScipyNelderMead(),
    ) -> MixtureModel:
        """Performs cluster-based initialization of mixture model parameters.

        Parameters
        ----------
        X : ArrayLike
            Input data points for initialization.
        dists : list[ContinuousDistribution]
            List of distribution models to initialize.
        cluster_match_strategy : ClusterMatchStrategy
            Strategy for matching clusters to distribution models.
        estimation_strategies : list[EstimationStrategy]
            List of estimation strategies for each distribution model.
        optimizer : Optimizer
            Optimizer that will be used in estimation strategies.
            By default, ScipyNelderMead.

        Returns
        -------
        MixtureModel
            Initialized mixture model with estimated parameters and weights.

        Notes
        -----
        The method follows these steps:
        1. Sets up the models and configuration
        2. Performs clustering on the input data
        3. Estimates parameters using either accurate or fast initialization
        4. Normalizes component weights
        5. Returns the initialized mixture model
        """
        X = np.asarray(X, dtype=np.float64)
        self.models = dists
        self.n_components = len(dists)
        H = self._clusterize(X, self.clusterizer)
        self.cluster_match_strategy = cluster_match_strategy
        self.estimation_strategies = estimation_strategies

        if self.is_accurate:
            distributions, weights = self._accurate_init(X, H, optimizer)
        else:
            distributions, weights = self._fast_init(X, H, optimizer)

        total_weight = sum(weights)
        normalized_weights: list[float] = [w / total_weight for w in weights]
        current_mixture = MixtureModel(distributions, normalized_weights)  # type: ignore[var-annotated]
        return current_mixture
