"""
A module providing a cluster-based initializer for mixture models.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Optional

import numpy as np
from numpy.typing import ArrayLike

from ...core import MixtureModel
from ...distributions import ContinuousDistribution
from ...optimizers import Optimizer, ScipyNelderMead
from ...typings import Clusterizer
from .._estimation_strategies import q_function_strategy
from ..initializer import Initializer
from ._cluster_match_algorithms import _match_greedy, _match_hungarian, _match_permutations
from ._score_functions import (
    _calculate_component_aic,
    _calculate_component_log_likelihood,
    _calculate_mixture_aic,
    _calculate_mixture_log_likelihood,
)
from .strategies import EstimationStrategy, MatchingMethod, ScoringMethod
from .utils import Context, MatchingResult, _validate_clusters_distributions


class ClusterizeInitializer(Initializer):
    """Cluster-based initializer for mixture model parameters.

    This initializer uses clustering algorithms to partition the data and then
    estimates initial parameters for mixture components based on the clustering results.
    Supports both hard clustering (crisp assignments) and soft clustering (fuzzy assignments).
    For homogeneous mixtures fast (is_accurate = False) initialization is recommended.

    Attributes
    ----------
    _estimation_strategies : ClassVar[Mapping[EstimationStrategy, Callable]]
        Internal mapping linking `EstimationStrategy` enums to their corresponding
        parameter estimation functions (e.g., Q-function strategy).
    _MATCHING_METHOD : ClassVar[dict[MatchingMethod, Callable]]
        Internal mapping linking `MatchingMethod` enums to specific cluster matching
        algorithms (Greedy, Hungarian, Permutations).
    _SCORING_METHOD : ClassVar[dict[ScoringMethod, tuple[Callable, Callable]]]
        Internal mapping linking `ScoringMethod` enums to tuples of scoring functions.
        Each tuple contains `(component_score_func, mixture_score_func)`.

    Parameters
    ----------
    is_accurate : bool
        If True, uses accurate initialization with optimal cluster-model matching.
        If False, uses fast initialization with direct cluster assignments.
    is_soft : bool
        If True, uses soft clustering (fuzzy assignments).
        If False, uses hard clustering (crisp assignments).
    clusterizer : Clusterizer
        The clustering algorithm instance. Must comply with either HardClusterizer
        (fit_predict) or SoftClusterizer (fit_transform) protocols.
    optimizer : Optimizer, optional
        Optimizer for parameter estimation. Default is ScipyNelderMead.

    Methods
    -------
    perform(X, dists, method, score_func, estimation_strategies, optimizer, clusterizer)
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

    _estimation_strategies: ClassVar[Mapping[EstimationStrategy, Callable]] = MappingProxyType(
        {EstimationStrategy.QFUNCTION: q_function_strategy}
    )
    _MATCHING_METHOD: ClassVar[dict[MatchingMethod, Callable]] = {
        MatchingMethod.GREEDY: _match_greedy,
        MatchingMethod.HUNGARIAN: _match_hungarian,
        MatchingMethod.PERMUTATIONS: _match_permutations,
    }
    _SCORING_METHOD: ClassVar[dict[ScoringMethod, tuple[Callable, Callable]]] = {
        ScoringMethod.AIC: (_calculate_component_aic, _calculate_mixture_aic),
        ScoringMethod.LIKELIHOOD: (
            lambda m, X, H_k: -_calculate_component_log_likelihood(m, X, H_k),
            lambda m, X: -_calculate_mixture_log_likelihood(m, X),
        ),
    }

    def __init__(
        self, is_accurate: bool, is_soft: bool, clusterizer: Clusterizer, optimizer: Optimizer = ScipyNelderMead()
    ):
        """Initializes the cluster-based initializer.

        Parameters
        ----------
        is_accurate : bool
            If True, uses accurate initialization with optimal cluster-model matching.
            If False, uses fast initialization with direct cluster assignments.
        is_soft : bool
            If True, uses soft clustering (fuzzy assignments).
            If False, uses hard clustering (crisp assignments).
        clusterizer : Clusterizer
            The clustering algorithm instance.
        optimizer : Optimizer, optional
            Optimizer for parameter estimation. Default is ScipyNelderMead.
        """
        self.is_soft = is_soft
        self.is_accurate = is_accurate
        self.clusterizer = clusterizer
        self.optimizer = optimizer
        self.n_components: Optional[int] = None
        self.method: MatchingMethod = MatchingMethod.GREEDY
        self.score_func: ScoringMethod = ScoringMethod.LIKELIHOOD
        self.estimation_strategies: list[EstimationStrategy] = []
        self.models: list[ContinuousDistribution] = []

    def _clusterize(self, X: np.ndarray, clusterizer: Clusterizer) -> np.ndarray:
        """Performs clustering on the input data and returns weight matrix.

        Parameters
        ----------
        X : np.ndarray
            Input data points to cluster.
        clusterizer : Clusterizer
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

        if X.ndim == 1:
            X = X.reshape(-1, 1)

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
                    valid_indices = np.where(non_outlier_mask)[0]
                    valid_data_labels = labels[non_outlier_mask]
                    col_indices = np.searchsorted(valid_labels, valid_data_labels)
                    H[valid_indices, col_indices] = 1.0
                else:
                    col_indices = np.searchsorted(valid_labels, labels)
                    H[np.arange(len(X)), col_indices] = 1.0
                return H

            except Exception as e:
                raise ValueError(f"Hard clusterizer failed: {e}")
        else:
            raise ValueError("Clusterizer doesn't have required method")

    def _match_clusters_for_models(
        self,
        models: list[ContinuousDistribution],
        X: np.ndarray,
        H: np.ndarray,
        estimation_strategies: list[Callable],
        method: MatchingMethod,
        score_func: ScoringMethod,
        optimizer: Optimizer = ScipyNelderMead(),
    ) -> MatchingResult:
        """Matches clusters to models using a specified strategy and scoring function.

        Parameters
        ----------
        models : list[ContinuousDistribution]
            List of distributions available for assignment.
        X : np.ndarray
            Input data points.
        H : np.ndarray
            Weight matrix where H[i, k] is the probability of point i in cluster k.
        estimation_strategies : list[Callable]
            Estimation functions for each model.
        method : MatchingMethod
            The cluster matching strategy (Greedy, Hungarian, etc.).
        score_func : ScoringMethod
            The scoring criterion (AIC, Likelihood) used for optimization.
        optimizer : Optimizer, optional
            Optimizer used in estimation strategies. Default is ScipyNelderMead.

        Returns
        -------
        MatchingResult
            A tuple containing:
            - Ordered list of distribution models.
            - List of parameter dictionaries.
            - List of component weights.

        Notes
        -----
        The function follows these steps:
        1. Calculates minimum sample threshold (1% of data size).
        2. Validates clusters and filters out those with insufficient samples.
        3. Prepares the execution context.
        4. Dispatches to the specific matching algorithm defined by `method`.
        """
        n_models = len(models)
        min_samples = int(np.ceil(len(X) * 0.01))
        valid_clusters, cluster_weights = _validate_clusters_distributions(
            H, n_models, len(estimation_strategies), min_samples
        )

        if not valid_clusters:
            default_params: list[dict[str, float]] = [{} for _ in range(n_models)]
            return models, default_params, [1.0 / n_models] * n_models

        method_func = self._MATCHING_METHOD[method]
        score_component_func, score_mixture_func = self._SCORING_METHOD[score_func]

        context: Context = {
            "models": models,
            "X": X,
            "H": H,
            "estimation_strategies": estimation_strategies,
            "optimizer": optimizer,
            "valid_clusters": valid_clusters,
            "cluster_weights": cluster_weights,
            "score_func_component": score_component_func,
            "score_func_mixture": score_mixture_func,
        }

        return method_func(context=context)

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

        estimation_funcs = [self._estimation_strategies[strategy] for strategy in self.estimation_strategies]

        distributions, params, weights = self._match_clusters_for_models(
            models=self.models,
            X=X,
            H=H,
            estimation_strategies=estimation_funcs,
            method=self.method,
            score_func=self.score_func,
            optimizer=optimizer,
        )
        if not all(params):
            return self._fast_init(X, H, optimizer)

        new_distributions = []
        for i, dist in enumerate(distributions):
            params_names, params_values = zip(*params[i].items())
            dist.set_params_from_vector(list(params_names), list(params_values))
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
            If n_components is not set or if clusterizer failed.
        """
        if self.n_components is None:
            raise ValueError("n_components must be set before calling _fast_init")

        distributions: list[ContinuousDistribution] = []
        weights: list[float] = []
        estimation_funcs = [self._estimation_strategies[strategy] for strategy in self.estimation_strategies]

        n_found_clusters = H.shape[1]
        if n_found_clusters < self.n_components:
            raise IndexError(
                f"Clusterizer haven't found enough clusters. {n_found_clusters} found when {self.n_components} needed."
            )

        for k in range(self.n_components):
            model = self.models[k]
            H_k = H[:, k]
            params = estimation_funcs[k](model, X, H_k, optimizer)
            params_names, params_values = zip(*params.items())
            model.set_params_from_vector(params_names, params_values)
            weight = np.sum(H, axis=0)[k] / len(X)

            distributions.append(model)
            weights.append(float(weight))

        return distributions, weights

    def perform(
        self,
        X: ArrayLike,
        dists: list[ContinuousDistribution],
        method: MatchingMethod = MatchingMethod.GREEDY,
        score_func: ScoringMethod = ScoringMethod.LIKELIHOOD,
        estimation_strategies: list[EstimationStrategy] | None = None,
        optimizer: Optimizer | None = None,
        clusterizer: Clusterizer | None = None,
        **kwargs: Any,
    ) -> MixtureModel:
        """Performs cluster-based initialization of mixture model parameters.

        Parameters
        ----------
        X : ArrayLike
            Input data points for initialization.
        dists : list[ContinuousDistribution]
            List of distribution models to initialize.
        method : MatchingMethod, optional
            The algorithm used to match the clusters. Default is GREEDY.
        score_func : ScoringMethod, optional
            The metric used to score the fit. Default is LIKELIHOOD.
        estimation_strategies : list[EstimationStrategy], optional
            List of estimation strategies. If None, uses QFUNCTION for all models.
        optimizer : Optimizer, optional
            Optimizer for parameter estimation. If None, uses the one from __init__.
        clusterizer : Clusterizer, optional
            The clustering algorithm instance. If None, uses the one from __init__.
        **kwargs : Any
            Additional arguments for compatibility.

        Returns
        -------
        MixtureModel
            Initialized mixture model with estimated parameters and weights

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
        clusterizer = clusterizer or self.clusterizer
        optimizer = optimizer or self.optimizer
        H = self._clusterize(X, clusterizer)
        self.method = method
        self.score_func = score_func
        self.estimation_strategies = estimation_strategies or [EstimationStrategy.QFUNCTION] * self.n_components

        if self.is_accurate:
            distributions, weights = self._accurate_init(X, H, optimizer)
        else:
            distributions, weights = self._fast_init(X, H, optimizer)

        current_mixture: MixtureModel = MixtureModel(distributions, weights)
        return current_mixture
