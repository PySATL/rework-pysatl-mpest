from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Optional

import numpy as np
from initializer import Initializer

from rework_pysatl_mpest import ContinuousDistribution, MixtureModel
from rework_pysatl_mpest.Initializers.clusterMatchStrategy import (
    match_clusters_for_models_akaike,
    match_clusters_for_models_log_likelihood,
)
from rework_pysatl_mpest.Initializers.q_function import q_function_strategy
from rework_pysatl_mpest.Initializers.strategies import ClusterMatchStrategy, EstimationStrategy
from rework_pysatl_mpest.optimizers.scipy_nelder_mead import ScipyNelderMead


class ClusterizeInitializer(Initializer):
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
        self.is_soft = is_soft
        self.is_accurate = is_accurate
        self.clusterizer = clusterizer
        self.n_components: Optional[int] = None
        self.cluster_match_strategy: ClusterMatchStrategy = ClusterMatchStrategy.LIKELIHOOD
        self.estimation_strategies: list[EstimationStrategy] = []
        self.models: list[ContinuousDistribution] = []

    def _clusterize(self, X: np.ndarray, clusterizer: Any) -> np.ndarray:
        X = X.reshape(-1, 1)
        if self.is_soft and hasattr(clusterizer, "fit_transform"):
            try:
                weights_matrix = clusterizer.fit_transform(X)
                return weights_matrix
            except Exception as e:
                raise ValueError(f"Fuzzy clusterizer failed: {e}")

        elif not self.is_soft and hasattr(clusterizer, "fit_predict"):
            try:
                labels = clusterizer.fit_predict(X)

                unique_labels = np.unique(labels)
                valid_labels = unique_labels[unique_labels != -1]
                n_clusters = len(valid_labels)

                weights_matrix = np.zeros((len(X), n_clusters))

                if -1 in labels:
                    outlier_mask = labels == -1
                    valid_labels = labels[~outlier_mask]

                    weights_matrix[outlier_mask, :] = 1.0 / n_clusters

                    for i, label in enumerate(valid_labels):
                        if label != -1:
                            cluster_idx = np.where(valid_labels == label)[0][0]
                            weights_matrix[~outlier_mask][i, cluster_idx] = 1.0

                else:
                    for i, label in enumerate(labels):
                        cluster_idx = np.where(unique_labels == label)[0][0]
                        weights_matrix[i, cluster_idx] = 1.0
                return weights_matrix
            except Exception as e:
                raise ValueError(f"Hard clusterizer failed: {e}")
        else:
            raise ValueError("Clusterizer doesn't have required method")

    def _accurate_init(self, X: np.ndarray, H: np.ndarray) -> tuple[list[ContinuousDistribution], list[float]]:
        if self.n_components is None:
            raise ValueError("n_components must be set before calling _accurate_init")

        if len(self.estimation_strategies) != len(self.models):
            raise ValueError("Count of models must match count of estimation strategies")

        cluster_match_func = self._cluster_match_strategies[self.cluster_match_strategy]

        estimation_funcs = []
        for strategy in self.estimation_strategies:
            estimation_funcs.append(self._estimation_strategies[strategy])

        distributions, params, weights = cluster_match_func(self.models, X, H, estimation_funcs)

        new_distributions = []
        for i, dist in enumerate(distributions):
            params_names = params[i].keys()
            params_values = params[i].values()
            dist.set_params_from_vector(params_names, params_values)
            new_distributions.append(dist)

        return new_distributions, weights

    def _fast_init(self, X: np.ndarray, H: np.ndarray) -> tuple[list[ContinuousDistribution], list[float]]:
        if self.n_components is None:
            raise ValueError("n_components must be set before calling _fast_init")

        distributions: list[ContinuousDistribution] = []
        weights: list[float] = []
        estimation_funcs = []
        for strategy in self.estimation_strategies:
            estimation_funcs.append(self._estimation_strategies[strategy])

        for k in range(self.n_components):
            model = self.models[k]
            H_k = H[:, k]
            params = estimation_funcs[k](model, X, H_k, ScipyNelderMead)
            params_names = params.keys()
            params_values = params.values()
            model.set_params_from_vector(params_names, params_values)
            weight = np.sum(H, axis=0)[k] / len(X)

            distributions.append(model)
            weights.append(float(weight))

        return distributions, weights

    def perform(
        self,
        X: np.ndarray,
        dists: list[ContinuousDistribution],
        cluster_match_info: ClusterMatchStrategy,
        estimation_info: list[EstimationStrategy],
    ) -> MixtureModel:
        self.models = dists
        self.n_components = len(dists)
        H = self._clusterize(X, self.clusterizer)
        self.cluster_match_strategy = cluster_match_info
        self.estimation_strategies = estimation_info

        if self.is_accurate:
            distributions, weights = self._accurate_init(X, H)
        else:
            distributions, weights = self._fast_init(X, H)

        total_weight = sum(weights)
        normalized_weights: list[float] = [w / total_weight for w in weights]
        current_mixture = MixtureModel(distributions, normalized_weights)
        return current_mixture
