from typing import Any, Callable, Optional

import numpy as np
from initializer import Initializer

from rework_pysatl_mpest import ContinuousDistribution, MixtureModel
from rework_pysatl_mpest.estimators.iterative.Initializers.clusterMatchStrategy import find_best_cluster_for_model


class ClusterizeInitializer(Initializer):
    MIN_SAMPLES = 10

    def __init__(self, is_accurate: bool, is_soft: bool, clusterizer: Any):
        self.is_soft = is_soft
        self.is_accurate = is_accurate
        self.clusterizer = clusterizer
        self.n_components: Optional[int] = None
        self.models: list[ContinuousDistribution] = []

    def _clusterize(self, X: np.ndarray, clusterizer: Any) -> np.ndarray:
        if (
            hasattr(clusterizer, "n_clusters")
            and self.n_components is not None
            and self.n_components != clusterizer.n_clusters
        ):
            raise ValueError("Count of components and clusters doesn't match.")

        X = X.reshape(-1, 1)
        labels = clusterizer.fit_predict(X)

        if -1 in labels:
            if self.n_components is not None:
                labels[labels == -1] = np.random.choice(range(self.n_components), np.sum(labels == -1))
            else:
                unique_labels = np.unique(labels)
                valid_labels = unique_labels[unique_labels != -1]
                if len(valid_labels) > 0:
                    labels[labels == -1] = np.random.choice(valid_labels, np.sum(labels == -1))

        return labels

    def _accurate_init(self, X: np.ndarray, labels: np.ndarray) -> tuple[list[ContinuousDistribution], list[float]]:
        if self.n_components is None:
            raise ValueError("n_components must be set before calling _accurate_init")

        clusters = {k: X[labels == k] for k in range(self.n_components)}
        distributions: list[ContinuousDistribution] = []
        weights: list[float] = []

        for i, model in enumerate(self.models):
            best_k, best_params, best_score = find_best_cluster_for_model(model, clusters)

            if best_k is None or best_params is None:
                X_k = np.random.choice(X, size=10, replace=True)
                weight = 1.0 / self.n_components
                param_names = list(model.params)
                default_params = [float(np.mean(X_k)), float(np.clip(np.std(X_k), 0.1, 100.0))]
                model.set_params_from_vector(param_names, default_params)
                distribution = model
            else:
                weight = len(clusters[best_k]) / len(X)
                clusters.pop(best_k)
                dist_class, params_dict = best_params
                param_names = list(params_dict.keys())
                param_values = list(params_dict.values())
                model.set_params_from_vector(param_names, param_values)
                distribution = model

            distributions.append(distribution)
            weights.append(float(weight))

        return distributions, weights

    def _fast_init(self, X: np.ndarray, labels: np.ndarray) -> tuple[list[ContinuousDistribution], list[float]]:
        if self.n_components is None:
            raise ValueError("n_components must be set before calling _fast_init")

        distributions: list[ContinuousDistribution] = []
        weights: list[float] = []

        for k in range(self.n_components):
            X_k = X[labels == k]
            model = self.models[k]
            weight = len(X_k) / len(X)

            if len(X_k) == 0:
                X_k = np.random.choice(X, size=10, replace=True)
                weight = 1.0 / self.n_components

            params = [np.mean(X_k), np.clip(np.std(X_k), 0.1, 100.0)]
            params_name = list(model.params)
            params_dict = {str(param_name): float(param) for param_name, param in zip(params_name, params)}

            param_names = list(params_dict.keys())
            param_values = list(params_dict.values())
            model.set_params_from_vector(param_names, param_values)

            distributions.append(model)
            weights.append(float(weight))

        return distributions, weights

    def perform(self, x: np.ndarray, dists: list[ContinuousDistribution], info: list[Callable]) -> MixtureModel:
        self.models = dists
        self.n_components = len(dists)
        labels = self._clusterize(x, self.clusterizer)

        if self.is_accurate:
            distributions, weights = self._accurate_init(x, labels)
        else:
            distributions, weights = self._fast_init(x, labels)

        total_weight = sum(weights)
        normalized_weights: list[float] = [w / total_weight for w in weights]
        current_mixture = MixtureModel(distributions, normalized_weights)
        return current_mixture
