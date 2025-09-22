from copy import deepcopy
from itertools import permutations
from typing import Callable

import numpy as np

from rework_pysatl_mpest import ContinuousDistribution
from rework_pysatl_mpest.optimizers.scipy_powell import ScipyPowell


def match_clusters_for_models_log_likelihood(
    models: list[ContinuousDistribution],
    X: np.ndarray,
    H: np.ndarray,
    estimation_info: list[Callable],
    min_samples: int = 10,
) -> tuple[list[ContinuousDistribution], list[dict[str, float]], list[float]]:
    X = X.flatten()
    n_clusters = H.shape[1]
    n_models = len(models)

    if len(estimation_info) != n_models:
        raise ValueError("Number of estimation functions must match number of models")

    updated_params_list = []
    model_weights = []

    cluster_weights = np.sum(H, axis=0)

    valid_clusters = [k for k in range(n_clusters) if cluster_weights[k] >= min_samples]
    used_clusters = set()

    for i, (model, estimation_func) in enumerate(zip(models, estimation_info)):
        best_score = -np.inf
        best_params = {}
        best_cluster_weight = 0.0
        best_cluster = None

        for k in valid_clusters:
            if k in used_clusters:
                continue
            H_k = H[:, k]

            temp_model = deepcopy(model)
            new_params = estimation_func(temp_model, X, H_k, ScipyPowell)
            param_names = new_params.keys()
            param_values = new_params.values()
            temp_model.set_params_from_vector(param_names, param_values)

            log_probs = np.clip(temp_model.lpdf(X), -1e9, -1e-9)
            weighted_log_likelihood = np.sum(H_k * log_probs)

            effective_n = cluster_weights[k]
            score = weighted_log_likelihood / effective_n if effective_n > 0 else -np.inf

            if score > best_score:
                best_score = score
                best_params = new_params
                best_cluster_weight = cluster_weights[k] / len(X)
                best_cluster = k
        used_clusters.add(best_cluster)
        updated_params_list.append(best_params)
        model_weights.append(float(best_cluster_weight))

    total_weight = sum(model_weights)
    normalized_weights = [w / total_weight for w in model_weights]

    return models, updated_params_list, normalized_weights


def match_clusters_for_models_akaike(
    models: list[ContinuousDistribution],
    X: np.ndarray,
    H: np.ndarray,
    estimation_info: list[Callable],
    min_samples: int = 10,
) -> tuple[list[ContinuousDistribution], list[dict[str, float]], list[float]]:
    n_clusters = H.shape[1]
    n_models = len(models)

    aic_scores_dict = {}

    cluster_weights = np.sum(H, axis=0)
    valid_clusters = [k for k in range(n_clusters) if cluster_weights[k] >= min_samples]

    for i, (model, estimation_func) in enumerate(zip(models, estimation_info)):
        for k in valid_clusters:
            H_k = H[:, k]

            temp_model = deepcopy(model)
            new_params = estimation_func(temp_model, X, H_k, ScipyPowell)
            param_names = new_params.keys()
            param_values = new_params.values()
            temp_model.set_params_from_vector(param_names, param_values)

            log_probs = np.clip(temp_model.lpdf(X), -1e9, -1e-9)
            weighted_log_likelihood = np.sum(H_k * log_probs)

            k_params = len(model.params)

            aic_score = 2 * k_params - 2 * weighted_log_likelihood

            key = f"{i}_{k}"
            aic_scores_dict[key] = {
                "aic_score": aic_score,
                "params": new_params,
                "cluster_weight": cluster_weights[k] / len(X),
                "model_idx": i,
                "cluster_idx": k,
            }

    best_total_aic = np.inf
    best_params_assignment = []
    best_weights_assignment = []

    for cluster_perm in permutations(valid_clusters, n_models):
        total_aic = 0
        params_assignment = []
        weights_assignment = []
        valid_assignment = True

        used_clusters = set()
        used_models = set()

        for i, cluster_idx in enumerate(cluster_perm):
            if cluster_idx in used_clusters:
                valid_assignment = False
                break
            used_clusters.add(cluster_idx)

            if i in used_models:
                valid_assignment = False
                break
            used_models.add(i)

            key = f"{i}_{cluster_idx}"

            data = aic_scores_dict[key]
            total_aic += data["aic_score"]
            params_assignment.append(data["params"])
            weights_assignment.append(data["cluster_weight"])

        if (
            valid_assignment
            and len(used_models) == n_models
            and len(used_clusters) == n_models
            and total_aic < best_total_aic
        ):
            best_total_aic = total_aic
            best_params_assignment = params_assignment
            best_weights_assignment = weights_assignment

    total_weight = sum(best_weights_assignment)
    normalized_weights = [w / total_weight for w in best_weights_assignment]

    return models, best_params_assignment, normalized_weights
