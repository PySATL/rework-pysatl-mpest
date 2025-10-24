"""
A module providing cluster matching strategies for mixture model initialization.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from copy import copy
from itertools import permutations
from typing import Callable

import numpy as np

from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.optimizers import Optimizer, ScipyNelderMead


def match_clusters_for_models_log_likelihood(
    models: list[ContinuousDistribution],
    X: np.ndarray,
    H: np.ndarray,
    estimation_strategies: list[Callable],
    min_samples: int = 10,
    optimizer: Optimizer = ScipyNelderMead(),
) -> tuple[list[ContinuousDistribution], list[dict[str, float]], list[float]]:
    """Matches clusters to models using weighted log-likelihood criteria.

    This function assigns each distribution model to the cluster that maximizes
    the weighted log-likelihood score. The assignment is performed sequentially,
    with each model selecting the best available cluster.

    Parameters
    ----------
    models : list[ContinuousDistribution]
        List of distribution models to be matched with clusters.
    X : np.ndarray
        Input data points used for parameter estimation.
    H : np.ndarray
        Weight matrix where ``H[i, k]`` represents the probability that data point ``i``
        belongs to cluster ``k``.
    estimation_strategies : list[Callable]
        List of estimation functions for each model, used to estimate parameters
        for a given cluster.
    min_samples : int, optional
        Minimum number of samples required for a cluster to be considered valid.
        Default is 10.
    optimizer : Optimizer
        Optimizer that will be used in estimation strategies.
        By default, ScipyNelderMead.

    Returns
    -------
    tuple[list[ContinuousDistribution], list[dict[str, float]], list[float]]
        A tuple containing:

        - The original list of models
        - List of parameter dictionaries for each model
        - List of weights for each model

    Raises
    ------
    ValueError
        If the sum of weights in H matrix is not equal to 1 for each data point,
        or if the number of estimation functions doesn't match the number of models.

    Notes
    -----
    The function performs the following steps:

    1. Validates input constraints
    2. Identifies valid clusters with sufficient samples
    3. Sequentially assigns each model to the best available cluster
    4. Estimates parameters using the provided estimation functions
    5. Normalizes the resulting weights

    If insufficient valid clusters are found, returns default parameters and equal weights.
    """

    if not np.allclose(np.sum(H, axis=1), 1, atol=1e-10):
        raise ValueError("Sum of H matrix weights must be equal to 1")

    X = X.flatten()
    n_clusters = H.shape[1]
    n_models = len(models)

    if len(estimation_strategies) != n_models:
        raise ValueError("Number of estimation functions must match number of models")

    updated_params_list = []
    model_weights = []

    cluster_weights = np.sum(H, axis=0)

    valid_clusters = [k for k in range(n_clusters) if cluster_weights[k] >= min_samples]
    if len(valid_clusters) != n_models:
        default_params: list[dict] = [{} for _ in range(n_models)]
        equal_weights = [1.0 / n_models] * n_models
        return models, default_params, equal_weights

    used_clusters = set()

    for i, (model, estimation_func) in enumerate(zip(models, estimation_strategies)):
        best_score = -np.inf
        best_params = {}
        best_cluster_weight = 0.0
        best_cluster = None
        temp_model = copy(model)
        default_params_names, default_params_values = (
            list(temp_model.params),
            temp_model.get_params_vector(list(temp_model.params)),
        )

        for k in valid_clusters:
            if k in used_clusters:
                continue
            H_k = H[:, k]

            new_params = estimation_func(temp_model, X, H_k, optimizer)
            param_names = new_params.keys()
            param_values = new_params.values()
            temp_model.set_params_from_vector(param_names, param_values)

            log_probs = np.clip(temp_model.lpdf(X), -1e9, -1e-9)
            weighted_log_likelihood = np.sum(H_k * log_probs)

            effective_n = cluster_weights[k]
            score = weighted_log_likelihood / effective_n

            if score > best_score:
                best_score = score
                best_params = new_params
                best_cluster_weight = cluster_weights[k] / len(X)
                best_cluster = k

            temp_model.set_params_from_vector(default_params_names, default_params_values)

        used_clusters.add(best_cluster)
        updated_params_list.append(best_params)
        model_weights.append(float(best_cluster_weight))

    return models, updated_params_list, model_weights


def match_clusters_for_models_akaike(
    models: list[ContinuousDistribution],
    X: np.ndarray,
    H: np.ndarray,
    estimation_strategies: list[Callable],
    min_samples: int = 10,
    optimizer: Optimizer = ScipyNelderMead(),
) -> tuple[list[ContinuousDistribution], list[dict[str, float]], list[float]]:
    """Matches clusters to models using Akaike Information Criterion (AIC).

    This function evaluates all possible permutations of cluster-model assignments
    and selects the combination that minimizes the total AIC score.

    Parameters
    ----------
    models : list[ContinuousDistribution]
        List of distribution models to be matched with clusters.
    X : np.ndarray
        Input data points used for parameter estimation.
    H : np.ndarray
        Weight matrix where ``H[i, k]`` represents the probability that data point ``i``
        belongs to cluster ``k``.
    estimation_strategies : list[Callable]
        List of estimation functions for each model, used to estimate parameters
        for a given cluster.
    min_samples : int, optional
        Minimum number of samples required for a cluster to be considered valid.
        Default is 10.
    optimizer : Optimizer
        Optimizer that will be used in estimation strategies.
        By default, ScipyNelderMead.

    Returns
    -------
    tuple[list[ContinuousDistribution], list[dict[str, float]], list[float]]
        A tuple containing:

        - The original list of models
        - List of parameter dictionaries for each model
        - List of weights for each model

    Raises
    ------
    ValueError
        If the sum of weights in H matrix is not equal to 1 for each data point,
        or if the number of estimation functions doesn't match the number of models.

    Notes
    -----
    The function performs the following steps:

    1. Validates input constraints
    2. Computes AIC scores for all possible model-cluster combinations
    3. Evaluates all permutations to find the assignment with minimum total AIC
    4. Returns the best parameter assignment and normalized weights

    AIC is calculated as: ``2 * k - 2 * log_likelihood``, where ``k`` is the number of parameters.
    """

    if not np.allclose(np.sum(H, axis=1), 1, atol=1e-10):
        raise ValueError("Sum of H matrix weights must be equal to 1")

    n_clusters = H.shape[1]
    n_models = len(models)

    if len(estimation_strategies) != n_models:
        raise ValueError("Number of estimation functions must match number of models")

    aic_scores_dict = {}

    cluster_weights = np.sum(H, axis=0)
    valid_clusters = [k for k in range(n_clusters) if cluster_weights[k] >= min_samples]

    if len(valid_clusters) != n_models:
        default_params: list[dict] = [{} for _ in range(n_models)]
        equal_weights = [1.0 / n_models] * n_models
        return models, default_params, equal_weights

    for i, (model, estimation_func) in enumerate(zip(models, estimation_strategies)):
        temp_model = copy(model)
        default_params_names, default_params_values = (
            list(temp_model.params),
            temp_model.get_params_vector(list(temp_model.params)),
        )
        for k in valid_clusters:
            H_k = H[:, k]

            new_params = estimation_func(temp_model, X, H_k, optimizer)
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
            temp_model.set_params_from_vector(default_params_names, default_params_values)

    best_total_aic = np.inf
    best_params_assignment = []
    best_weights_assignment = []

    for cluster_perm in permutations(valid_clusters, n_models):
        total_aic = 0
        params_assignment = []
        weights_assignment = []
        valid_assignment = True

        for i, cluster_idx in enumerate(cluster_perm):
            key = f"{i}_{cluster_idx}"
            data = aic_scores_dict[key]
            total_aic += data["aic_score"]
            params_assignment.append(data["params"])
            weights_assignment.append(data["cluster_weight"])

        if valid_assignment and total_aic < best_total_aic:
            best_total_aic = total_aic
            best_params_assignment = params_assignment
            best_weights_assignment = weights_assignment

    return models, best_params_assignment, best_weights_assignment
