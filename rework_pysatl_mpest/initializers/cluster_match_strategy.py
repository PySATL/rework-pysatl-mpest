"""
A module providing cluster matching strategies for mixture model initialization.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from copy import copy
from itertools import permutations
from typing import Any, Callable, TypedDict

import numpy as np
from scipy.optimize import linear_sum_assignment

from rework_pysatl_mpest.core import MixtureModel
from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.initializers.strategies import MatchingMethod, ScoringMethod
from rework_pysatl_mpest.optimizers import Optimizer, ScipyNelderMead

MatchingResult = tuple[list[ContinuousDistribution], list[dict[str, float]], list[float]]
Context = dict[str, Any]
ScoreFunc = Callable[[MixtureModel | ContinuousDistribution, np.ndarray, np.ndarray | None], float]


class FitResult(TypedDict):
    """A TypedDict to represent the result of a single model-cluster fit."""

    model: ContinuousDistribution
    params: dict[str, float]
    score: float
    weight: float


def _validate_clusters_distributions(
    H: np.ndarray, models_count: int, estimation_strategies_count: int, min_samples: int
) -> tuple[list[int], list[float]]:
    """Validates clusters and models for further comparison"""
    if not np.allclose(np.sum(H, axis=1), 1, atol=1e-10):
        raise ValueError("Sum of H matrix weights must be equal to 1")

    n_clusters = H.shape[1]

    if estimation_strategies_count != models_count:
        raise ValueError("Number of estimation functions must match number of models")

    cluster_weights: list[float] = np.sum(H, axis=0)

    valid_clusters = [k for k in range(n_clusters) if cluster_weights[k] >= min_samples]
    if len(valid_clusters) != models_count:
        return [], []
    return valid_clusters, cluster_weights


def _estimate_and_score_component(
    model: ContinuousDistribution,
    estimation_func: Callable,
    score_func: ScoreFunc,
    X: np.ndarray,
    H_k: np.ndarray,
    optimizer: Optimizer,
) -> FitResult:
    """Estimates parameters for a model-cluster pair and computes its score"""
    temp_model = copy(model)
    new_params: dict[str, float] = estimation_func(temp_model, X, H_k, optimizer)
    temp_model.set_params_from_vector(list(new_params.keys()), list(new_params.values()))

    score = score_func(temp_model, X, H_k)

    return {"model": temp_model, "params": new_params, "score": score, "weight": -1.0}


def _calculate_component_log_likelihood(model: ContinuousDistribution, X: np.ndarray, H_k: np.ndarray) -> float:
    """Calculates the weighted log-likelihood for a single component"""
    log_probs = np.clip(model.lpdf(X), -1e9, -1e-9)
    return np.sum(H_k * log_probs)


def _calculate_mixture_log_likelihood(model: MixtureModel, X: np.ndarray) -> float:
    """Calculates the total log-likelihood for a mixture model"""
    X_flattened = np.asarray(X)
    dim_const = 2
    if X_flattened.ndim == dim_const and X_flattened.shape[1] == 1:
        X_flattened = X_flattened.flatten()

    return np.sum(model.loglikelihood(X_flattened))


def _calculate_component_aic(model: ContinuousDistribution, X: np.ndarray, H_k: np.ndarray) -> float:
    """Calculates AIC for a single component based on weighted log-likelihood"""
    weighted_log_likelihood = _calculate_component_log_likelihood(model, X, H_k)
    k_params = len(model.params)
    return 2 * k_params - 2 * weighted_log_likelihood


def _calculate_mixture_aic(model: MixtureModel, X: np.ndarray) -> float:
    """Calculates AIC for the entire mixture model"""
    log_likelihood = _calculate_mixture_log_likelihood(model, X)
    k_params = sum(len(dist.params) for dist in model.components)

    k_params += model.n_components - 1

    return 2 * k_params - 2 * log_likelihood


def _precompute_fits(context: Context) -> list[list[FitResult]]:
    """
    Pre-computes model fits
    """
    models = context["models"]
    estimation_strategies = context["estimation_strategies"]
    valid_clusters = context["valid_clusters"]
    score_func_component = context["score_func_component"]
    X, H, optimizer = context["X"], context["H"], context["optimizer"]
    cluster_weights = context["cluster_weights"]
    n_samples = len(X)

    cached_fits: list[list[FitResult]] = []
    computation_cache: dict[tuple, FitResult] = {}

    for model, est_func in zip(models, estimation_strategies):
        row: list[FitResult] = []
        for k in valid_clusters:
            cache_key = (model.__class__, est_func, k)

            if cache_key not in computation_cache:
                fit_result = _estimate_and_score_component(model, est_func, score_func_component, X, H[:, k], optimizer)
                fit_result["weight"] = cluster_weights[k] / n_samples
                computation_cache[cache_key] = fit_result

            row.append(computation_cache[cache_key])
        cached_fits.append(row)

    return cached_fits


def _match_greedy(context: Context) -> MatchingResult:
    """sequentially assign each model to its best available cluster"""
    updated_params_list: list[dict[str, float]] = []
    model_weights: list[float] = []
    used_clusters = set()

    for model, estimation_func in zip(context["models"], context["estimation_strategies"]):
        best_score = np.inf
        best_params: dict[str, float] = {}
        best_cluster_weight = 0.0
        best_cluster_idx = -1

        for k in context["valid_clusters"]:
            if k in used_clusters:
                continue

            fit_result = _estimate_and_score_component(
                model,
                estimation_func,
                context["score_func_component"],
                context["X"],
                context["H"][:, k],
                context["optimizer"],
            )
            score: float = fit_result["score"]
            if score < best_score:
                best_score = fit_result["score"]
                best_params = fit_result["params"]
                best_cluster_weight = context["cluster_weights"][k] / len(context["X"])
                best_cluster_idx = k

        if best_cluster_idx != -1:
            used_clusters.add(best_cluster_idx)
            updated_params_list.append(best_params)
            model_weights.append(float(best_cluster_weight))

    return context["models"], updated_params_list, model_weights


def _match_hungarian(context: Context) -> MatchingResult:
    """find optimal assignment that minimizes the total score"""
    models = context["models"]
    cached_fits = _precompute_fits(context)

    cost_matrix = np.array(
        [[cached_fits[i][j]["score"] for j in range(len(cached_fits[0]))] for i in range(len(models))]
    )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assigned_models = [models[i] for i in row_ind]
    best_params = [cached_fits[i][j]["params"] for i, j in zip(row_ind, col_ind)]
    best_weights = [cached_fits[i][j]["weight"] for i, j in zip(row_ind, col_ind)]

    return assigned_models, best_params, best_weights


def _match_permutations(context: Context) -> MatchingResult:
    """find assignment that minimizes the total mixture score"""
    models, X, score_func_mixture = context["models"], context["X"], context["score_func_mixture"]
    cached_fits = _precompute_fits(context)
    n_models, n_valid_clusters = len(models), len(cached_fits[0])

    best_total_score = np.inf
    best_params: list[dict[str, float]] = []
    best_weights: list[float] = []
    best_model_order: list[ContinuousDistribution] = []

    for model_perm_indices in permutations(range(n_models)):
        for cluster_perm_indices in permutations(range(n_valid_clusters), n_models):
            perm_models = [
                cached_fits[model_perm_indices[i]][cluster_perm_indices[i]]["model"] for i in range(n_models)
            ]
            perm_params = [
                cached_fits[model_perm_indices[i]][cluster_perm_indices[i]]["params"] for i in range(n_models)
            ]
            perm_weights = [
                float(cached_fits[model_perm_indices[i]][cluster_perm_indices[i]]["weight"]) for i in range(n_models)
            ]

            normalized_weights = [float(w) / sum(perm_weights) for w in perm_weights]
            temp_mixture = MixtureModel(components=perm_models, weights=normalized_weights)
            total_score = score_func_mixture(temp_mixture, X)

            if total_score < best_total_score:
                best_total_score = total_score
                best_params = perm_params
                best_weights = normalized_weights
                best_model_order = [models[i] for i in model_perm_indices]

    return best_model_order, best_params, best_weights


_MATCHING_METHOD: dict[MatchingMethod, Callable] = {
    MatchingMethod.GREEDY: _match_greedy,
    MatchingMethod.HUNGARIAN: _match_hungarian,
    MatchingMethod.PERMUTATIONS: _match_permutations,
}

_SCORING_METHOD: dict[ScoringMethod, tuple[Callable, Callable]] = {
    ScoringMethod.AIC: (_calculate_component_aic, _calculate_mixture_aic),
    ScoringMethod.LIKELIHOOD: (
        lambda m, X, H_k: -_calculate_component_log_likelihood(m, X, H_k),
        _calculate_mixture_log_likelihood,
    ),
}


def match_clusters_for_models(
    models: list[ContinuousDistribution],
    X: np.ndarray,
    H: np.ndarray,
    estimation_strategies: list[Callable],
    method: MatchingMethod,
    score_func: ScoringMethod,
    min_samples: int = 10,
    optimizer: Optimizer = ScipyNelderMead(),
) -> MatchingResult:
    """
    Matches clusters to models using a specified strategy and scoring function.

    Parameters
    ----------
    models : list[ContinuousDistribution]
        List of distributions
    X : np.ndarray
        Input data points
    H : np.ndarray
        Weight matrix where H[i, k] is the probability of point i in cluster k
    estimation_strategies : list[Callable]
        Estimation functions for each model
    method : MatchingMethod, optional
        The cluster matching strategy to use. Default is MatchingMethod.GREEDY
    score_func : ScoringMethod, optional
        The scoring criterion to use for optimization. Can be AIC or LIKELIHOOD
        Default is ScoringMethod.AIC
    min_samples : int, optional
        Minimum samples for a cluster to be valid. Default is 10
    optimizer : Optimizer, optional
        Optimizer used in estimation strategies. Default is ScipyNelderMead

    Returns
    -------
    MatchingResult
        A tuple containing (ordered models, parameters, weights)
    """
    n_models = len(models)
    valid_clusters, cluster_weights = _validate_clusters_distributions(
        H, n_models, len(estimation_strategies), min_samples
    )

    if not valid_clusters:
        default_params: list[dict[str, float]] = [{} for _ in range(n_models)]
        return models, default_params, [1.0 / n_models] * n_models

    method_func = _MATCHING_METHOD[method]
    score_component_func, score_mixture_func = _SCORING_METHOD[score_func]

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
