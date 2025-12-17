"""
A module providing cluster matching strategies for mixture model initialization.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from itertools import permutations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...core import MixtureModel
from ...distributions import ContinuousDistribution
from .utils import (
    Context,
    MatchingResult,
    _estimate_and_score_component,
    _precompute_fits,
)


def _match_greedy(context: Context) -> MatchingResult:
    """Sequentially assigns each model to its best available cluster.

    Iterates through the models in order. For each model, calculates the score
    against all currently *unused* clusters and assigns the one with the best
    (lowest) score.

    Parameters
    ----------
    context : Context
        The execution context dictionary containing:
        - "models": List of distributions.
        - "X": Input data.
        - "H": Responsibility matrix.
        - "estimation_strategies": List of estimation functions.
        - "optimizer": The optimizer instance.
        - "valid_clusters": List of valid cluster indices.
        - "cluster_weights": List of float.
        - "score_func_component": Function to score individual components.
        - "score_func_mixture": Function to score mixture.

    Returns
    -------
    MatchingResult
        A tuple containing:
        - The list of models (unchanged order).
        - List of estimated parameter dictionaries.
        - List of component weights.

    Notes
    -----
    **Complexity**
    O(N^2), where N is the number of models/clusters.

    **Behavior**
    This is a locally optimal strategy. Once a cluster is assigned to a model,
    it cannot be reassigned, even if it would be a better fit for a subsequent model.
    """
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
                best_cluster_weight = fit_result["weight"]
                best_cluster_idx = k

        if best_cluster_idx != -1:
            used_clusters.add(best_cluster_idx)
            updated_params_list.append(best_params)
            model_weights.append(float(best_cluster_weight))

    return context["models"], updated_params_list, model_weights


def _match_hungarian(context: Context) -> MatchingResult:
    """Finds the optimal assignment that minimizes the sum of individual scores.

    Uses the Hungarian algorithm (Munkres algorithm) via `linear_sum_assignment`
    to solve the linear assignment problem. It constructs a cost matrix from
    precomputed fits and finds the unique assignment of models to clusters that
    minimizes the total cost.

    Parameters
    ----------
    context : Context
        The execution context dictionary containing:
        - "models": List of distributions.
        - "X": Input data.
        - "H": Responsibility matrix.
        - "estimation_strategies": List of estimation functions.
        - "optimizer": The optimizer instance.
        - "valid_clusters": List of valid cluster indices.
        - "cluster_weights": List of float.
        - "score_func_component": Function to score individual components.
        - "score_func_mixture": Function to score mixture.

    Returns
    -------
    MatchingResult
        A tuple containing:
        - The re-ordered list of models matching the assignment.
        - List of estimated parameter dictionaries.
        - List of component weights.

    Notes
    -----
    **Complexity**
    O(N^3), where N is the number of models/clusters.

    **Behavior**
    Unlike the greedy approach, this finds the global optimum for the sum of
    individual component scores (e.g., sum of AICs of individual distributions).
    """
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
    """Finds the assignment that minimizes the total mixture score via brute-force.

    Generates all possible permutations of model-to-cluster assignments. Unlike
    the Hungarian method, which minimizes the sum of component scores, this method
    constructs a full `MixtureModel` for every permutation and evaluates the
    `score_func_mixture` (e.g., total Mixture AIC).

    Parameters
    ----------
    context : Context
        The execution context dictionary containing:
        - "models": List of distributions.
        - "X": Input data.
        - "H": Responsibility matrix.
        - "estimation_strategies": List of estimation functions.
        - "optimizer": The optimizer instance.
        - "valid_clusters": List of valid cluster indices.
        - "cluster_weights": List of float.
        - "score_func_component": Function to score individual components.
        - "score_func_mixture": Function to score mixture.

    Returns
    -------
    MatchingResult
        A tuple containing:
        - The best ordered list of models.
        - List of estimated parameter dictionaries.
        - List of component weights.

    Notes
    -----
    **Complexity**
    O(N!), where N is the number of models.

    **Performance**
    This provides the most accurate initialization metric but is computationally
    expensive. Recommended only for a small number of components (e.g., < 6)
    or if you need guaranty best models-clusters match.
    """
    models, X, score_func_mixture = context["models"], context["X"], context["score_func_mixture"]
    cached_fits = _precompute_fits(context)
    n_models, n_valid_clusters = len(models), len(cached_fits[0])

    best_total_score = np.inf
    best_params: list[dict[str, float]] = []
    best_weights: list[float] = []
    best_model_order: list[ContinuousDistribution] = []

    for cluster_perm_indices in permutations(range(n_valid_clusters), n_models):
        perm_models = [cached_fits[i][cluster_perm_indices[i]]["model"] for i in range(n_models)]
        perm_params = [cached_fits[i][cluster_perm_indices[i]]["params"] for i in range(n_models)]
        perm_weights = [float(cached_fits[i][cluster_perm_indices[i]]["weight"]) for i in range(n_models)]

        normalized_weights = [float(w) / sum(perm_weights) for w in perm_weights]
        temp_mixture: MixtureModel = MixtureModel(components=perm_models, weights=normalized_weights)
        total_score = score_func_mixture(temp_mixture, X)

        if total_score < best_total_score:
            best_total_score = total_score
            best_params = perm_params
            best_weights = normalized_weights
            best_model_order = [models[i] for i in range(n_models)]

    return best_model_order, best_params, best_weights
