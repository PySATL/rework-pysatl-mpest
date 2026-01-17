"""
A module providing different utilities that can be suitable for initializers.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from copy import copy
from typing import Any, Callable, TypedDict

import numpy as np

from ...core import MixtureModel
from ...distributions import ContinuousDistribution
from ...optimizers import Optimizer

MatchingResult = tuple[list[ContinuousDistribution], list[dict[str, float]], list[float]]
Context = dict[str, Any]
ScoreFunc = Callable[[MixtureModel | ContinuousDistribution, np.ndarray, np.ndarray | None], float]


class FitResult(TypedDict):
    """A TypedDict to represent the result of a single model-cluster fit.

    Attributes
    ----------
    model : ContinuousDistribution
        The distribution instance with estimated parameters.
    params : dict[str, float]
        The estimated parameters (e.g., {'loc': 0.5, 'scale': 1.0}).
    score : float
        The goodness-of-fit score (e.g., AIC or negative log-likelihood).
    weight : float
        The calculated weight of this component in the mixture.
    """

    model: ContinuousDistribution
    params: dict[str, float]
    score: float
    weight: float


def _validate_clusters_distributions(
    H: np.ndarray, models_count: int, estimation_strategies_count: int, min_samples: int
) -> tuple[list[int], list[float]]:
    """Validates cluster weights and alignment with models.

    Ensures that the responsibility matrix H sums to 1 (row-wise normalization),
    matches dimensions with provided models, and filters out clusters that do not
    meet the minimum sample requirement.

    Parameters
    ----------
    H : np.ndarray
        Weight matrix of shape (n_samples, n_clusters) where H[i, k] is the
        probability that point i belongs to cluster k.
    models_count : int
        The number of distribution models provided for initialization.
    estimation_strategies_count : int
        The number of estimation strategies provided.
    min_samples : int
        The minimum sum of weights (effective sample count) required for a
        cluster to be considered valid.

    Returns
    -------
    tuple[list[int], list[float]]
        A tuple containing:
        - List of indices for valid clusters (those meeting min_samples).
        - List of total weights (sum of columns) for the valid clusters.

    Raises
    ------
    ValueError
        If the sum of weights in H matrix is not equal to 1 for each data point
        (tolerance 1e-10).
    ValueError
        If the number of estimation functions doesn't match the number of models.
    """
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
    """Estimates parameters for a model-cluster pair and computes its score.

    Creates a copy of the provided distribution model, uses the estimation
    strategy to find optimal parameters based on the weighted data `H_k`,
    and calculates a goodness-of-fit score.

    Parameters
    ----------
    model : ContinuousDistribution
        The template distribution model.
    estimation_func : Callable
        The strategy function to estimate parameters (e.g., Q-Function).
    score_func : ScoreFunc
        The function to calculate the score (e.g., AIC, Likelihood).
    X : np.ndarray
        Input data points.
    H_k : np.ndarray
        Weight vector for the specific cluster being evaluated.
    optimizer : Optimizer
        Optimization algorithm to be used by the estimation strategy.

    Returns
    -------
    FitResult
        A dictionary containing the fitted model, estimated parameters,
        calculated score, and component weight.
    """
    temp_model = copy(model)
    new_params: dict[str, float] = estimation_func(temp_model, X, H_k, optimizer)
    temp_model.set_params_from_vector(list(new_params.keys()), list(new_params.values()))

    score = score_func(temp_model, X, H_k)
    weight = float(np.sum(H_k) / len(X))

    return {"model": temp_model, "params": new_params, "score": score, "weight": weight}


def _precompute_fits(context: Context) -> list[list[FitResult]]:
    """Calculates all possible model-to-cluster fits.

    Iterates through every distribution model and every valid cluster to estimate
    parameters and calculate goodness-of-fit scores. This results in a matrix
    of fit results used by global optimization strategies (Hungarian and Permutations).

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
    list[list[FitResult]]
        A matrix (list of lists) where entry [i][j] contains the FitResult
        (params, score, weight) for the i-th model and the j-th valid cluster.
    """
    models = context["models"]
    estimation_strategies = context["estimation_strategies"]
    valid_clusters = context["valid_clusters"]
    score_func_component = context["score_func_component"]
    X, H, optimizer = context["X"], context["H"], context["optimizer"]

    cached_fits: list[list[FitResult]] = []

    for model, est_func in zip(models, estimation_strategies):
        row: list[FitResult] = []
        for k in valid_clusters:
            fit_result = _estimate_and_score_component(model, est_func, score_func_component, X, H[:, k], optimizer)
            row.append(fit_result)
        cached_fits.append(row)

    return cached_fits
