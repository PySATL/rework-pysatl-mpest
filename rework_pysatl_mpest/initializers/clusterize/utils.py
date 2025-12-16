from copy import copy
from typing import Callable, Any, TypedDict

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

    Ensures that the responsibility matrix H sums to 1, matches dimensions with
    provided models, and filters out clusters with insufficient samples.
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

    Creates a copy of the model, estimates parameters using the provided strategy
    on the weighted data (H_k), and calculates the fit score.
    """
    temp_model = copy(model)
    new_params: dict[str, float] = estimation_func(temp_model, X, H_k, optimizer)
    temp_model.set_params_from_vector(list(new_params.keys()), list(new_params.values()))

    score = score_func(temp_model, X, H_k)
    weight = float(np.sum(H_k) / len(X))

    return {"model": temp_model, "params": new_params, "score": score, "weight": weight}


def _precompute_fits(context: Context) -> list[list[FitResult]]:
    """Pre-computes and caches parameter estimates and scores for all pairs.

    Iterates through every distribution model and every valid cluster to estimate
    parameters and calculate the goodness-of-fit score. This results in a cost
    matrix used by global optimization strategies (Hungarian and Permutations).

    Parameters
    ----------
    context : Context
        The execution context containing models, data (X), weights (H),
        optimizers, and scoring functions.

    Returns
    -------
    list[list[FitResult]]
        A matrix (list of lists) where entry [i][j] contains the fit result
        (params, score, weight) for the i-th model and the j-th cluster.
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