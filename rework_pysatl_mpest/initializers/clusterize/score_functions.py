import numpy as np

from ...core import MixtureModel
from ...distributions import ContinuousDistribution


def _calculate_component_log_likelihood(model: ContinuousDistribution, X: np.ndarray, H_k: np.ndarray) -> float:
    """Calculates the weighted log-likelihood for a single component"""
    log_probs = np.clip(model.lpdf(X), -1e9, -1e-9)
    return np.sum(H_k * log_probs)


def _calculate_mixture_log_likelihood(model: MixtureModel, X: np.ndarray) -> float:
    """Calculates the total log-likelihood for a mixture model"""
    return np.sum(model.loglikelihood(X))


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