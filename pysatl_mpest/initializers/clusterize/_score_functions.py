"""
A module providing score functions (e.g AIC, likelihood for both mixture and single component)
for cluster match methods.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from ...core import MixtureModel
from ...distributions import ContinuousDistribution


def _calculate_component_log_likelihood(model: ContinuousDistribution, X: np.ndarray, H_k: np.ndarray) -> float:
    """Calculates the weighted log-likelihood for a single component.

    Computes the log-probability density of the data under the given model,
    weighted by the cluster responsibilities `H_k`. Values are clipped for
    numerical stability to avoid log(0).

    Parameters
    ----------
    model : ContinuousDistribution
        The distribution component to evaluate.
    X : np.ndarray
        Input data points.
    H_k : np.ndarray
        Weight vector (responsibilities) where H_k[i] is the probability
        that point X[i] belongs to this component.

    Returns
    -------
    float
        The weighted sum of log-probabilities.
    """
    log_probs = np.clip(model.lpdf(X), a_min=-1e9, a_max=1e9)
    return np.sum(H_k * log_probs)


def _calculate_mixture_log_likelihood(model: MixtureModel, X: np.ndarray) -> float:
    """Calculates the total log-likelihood for a mixture model.

    Parameters
    ----------
    model : MixtureModel
        The full mixture model containing all components and weights.
    X : np.ndarray
        Input data points.

    Returns
    -------
    float
        The sum of log-likelihoods of all data points under the mixture model.
    """
    return model.loglikelihood(X)


def _calculate_component_aic(model: ContinuousDistribution, X: np.ndarray, H_k: np.ndarray) -> float:
    """Calculates AIC for a single component based on weighted log-likelihood.

    AIC = 2k - 2ln(L), where k is the number of parameters in the component
    and L is the weighted likelihood.

    Parameters
    ----------
    model : ContinuousDistribution
        The distribution component.
    X : np.ndarray
        Input data points.
    H_k : np.ndarray
        Weight vector for this component.

    Returns
    -------
    float
        The Akaike Information Criterion score.
    """
    weighted_log_likelihood = _calculate_component_log_likelihood(model, X, H_k)
    k_params = len(model.params)
    return 2 * k_params - 2 * weighted_log_likelihood


def _calculate_mixture_aic(model: MixtureModel, X: np.ndarray) -> float:
    """Calculates AIC for the entire mixture model.

    Takes into account parameters of all components as well as the mixture weights
    (degrees of freedom).

    Parameters
    ----------
    model : MixtureModel
        The full mixture model.
    X : np.ndarray
        Input data points.

    Returns
    -------
    float
        The Akaike Information Criterion score for the mixture.

    Notes
    -----
    The number of parameters (k) is calculated as:
    Sum of parameters of all components + (Number of components - 1).
    The (-1) accounts for the constraint that weights must sum to 1.
    """
    log_likelihood = _calculate_mixture_log_likelihood(model, X)
    k_params = sum(len(dist.params) for dist in model.components)

    k_params += model.n_components - 1

    return 2 * k_params - 2 * log_likelihood
