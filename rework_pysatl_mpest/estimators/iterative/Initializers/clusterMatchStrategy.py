from functools import singledispatch

import numpy as np
from scipy.stats import norm

from rework_pysatl_mpest import ContinuousDistribution, Exponential


@singledispatch
def find_best_cluster_for_model(
    model: ContinuousDistribution, clusters: dict[int, np.ndarray], min_samples=10
    ) -> tuple[int | None, tuple[type[ContinuousDistribution], dict[str, float]], float]:
    best_k, best_params, best_score = None, {}, -np.inf
    for k, X_k in clusters.items():
        x_flat = X_k.flatten()
        if len(x_flat) < min_samples:
            continue
        try:
            mean = np.mean(x_flat)
            std = np.clip(np.std(x_flat), 0.1, 100.0)
            params = {'mean': float(mean), 'std': float(std)}
            score = np.sum(np.clip(norm.logpdf(x_flat, mean, std), -1e10, 1e10))
            if score > best_score:
                best_score = score
                best_k = k
                best_params = params
        except ValueError:
            continue
    return best_k, (ContinuousDistribution, best_params), best_score


@find_best_cluster_for_model.register(Exponential)
def _(
    model: Exponential, clusters: dict[int, np.ndarray], min_samples=10
    ) -> tuple[int | None, tuple[type[Exponential], dict[str, float]], float]:
    best_k, best_params, best_score = None, {}, -np.inf
    for k, X_k in clusters.items():
        x_flat = X_k.flatten()
        if len(x_flat) < min_samples:
            continue
        try:
            loc_est = float(np.min(x_flat) - 1e-6)

            above_loc = x_flat[x_flat > loc_est]
            if len(above_loc) == 0:
                continue

            scale_est = np.mean(above_loc) - loc_est
            rate_est = 1.0 / scale_est

            loc_est = float(np.clip(loc_est, -100.0, np.min(x_flat) - 1e-6))
            rate_est = float(np.clip(rate_est, 0.01, 100.0))

            temp_dist = Exponential(loc=loc_est, rate=rate_est)

            log_likelihood = np.sum(temp_dist.lpdf(x_flat))
            score = float(np.clip(log_likelihood, -1e10, 1e10))

            if score > best_score:
                best_score = score
                best_k = k
                best_params = {'loc': loc_est, 'rate': rate_est}

        except (ValueError, ZeroDivisionError, RuntimeWarning):
            continue
    return best_k, (Exponential, best_params), best_score
