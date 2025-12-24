"""
Common utilities and configurations for benchmarks.
"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import random

import numpy as np
from rework_pysatl_mpest.distributions import ContinuousDistribution

# --- CONSTANTS ---
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
RNG_GENERATOR = np.random.default_rng(RNG_SEED)

# Mapping string names to numpy dtypes
DTYPES_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}

# Standard sample sizes for scaling tests
SAMPLE_SIZES = [1000, 10000, 1000000]

# Shapes for generation benchmarks
GENERATE_SHAPES = {
    "Scalar": None,
    "Vector_1k": 1000,
    "Vector_1M": 1000000,
    "Matrix_Small": (100, 100),  # 10k elements
    "Matrix_Large": (1000, 1000),  # 1M elements
}

DISTRIBUTIONS = ["Beta", "Cauchy", "Exponential", "Normal", "Pareto", "Uniform", "Weibull"]


class LibAdapter:
    """
    Handles dynamic imports and API differences between old commits and new rework.
    """

    @classmethod
    def create_dist(cls, name, dtype=np.float64, **kwargs):
        try:
            import rework_pysatl_mpest.distributions as dists

            DistClass = getattr(dists, name)
        except (ImportError, AttributeError):
            try:
                from rework_pysatl_mpest import core

                DistClass = getattr(core, name)
            except AttributeError:
                raise NotImplementedError(f"Distribution {name} not found")

        try:
            dist = DistClass(dtype=dtype, **kwargs)
        except TypeError:
            if dtype != np.float64:
                raise NotImplementedError(f"Old version does not support {dtype}")

            dist = DistClass(**kwargs)

        return dist

    @classmethod
    def create_mixture(cls, components, dtype=np.float64, weights=None):
        try:
            from rework_pysatl_mpest.core import MixtureModel
        except ImportError:
            raise NotImplementedError("MixtureModel not found")

        try:
            model = MixtureModel(components=components, weights=weights, dtype=dtype)
        except TypeError:
            if dtype != np.float64:
                raise NotImplementedError(f"Old version does not support {dtype}")
            model = MixtureModel(components=components, weights=weights)

        return model


def get_components(dist_name, dtype=np.float64, n_components=2) -> list[ContinuousDistribution]:
    """
    Factory to create a list of components compatible with all versions.
    Ensures parameters are distinct enough to form valid clusters.

    Parameters
    ----------
    dist_name : str
        Name of the distribution class.
    dtype : type, optional
        Numpy dtype for the components.
    n_components : int, optional
        Number of components to generate.

    Returns
    -------
    list[ContinuousDistribution]
        List of instantiated distribution objects.
    """

    # Helper to cast parameters to correct type (if numpy) or float
    def d(val):
        return dtype(val) if hasattr(dtype, "__call__") else float(val)

    if dist_name == "Normal":
        return [
            LibAdapter.create_dist("Normal", dtype, loc=d(i * 5.0), scale=d(1.0 + i * 0.2)) for i in range(n_components)
        ]

    elif dist_name == "Exponential":
        return [
            LibAdapter.create_dist("Exponential", dtype, loc=d(i * 2.0), rate=d(1.0 / (i + 1)))
            for i in range(n_components)
        ]

    elif dist_name == "Pareto":
        return [
            LibAdapter.create_dist("Pareto", dtype, shape=d(max(0.5, 3.0 - i * 0.5)), scale=d(1.0 + i))
            for i in range(n_components)
        ]

    elif dist_name == "Weibull":
        return [
            LibAdapter.create_dist("Weibull", dtype, shape=d(3.0), loc=d(i * 5.0), scale=d(1.0 + i * 0.5))
            for i in range(n_components)
        ]

    elif dist_name == "Beta":
        return [
            LibAdapter.create_dist(
                "Beta",
                dtype,
                alpha=d(2.0 + i * 0.5),
                beta=d(max(0.1, 5.0 - i * 0.5)),
                left_border=d(0.0),
                right_border=d(1.0),
            )
            for i in range(n_components)
        ]

    elif dist_name == "Cauchy":
        return [LibAdapter.create_dist("Cauchy", dtype, loc=d(i * 5.0), scale=d(1.0)) for i in range(n_components)]

    elif dist_name == "Uniform":
        return [
            LibAdapter.create_dist("Uniform", dtype, left_border=d(i * 2), right_border=d(i * 2 + 1))
            for i in range(n_components)
        ]

    else:
        raise ValueError(f"Unknown distribution: {dist_name}")


class Benchmark:
    """
    Base class for all benchmarks.
    Can be used to set common timeouts, repeats, or warmup strategies.
    """

    pass
