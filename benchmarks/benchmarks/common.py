"""
Common utilities and configurations for benchmarks.
"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import random
import numpy as np
from rework_pysatl_mpest.distributions import (
    Beta,
    Cauchy,
    Exponential,
    Normal,
    Pareto,
    Uniform,
    Weibull,
)

# Various pre-crafted datasets/variables for testing
# !!! Must not be changed -- only appended !!!
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
    "Matrix_Small": (100, 100),    # 10k elements
    "Matrix_Large": (1000, 1000),  # 1M elements
}

DISTRIBUTIONS = ["Beta", "Cauchy", "Exponential", "Normal", "Pareto", "Uniform", "Weibull"]

def get_components(dist_name, dtype=np.float64, n_components=2):
    """
    Factory to create a list of components for a mixture.
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

    if dist_name == "Normal":
        return [Normal(loc=dtype(i * 5.0), scale=dtype(1.0 + i * 0.2), dtype=dtype)
                for i in range(n_components)]

    elif dist_name == "Exponential":
        return [Exponential(loc=dtype(i * 2.0), rate=dtype(1.0 / (i + 1)), dtype=dtype)
                for i in range(n_components)]

    elif dist_name == "Pareto":
        return [Pareto(shape=dtype(max(0.5, 3.0 - i * 0.5)), scale=dtype(1.0 + i), dtype=dtype)
                for i in range(n_components)]

    elif dist_name == "Weibull":
        return [Weibull(shape=dtype(3.0), loc=dtype(i * 5.0), scale=dtype(1.0 + i * 0.5), dtype=dtype)
                for i in range(n_components)]

    elif dist_name == "Beta":
        return [Beta(alpha=dtype(2.0 + i * 0.5), beta=dtype(max(0.1, 5.0 - i * 0.5)), left_border=dtype(0.0), right_border=dtype(1.0),
                     dtype=dtype)
                for i in range(n_components)]

    elif dist_name == "Cauchy":
        return [Cauchy(loc=dtype(i * 5.0), scale=dtype(1.0), dtype=dtype)
                for i in range(n_components)]

    elif dist_name == "Uniform":
        return [Uniform(left_border=dtype(i * 2), right_border=dtype(i * 2 + 1), dtype=dtype)
                for i in range(n_components)]

    else:
        raise ValueError(f"Unknown distribution: {dist_name}")


class Benchmark:
    """
    Base class for all benchmarks.
    Can be used to set common timeouts, repeats, or warmup strategies.
    """
    pass
