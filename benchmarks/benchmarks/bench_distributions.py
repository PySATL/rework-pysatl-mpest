"""
Benchmarks for individual ContinuousDistribution classes.
"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from copy import copy

from .common import DISTRIBUTIONS, DTYPES_MAP, GENERATE_SHAPES, RNG_GENERATOR, SAMPLE_SIZES, Benchmark, get_components


class DistributionMethods(Benchmark):
    """
    Benchmarks for computational methods (PDF, LPDF, PPF, Gradients).
    Operates on 1D arrays of size `n_samples`.
    """

    params = (
        DISTRIBUTIONS,  # dist_name
        SAMPLE_SIZES,  # n_samples
        list(DTYPES_MAP.keys()),  # dtype_name
    )
    param_names = ["dist_name", "n_samples", "dtype_name"]

    def setup(self, dist_name, n_samples, dtype_name):
        if dtype_name == "float16":
            warnings.simplefilter("ignore", RuntimeWarning)

        dtype = DTYPES_MAP[dtype_name]
        # Initialize distribution
        self.dist = get_components(dist_name, dtype, 1)[0]

        # Generate X via the distribution itself to ensure valid domain support
        self.X = self.dist.generate(n_samples).astype(dtype)
        # Probabilities for PPF
        self.P = RNG_GENERATOR.uniform(0.01, 0.99, size=n_samples).astype(dtype)

    # --- Time Benchmarks ---

    def time_pdf(self, dist_name, n_samples, dtype_name):
        self.dist.pdf(self.X)

    def time_lpdf(self, dist_name, n_samples, dtype_name):
        self.dist.lpdf(self.X)

    def time_ppf(self, dist_name, n_samples, dtype_name):
        self.dist.ppf(self.P)

    def time_log_gradients(self, dist_name, n_samples, dtype_name):
        self.dist.log_gradients(self.X)

    # --- Memory Benchmarks (Peak) ---

    def peakmem_pdf(self, dist_name, n_samples, dtype_name):
        self.dist.pdf(self.X)

    def peakmem_log_gradients(self, dist_name, n_samples, dtype_name):
        self.dist.log_gradients(self.X)


class DistributionGenerate(Benchmark):
    """
    Benchmarks for Random Variate Generation (RVG).
    Tests scalar, vector, and matrix generation.
    """

    params = (
        DISTRIBUTIONS,  # dist_name
        list(GENERATE_SHAPES.keys()),  # shape_name
        list(DTYPES_MAP.keys()),  # dtype_name
    )
    param_names = ["dist_name", "shape_name", "dtype_name"]

    def setup(self, dist_name, shape_name, dtype_name):
        if dtype_name == "float16":
            warnings.simplefilter("ignore", RuntimeWarning)

        dtype = DTYPES_MAP[dtype_name]
        self.dist = get_components(dist_name, dtype, 1)[0]
        self.shape = GENERATE_SHAPES[shape_name]

    # --- Time Benchmarks ---

    def time_generate(self, dist_name, shape_name, dtype_name):
        self.dist.generate(self.shape)

    # --- Memory Benchmarks (Peak) ---

    def peakmem_generate(self, dist_name, shape_name, dtype_name):
        self.dist.generate(self.shape)


class DistributionAstype(Benchmark):
    """
    Benchmarks for casting distributions to different precisions.
    """

    params = (
        DISTRIBUTIONS,  # dist_name
        list(DTYPES_MAP.keys()),  # dtype_name
        list(DTYPES_MAP.keys()),  # conv_dtype_name
    )
    param_names = ["dist_name", "dtype_name", "conv_dtype_name"]

    def setup(self, dist_name, dtype_name, conv_dtype_name):
        dtype = DTYPES_MAP[dtype_name]
        self.dist = get_components(dist_name, dtype, 1)[0]
        self.conv_dtype = DTYPES_MAP[conv_dtype_name]

        if not callable(getattr(self.dist, "astype", None)):
            raise NotImplementedError(f"Old version {dist_name} does not support .astype")

    # --- Time Benchmarks ---

    def time_astype(self, dist_name, dtype_name, conv_dtype_name):
        self.dist.astype(self.conv_dtype)

    # --- Memory Benchmarks (Peak) ---

    def peakmem_astype(self, dist_name, dtype_name, conv_dtype_name):
        self.dist.astype(self.conv_dtype)


class DistributionCopy(Benchmark):
    """
    Benchmarks for object copying overhead.
    High frequency usage in Pipeline state saving and Optimization strategies.
    """

    params = (
        DISTRIBUTIONS,  # dist_name
        list(DTYPES_MAP.keys()),  # dtype_name
    )
    param_names = ["dist_name", "dtype_name"]

    def setup(self, dist_name, dtype_name):
        dtype = DTYPES_MAP[dtype_name]
        self.dist = get_components(dist_name, dtype, 1)[0]

    # --- Time Benchmarks ---

    def time_copy(self, dist_name, dtype_name):
        copy(self.dist)

    # --- Memory Benchmarks (Peak) ---

    def peakmem_copy(self, dist_name, dtype_name):
        copy(self.dist)


class ParameterOverhead(Benchmark):
    """
    Micro-benchmarks for Parameter descriptor overhead.
    Critical for numerical optimization loops where these are accessed frequently.
    """

    def setup(self):
        self.dist = get_components("Weibull", n_components=1)[0]
        self.param_names = ["shape", "loc", "scale"]
        self.new_values = [1.6, 0.1, 2.1]

    # --- Time Benchmarks ---

    def time_set_params_from_vector(self):
        self.dist.set_params_from_vector(self.param_names, self.new_values)

    def time_get_params_vector(self):
        self.dist.get_params_vector(self.param_names)
