"""
Benchmarks for the core MixtureModel class.
"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from copy import copy

from rework_pysatl_mpest.core import MixtureModel
from .common import Benchmark, DTYPES_MAP, GENERATE_SHAPES, SAMPLE_SIZES, get_components

class MixtureMethods(Benchmark):
    """
    Benchmarks for MixtureModel computational methods.
    """

    params = (
        [10], # n_components
        SAMPLE_SIZES,  # n_samples
        list(DTYPES_MAP.keys())  # dtype_name
    )
    param_names = ["n_components", "n_samples", "dtype_name"]

    def setup(self, n_components, n_samples, dtype_name):
        if dtype_name == "float16":
            warnings.simplefilter("ignore", RuntimeWarning)

        dtype = DTYPES_MAP[dtype_name]
        components = get_components("Normal", dtype, n_components)

        self.mixture = MixtureModel(components=components, dtype=dtype)
        # Pre-generate data to avoid measuring generation time
        self.X = self.mixture.generate(n_samples)

    # --- Time Benchmarks ---

    def time_pdf(self, n_components, n_samples, dtype_name):
        self.mixture.pdf(self.X)

    def time_lpdf(self, n_components, n_samples, dtype_name):
        self.mixture.lpdf(self.X)

    def time_loglikelihood(self, n_components, n_samples, dtype_name):
        self.mixture.loglikelihood(self.X)

    # --- Memory Benchmarks (Peak) ---

    def peakmem_pdf(self, n_components, n_samples, dtype_name):
        self.mixture.pdf(self.X)

    def peakmem_lpdf(self, n_components, n_samples, dtype_name):
        self.mixture.lpdf(self.X)

    def peakmem_loglikelihood(self, n_components, n_samples, dtype_name):
        self.mixture.loglikelihood(self.X)


class MixtureScalability(Benchmark):
    """
    Tests how MixtureModel scales with the number of components (K).
    Critical for identifying O(K) vs O(K^2) bottlenecks.
    """
    params = (
        [2, 5, 20, 100],  # n_components
        [10000],  # n_samples
    )
    param_names = ["n_components", "n_samples"]

    def setup(self, n_components, n_samples):
        # Create K distinct components
        components = get_components("Normal", n_components=n_components)

        self.mixture = MixtureModel(components)
        self.X = self.mixture.generate(n_samples)

    # --- Time Benchmarks ---

    def time_pdf_scaling(self, n_components, n_samples):
        self.mixture.pdf(self.X)

    def time_lpdf_scaling(self, n_components, n_samples):
        self.mixture.lpdf(self.X)

    def time_loglikelihood_scaling(self, n_components, n_samples):
        self.mixture.loglikelihood(self.X)

    # --- Memory Benchmarks (Peak) ---

    def peakmem_pdf_scaling(self, n_components, n_samples):
        self.mixture.pdf(self.X)

    def peakmem_lpdf_scaling(self, n_components, n_samples):
        self.mixture.lpdf(self.X)

    def peakmem_loglikelihood_scaling(self, n_components, n_samples):
        self.mixture.loglikelihood(self.X)


class MixtureGenerate(Benchmark):
    """
    Benchmarks for MixtureModel sampling.
    """

    params = (
        [2, 5, 20, 100],  # n_components
        list(GENERATE_SHAPES.keys()),  # shape_name
        list(DTYPES_MAP.keys())  # dtype_name
    )
    param_names = ["n_components", "shape_name", "dtype_name"]

    def setup(self, n_components, shape_name, dtype_name):
        if dtype_name == "float16":
            warnings.simplefilter("ignore", RuntimeWarning)

        dtype = DTYPES_MAP[dtype_name]
        components = get_components("Normal", n_components=n_components)

        self.mixture = MixtureModel(components=components, dtype=dtype)
        self.shape = GENERATE_SHAPES[shape_name]

    # --- Time Benchmarks ---

    def time_generate(self, n_components, shape_name, dtype_name):
        self.mixture.generate(self.shape)

    # --- Memory Benchmarks (Peak) ---

    def peakmem_generate(self, n_components, shape_name, dtype_name):
        self.mixture.generate(self.shape)


class MixtureAstype(Benchmark):
    """
    Benchmarks for casting MixtureModels.
    """

    params = (
        [2, 5, 20, 100],  # n_components
        list(DTYPES_MAP.keys()),  # dtype_name
        list(DTYPES_MAP.keys())  # conv_dtype_name
    )
    param_names = ["n_components", "dtype_name", "conv_dtype_name"]

    def setup(self, n_components, dtype_name, conv_dtype_name):
        dtype = DTYPES_MAP[dtype_name]
        components = get_components("Normal", dtype, n_components)

        self.mixture = MixtureModel(components=components, dtype=dtype)
        self.conv_dtype = DTYPES_MAP[conv_dtype_name]

    # --- Time Benchmarks ---

    def time_astype(self, n_components, dtype_name, conv_dtype_name):
        self.mixture.astype(self.conv_dtype)

    # --- Memory Benchmarks (Peak) ---

    def peakmem_astype(self, n_components, dtype_name, conv_dtype_name):
        self.mixture.astype(self.conv_dtype)

class MixtureCopy(Benchmark):
    """
    Benchmarks for object copying overhead.
    High frequency usage in Pipeline state saving.
    """

    params = (
        [2, 5, 20, 100],  # n_components
        list(DTYPES_MAP.keys())  # dtype_name
    )
    param_names = ["n_components", "dtype_name"]

    def setup(self, n_components, dtype_name):
        dtype = DTYPES_MAP[dtype_name]
        components = get_components("Normal", dtype, n_components)
        self.mixture = MixtureModel(components=components, dtype=dtype)

    # --- Time Benchmarks ---

    def time_copy(self, n_components, dtype_name):
        copy(self.mixture)

    # --- Memory Benchmarks (Peak) ---

    def peakmem_copy(self, n_components, dtype_name):
        copy(self.mixture)
