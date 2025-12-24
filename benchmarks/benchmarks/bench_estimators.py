"""
Benchmarks for the ECM Estimator.
"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from copy import copy

import numpy as np
from rework_pysatl_mpest.estimators import ECM
from rework_pysatl_mpest.estimators.iterative import (
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    PipelineState,
)
from rework_pysatl_mpest.estimators.iterative.breakpointers import StepBreakpointer
from rework_pysatl_mpest.optimizers import ScipyNelderMead

from .common import DISTRIBUTIONS, DTYPES_MAP, RNG_GENERATOR, SAMPLE_SIZES, Benchmark, LibAdapter, get_components


class StepOverhead(Benchmark):
    """
    Isolates E-step and M-step to identify bottlenecks.

    Warning: The M-step benchmark operates on a mutable state.
    Since we only test analytical strategies (Q-Function for Exp/Normal/Pareto/Weibull),
    the calculation time is largely independent of the parameter values,
    so the benchmark drift is negligible.
    """

    params = (
        DISTRIBUTIONS,  # dist_name
        [2],  # n_components
        SAMPLE_SIZES,  # n_samples
        list(DTYPES_MAP.keys()),  # dtype_name
        [True, False],  # is_soft
    )
    param_names = ["dist_name", "n_components", "n_samples", "dtype_name", "is_soft"]

    def setup(self, dist_name, n_components, n_samples, dtype_name, is_soft):
        if dtype_name == "float16":
            warnings.simplefilter("ignore", RuntimeWarning)

        dtype = DTYPES_MAP[dtype_name]

        # --- Setup ---
        self.comps_analytical = get_components(dist_name, dtype, n_components)
        if dist_name == "Weibull":
            for comp in self.comps_analytical:
                comp.fix_param("shape")
                comp.fix_param("loc")

        self.mix_analytical = LibAdapter.create_mixture(components=self.comps_analytical, dtype=dtype)
        self.X_analytical = self.mix_analytical.generate(n_samples)

        # --- Pipeline Components ---
        self.e_step = ExpectationStep(is_soft=is_soft)

        # Setup States
        # 1. State ready for E-step
        self.state_analytical_for_E = PipelineState(self.X_analytical, None, None, copy(self.mix_analytical), None)

        # 2. State ready for M-step (Pre-calculate H)
        self.state_analytical_for_M = self.e_step.run(
            PipelineState(self.X_analytical, None, None, copy(self.mix_analytical), None)
        )

        # Optimization Blocks
        self.blocks_analytical = [
            OptimizationBlock(i, c.params_to_optimize, MaximizationStrategy.QFUNCTION)
            for i, c in enumerate(self.mix_analytical)
        ]
        self.m_step_analytical = MaximizationStep(self.blocks_analytical, ScipyNelderMead())

    # --- Benchmarks ---

    def time_expectation_step(self, dist_name, n_components, n_samples, dtype_name, is_soft):
        """
        Measure calculating responsibilities (logsumexp overhead).
        """
        self.e_step.run(self.state_analytical_for_E)

    def time_maximization_analytical(self, dist_name, n_components, n_samples, dtype_name, is_soft):
        """
        Measure M-step using closed-form formulas.
        This run twice:
        1. With Soft H matrix (floats)
        2. With Hard H matrix (0s and 1s)
        """
        if dist_name not in ["Exponential", "Normal", "Pareto", "Weibull"]:
            return
        self.m_step_analytical.run(self.state_analytical_for_M)


class ECMAnalyticalCleanWithStepBreakpointer(Benchmark):
    """
    Benchmarks the ECM estimator with StepBreakpointer and no overflow or errors.
    Measures the speed of vectorized NumPy updates.
    """

    params = (
        ["Normal", "Exponential", "Pareto", "Weibull"],  # dist_name
        [2],  # n_components
        [5],  # max_steps
        SAMPLE_SIZES,  # n_samples
        list(DTYPES_MAP.keys()),  # dtype_name
    )
    param_names = ["dist_name", "n_components", "max_steps", "n_samples", "dtype_name"]

    # Increase timeout as fitting can be slow
    timeout = 300.0

    def setup(self, dist_name, n_components, max_steps, n_samples, dtype_name):
        if dtype_name == "float16":
            warnings.simplefilter("ignore", RuntimeWarning)

        dtype = DTYPES_MAP[dtype_name]
        true_comps = get_components(dist_name, dtype, n_components)
        self.X = LibAdapter.create_mixture(components=true_comps).generate(n_samples)

        start_comps = copy(true_comps)
        for comp in start_comps:
            new_params = np.asarray(comp.get_params_vector(comp.params), dtype=dtype) + np.ones(
                len(comp.params), dtype=dtype
            )
            comp.set_params_from_vector(comp.params, new_params)
            if dist_name == "Weibull":
                comp.fix_param("shape")
                comp.fix_param("loc")

        self.start_mixture = LibAdapter.create_mixture(components=start_comps, dtype=dtype)

        # Configure ECM to run for a fixed small number of steps to measure throughput
        self.ecm = ECM(breakpointers=[StepBreakpointer(max_steps=max_steps)], pruners=[], optimizer=ScipyNelderMead())

    def time_fit(self, dist_name, n_components, max_steps, n_samples, dtype_name):
        self.ecm.fit(self.X, self.start_mixture)


class ECMAnalyticalOverflow(Benchmark):
    """
    Benchmarks the 'Error Recovery' path.
    Specifically tests the overhead of catching NumericalStabilityError and
    restarting the pipeline with higher precision.
    Only relevant for float16 where overflow is easy to trigger.
    """

    params = (
        ["Normal", "Exponential", "Pareto", "Weibull"],  # dist_name
        [2],  # n_components
        SAMPLE_SIZES,  # n_samples
        ["float16"],  # dtype_name
    )
    param_names = ["dist_name", "n_components", "n_samples", "dtype_name"]
    timeout = 300.0

    def setup(self, dist_name, n_components, n_samples, dtype_name):
        dtype = DTYPES_MAP[dtype_name]

        overflow_val = 100.0  # Creates gradients/exp values > 65504 for some distributions

        # Create data clustered around a value that stresses float16 math
        X = RNG_GENERATOR.normal(loc=overflow_val, scale=overflow_val * 0.05, size=n_samples).astype(dtype)
        if dist_name in ["Exponential", "Pareto", "Weibull"]:
            X = np.abs(X)
        self.X = X

        start_comps = get_components(dist_name, dtype, n_components)
        if dist_name == "Weibull":
            for comp in start_comps:
                comp.fix_param("shape")
                comp.fix_param("loc")

        self.start_mix = LibAdapter.create_mixture(components=start_comps, dtype=dtype)

        # Run only 1 step to trigger the error immediately and measure recovery overhead
        self.ecm = ECM(breakpointers=[StepBreakpointer(max_steps=1)], pruners=[], optimizer=ScipyNelderMead())

    def time_fit_restart(self, dist_name, n_components, n_samples, dtype_name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # This call should fail internally, catch the error, promote types, and finish 1 step.
            self.ecm.fit(self.X, self.start_mix)
