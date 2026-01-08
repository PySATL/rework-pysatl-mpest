"""
Benchmarks for the ECM Estimator.
"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

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
from rework_pysatl_mpest.estimators.iterative._strategies.q_function import q_function_strategy
from rework_pysatl_mpest.optimizers import ScipyNelderMead

from .common import (
    DISTRIBUTIONS,
    DTYPES_MAP,
    RNG_GENERATOR,
    SAMPLE_SIZES,
    Benchmark,
    LibAdapter,
    get_components,
    measure_peak_memory,
    MutableBenchmark,
)


class EStep(Benchmark):
    """
    Isolates E-step.
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
        dtype = DTYPES_MAP[dtype_name]

        components = get_components(dist_name, dtype, n_components)
        mixture = LibAdapter.create_mixture(components=components, dtype=dtype)
        X = mixture.generate(n_samples)
        self.state = PipelineState(X, None, None, mixture, None)

        self.e_step = ExpectationStep(is_soft=is_soft)

    # --- Time Benchmarks ---

    def time_expectation_step(self, dist_name, n_components, n_samples, dtype_name, is_soft):
        self.e_step.run(self.state)

    # --- Memory Benchmarks (Peak) ---

    @measure_peak_memory
    def track_peakmem_expectation_step(self, dist_name, n_components, n_samples, dtype_name, is_soft):
        self.e_step.run(self.state)


class MStepAnalytical(MutableBenchmark):
    """
    Isolates M-step with analitical solution for strategy.
    Involve mutable state or in-place operations.
    """

    params = (
        ["Exponential", "Normal", "Pareto", "Weibull"],  # dist_name
        [2],  # n_components
        SAMPLE_SIZES,  # n_samples
        list(DTYPES_MAP.keys()),  # dtype_name
    )
    param_names = ["dist_name", "n_components", "n_samples", "dtype_name"]

    def setup(self, dist_name, n_components, n_samples, dtype_name):
        registered_names = {cls.__name__ for cls in q_function_strategy.registry.keys()}
        if dist_name not in registered_names:
            raise NotImplementedError(f"Version does not support analytical q-function for {dist_name}")

        dtype = DTYPES_MAP[dtype_name]

        # --- Setup ---
        components = get_components(dist_name, dtype, n_components)
        if dist_name == "Weibull":
            for comp in components:
                comp.fix_param("shape")
                comp.fix_param("loc")

        mixture = LibAdapter.create_mixture(components=components, dtype=dtype)
        X = mixture.generate(n_samples)

        # State ready for M-step (Pre-calculate H)
        self.state = ExpectationStep(is_soft=False).run(PipelineState(X, None, None, mixture, None))

        # Optimization Blocks
        blocks = [
            OptimizationBlock(i, component.params_to_optimize, MaximizationStrategy.QFUNCTION)
            for i, component in enumerate(mixture)
        ]
        self.m_step = MaximizationStep(blocks, ScipyNelderMead())

    # --- Time Benchmarks ---

    def time_maximization_step_analytical(self, dist_name, n_components, n_samples, dtype_name):
        self.m_step.run(self.state)

    # --- Memory Benchmarks (Peak) ---

    @measure_peak_memory
    def track_peakmem_maximization_step_analytical(self, dist_name, n_components, n_samples, dtype_name):
        self.m_step.run(self.state)


# TODO: Add Breakpointer, Pruners in params or in common


class ECMAnalyticalClean(MutableBenchmark):
    """
    Benchmarks the ECM estimator and no overflow.
    Involve mutable state or in-place operations.
    """

    params = (
        ["Normal", "Exponential", "Pareto", "Weibull"],  # dist_name
        [2],  # n_components
        SAMPLE_SIZES,  # n_samples
        list(DTYPES_MAP.keys()),  # dtype_name
        [[StepBreakpointer(max_steps=5)]],  # breakpointers
        [[]],  # pruners
    )
    param_names = ["dist_name", "n_components", "n_samples", "dtype_name", "breakpointers", "pruners"]

    # Increase timeout as fitting can be slow
    timeout = 300.0

    def setup(self, dist_name, n_components, n_samples, dtype_name, breakpointers, pruners):
        registered_names = {cls.__name__ for cls in q_function_strategy.registry.keys()}
        if dist_name not in registered_names:
            raise NotImplementedError(f"Version does not support analytical q-function for {dist_name}")

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

        # This implementation ignores the optimizer as it does not require numerical optimization.
        self.ecm = ECM(breakpointers=breakpointers, pruners=pruners, optimizer=None)

    # --- Time Benchmarks ---

    def time_fit(self, dist_name, n_components, n_samples, dtype_name, breakpointers, pruners):
        self.ecm.fit(self.X, self.start_mixture)

    # --- Memory Benchmarks (Peak) ---

    @measure_peak_memory
    def track_peakmem_fit(self, dist_name, n_components, n_samples, dtype_name, breakpointers, pruners):
        self.ecm.fit(self.X, self.start_mixture)


class ECMAnalyticalOverflow(MutableBenchmark):
    """
    Benchmarks the 'Error Recovery' path.
    Involving mutable state or in-place operations.

    Specifically tests the overhead of catching NumericalStabilityError and
    restarting the pipeline with higher precision.

    Only relevant for float16 where overflow is easy to trigger.
    """

    params = (
        ["Normal", "Exponential", "Pareto", "Weibull"],  # dist_name
        [2],  # n_components
        SAMPLE_SIZES,  # n_samples
    )
    param_names = ["dist_name", "n_components", "n_samples"]

    # Increase timeout as fitting can be slow
    timeout = 300.0

    def setup(self, dist_name, n_components, n_samples):
        try:
            from rework_pysatl_mpest.estimators.iterative._strategies.utils import handle_numerical_overflow
        except ImportError:
            raise NotImplementedError(f"Version does not support analytical overflow")

        dtype = np.float16
        from rework_pysatl_mpest.estimators.iterative._strategies.q_function import q_function_strategy

        registered_names = {cls.__name__ for cls in q_function_strategy.registry.keys()}
        if dist_name not in registered_names:
            raise NotImplementedError(f"Skipping {dist_name}: no analytical Q-function")

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

        self.start_mixture = LibAdapter.create_mixture(components=start_comps, dtype=dtype)

        # This implementation ignores the optimizer as it does not require numerical optimization
        # Run only 1 step to trigger the error immediately and measure recovery overhead
        self.ecm = ECM(breakpointers=[StepBreakpointer(max_steps=1)], pruners=[], optimizer=ScipyNelderMead())

    # --- Time Benchmarks ---

    def time_fit_restart(self, dist_name, n_components, n_samples):
        # This call should fail internally, catch the error, promote types, and finish 1 step.
        self.ecm.fit(self.X, self.start_mixture)

    # --- Memory Benchmarks (Peak) ---

    @measure_peak_memory
    def track_peakmem_fit_restart(self, dist_name, n_components, n_samples):
        # This call should fail internally, catch the error, promote types, and finish 1 step.
        self.ecm.fit(self.X, self.start_mixture)
