"""Provides a flexible framework for building iterative algorithms.

This package contains components for creating customizable iterative processes for
estimating parameters of mixture models, such as the Expectation-Maximization (EM)
algorithm and its variations.

The core idea is to construct a `Pipeline` that consists of a sequence of
`PipelineStep`s. This pipeline cyclically executes the defined steps until one
of the stopping conditions (`Breakpointer`) is met. Additionally, after each
iteration, `Pruner` strategies can be applied to remove insignificant components
from the model. The state of the entire process at each iteration is stored in a
`PipelineState` object.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from rework_pysatl_mpest.estimators.iterative.breakpointer import Breakpointer
from rework_pysatl_mpest.estimators.iterative.breakpointers.step_breakpointer import StepBreakpointer
from rework_pysatl_mpest.estimators.iterative.pipeline import Pipeline
from rework_pysatl_mpest.estimators.iterative.pipeline_state import PipelineState
from rework_pysatl_mpest.estimators.iterative.pipeline_step import PipelineStep
from rework_pysatl_mpest.estimators.iterative.pruner import Pruner
from rework_pysatl_mpest.estimators.iterative.pruners.prior_threshold_pruner import PriorThresholdPruner
from rework_pysatl_mpest.estimators.iterative.steps.block import MaximizationStrategy, OptimizationBlock
from rework_pysatl_mpest.estimators.iterative.steps.expectation_step import ExpectationStep
from rework_pysatl_mpest.estimators.iterative.steps.maximization_step import MaximizationStep

__all__ = [
    "Breakpointer",
    "ExpectationStep",
    "MaximizationStep",
    "MaximizationStrategy",
    "OptimizationBlock",
    "Pipeline",
    "PipelineState",
    "PipelineStep",
    "PriorThresholdPruner",
    "Pruner",
    "StepBreakpointer",
]
