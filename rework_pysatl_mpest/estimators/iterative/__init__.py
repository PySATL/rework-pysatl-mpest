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


from .breakpointer import Breakpointer
from .breakpointers import StepBreakpointer, LikelihoodBreakpointer
from .pipeline import Pipeline
from .pipeline_state import PipelineState
from .pipeline_step import PipelineStep
from .pruner import Pruner
from .pruners import PriorThresholdPruner
from .steps import ExpectationStep, MaximizationStep, MaximizationStrategy, OptimizationBlock

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
    "LikelihoodBreakpointer"
]
