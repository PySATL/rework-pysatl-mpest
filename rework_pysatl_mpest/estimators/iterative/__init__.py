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

Core Components
---------------
- `Pipeline`: The main orchestrator class that manages the iterative process.
- `PipelineState`: A data container (dataclass) that holds the current state
  of the pipeline and is passed between all its parts.
- `PipelineStep`: An abstract class for a single step in the pipeline
  (e.g., an E-step or M-step).
- `Breakpointer`: An abstract class for defining the termination condition for
  the iterative process (e.g., based on convergence or the number of iterations).
- `Pruner`: An abstract class for implementing a strategy to remove (prune)
  components from the mixture during the algorithm's execution.

Available Implementations
-------------------------

The concrete implementations of the base classes, available for use when
building a pipeline, will be listed below.

Steps (`PipelineStep`):
~~~~~~~~~~~~~~~~~~~~~~~
- ... (A list and brief description of implemented steps will be here)

Breakpointers (`Breakpointer`):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ... (A list and brief description of implemented breakpointers will be here)

Pruners (`Pruner`):
~~~~~~~~~~~~~~~~~~~
- ... (A list and brief description of implemented pruners will be here)


Usage Example
-------------

"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from rework_pysatl_mpest.estimators.iterative.pipeline import Pipeline
from rework_pysatl_mpest.estimators.iterative.pipeline_step import PipelineStep
from rework_pysatl_mpest.estimators.iterative.pipeline_state import PipelineState
from rework_pysatl_mpest.estimators.iterative.breakpointer import Breakpointer
from rework_pysatl_mpest.estimators.iterative.pruner import Pruner


__all__ = [
        "Pipeline",
        "PipelineState",
        "PipelineStep",
        "Breakpointer",
        "Pruner",
]
