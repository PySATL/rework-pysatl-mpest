"""Maximization step strategies for updating mixture component parameters."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable
from typing import Any

from pysatl_core.families import ParametricFamilyDistribution

from ....optimizers import Optimizer
from ..pipeline_state import PipelineState
from ..steps.block import OptimizationBlock
from .moments import moments_strategy as _moments_strategy
from .observed_data_likelihood import observed_data_likelihood_strategy as _observed_data_likelihood_strategy
from .q_function import q_function_strategy as _q_function_strategy

type StrategyFunction = Callable[
    [ParametricFamilyDistribution, PipelineState, OptimizationBlock, Optimizer],
    tuple[int, dict[str, float]],
]

q_function_strategy: StrategyFunction = _q_function_strategy
observed_data_likelihood_strategy: StrategyFunction = _observed_data_likelihood_strategy
moments_strategy: StrategyFunction = _moments_strategy

__all__ = ["StrategyFunction", "moments_strategy", "observed_data_likelihood_strategy", "q_function_strategy"]
