"""A module providing strategies for estimating the parameters of mixture components."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Callable

from ....distributions import ContinuousDistribution
from ....optimizers import Optimizer
from ....typings import DType
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock
from .q_function import q_function_strategy as _q_function_strategy

q_function_strategy: Callable[
    [ContinuousDistribution, PipelineState, OptimizationBlock, Optimizer], tuple[int, dict[str, DType]]
] = _q_function_strategy


__all__ = ["q_function_strategy"]
