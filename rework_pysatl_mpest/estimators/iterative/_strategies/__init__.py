"""A module providing strategies for estimating the parameters of mixture components."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable

from ....distributions import ContinuousDistribution
from ....optimizers import Optimizer
from ....typings import DType
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock
from .moments import moments_strategy as _moments_strategy
from .observed_data_likelihood import observed_data_likelihood_strategy as _observed_data_likelihood_strategy
from .q_function import q_function_strategy as _q_function_strategy

q_function_strategy: Callable[
    [ContinuousDistribution, PipelineState, OptimizationBlock, Optimizer], tuple[int, dict[str, DType]]
] = _q_function_strategy

observed_data_likelihood_strategy: Callable[
    [ContinuousDistribution, PipelineState, OptimizationBlock, Optimizer], tuple[int, dict[str, DType]]
] = _observed_data_likelihood_strategy

moments_strategy: Callable[
    [ContinuousDistribution, PipelineState, OptimizationBlock, Optimizer], tuple[int, dict[str, DType]]
] = _moments_strategy


__all__ = ["moments_strategy", "observed_data_likelihood_strategy", "q_function_strategy"]
