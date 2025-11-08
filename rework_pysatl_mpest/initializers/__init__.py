"""
initializers module for mixture model parameter initialization.

This module provides various strategies for initializing parameters of mixture models
before the main optimization process. Good initialization is crucial for achieving
fast convergence and high-quality solutions in mixture model estimation.

**Purpose**

initializers provide good starting points for EM algorithm and other optimization
methods, helping to avoid poor local optima and improving convergence.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from ._estimation_strategies.q_function import q_function_strategy, q_function_strategy_exponential
from .clusterize_initializer import ClusterizeInitializer
from .initializer import Initializer
from .strategies import EstimationStrategy

__all__ = [
    "ClusterizeInitializer",
    "EstimationStrategy",
    "Initializer",
    "q_function_strategy",
    "q_function_strategy_exponential",
]
