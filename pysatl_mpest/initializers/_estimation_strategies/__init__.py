"""
A subpackage containing estimation strategies for initialization.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .q_function import q_function_strategy, q_function_strategy_exponential

__all__ = [
    "q_function_strategy",
    "q_function_strategy_exponential",
]
