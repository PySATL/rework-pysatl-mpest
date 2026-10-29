"""Mock implementation for Optimizer."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

from pysatl_mpest.optimizers.optimizer import Optimizer


class MockOptimizer(Optimizer):
    """Minimal mock returning input parameters without optimization."""

    def minimize(self, target: Callable, params: list[float]) -> list[float]:
        return params
