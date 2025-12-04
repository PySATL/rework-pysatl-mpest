"""Tests for the base implementation of the Moments strategy."""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from unittest.mock import Mock

import numpy as np
import pytest
from rework_pysatl_mpest.core import Parameter
from rework_pysatl_mpest.distributions import ContinuousDistribution
from rework_pysatl_mpest.estimators.iterative import OptimizationBlock, PipelineState
from rework_pysatl_mpest.estimators.iterative._strategies import moments_strategy
from rework_pysatl_mpest.optimizers import Optimizer


class DummyDistribution(ContinuousDistribution):
    """
    A simple dummy implementation of ContinuousDistribution for testing purposes.
    It has two parameters: 'param1' and 'param2'.
    """

    param1 = Parameter()
    param2 = Parameter()

    def __init__(self, param1: float, param2: float, dtype: np.floating = np.float64):
        super().__init__(dtype=dtype)
        self.param1 = param1
        self.param2 = param2

    @property
    def name(self) -> str:
        return "Dummy"

    @property
    def params(self) -> set[str]:
        return {"param1", "param2"}

    def pdf(self, X):
        return np.array([])

    def ppf(self, P):
        return np.array([])

    def lpdf(self, X):
        return np.log(np.array([0.5] * len(X), dtype=self.dtype))

    def log_gradients(self, X):
        return np.array([])

    def generate(self, size: int):
        return np.array([])


def test_moments_strategy_base_implementation_raises_not_implemented():
    """
    Verifies that the base implementation of `moments_strategy` (via singledispatch)
    raises a NotImplementedError when called with a distribution that has no
    specifically registered implementation.
    """

    # Arrange
    dist = DummyDistribution(param1=1.0, param2=2.0)

    # Mocks for arguments that are not used before the error is raised
    mock_state = Mock(spec=PipelineState)
    mock_block = Mock(spec=OptimizationBlock)
    mock_optimizer = Mock(spec=Optimizer)

    expected_error_msg = "Moments strategy is not implemented for distribution 'Dummy'."

    # Act & Assert
    with pytest.raises(NotImplementedError, match=expected_error_msg):
        moments_strategy(dist, mock_state, mock_block, mock_optimizer)
