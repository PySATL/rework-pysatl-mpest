"""A module that provides tests for q_function"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import ClassVar
from unittest.mock import Mock

import numpy as np
from pysatl_mpest.distributions import Exponential
from pysatl_mpest.initializers._estimation_strategies.q_function import NUMERICAL_TOLERANCE, q_function_strategy
from pysatl_mpest.optimizers import Optimizer

COMPARISON_CONSTANT = 1e-10


class TestQFunctionStrategyExponential:
    def setup_method(self):
        self.mock_optimizer = Mock(spec=Optimizer)
        self.component = Exponential(loc=1.0, rate=2.0)

    def test_normal_case(self):
        X = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
        H_j = np.array([0.8, 0.9, 1.0, 0.7, 0.6])

        result = q_function_strategy(self.component, X, H_j, self.mock_optimizer)

        assert Exponential.PARAM_LOC in result
        assert Exponential.PARAM_RATE in result

        relevant_X = X[H_j > NUMERICAL_TOLERANCE]
        expected_loc = np.min(relevant_X).item()
        assert result[Exponential.PARAM_LOC] == expected_loc

        weighted_sum = np.dot(H_j, np.maximum(X - result[Exponential.PARAM_LOC], NUMERICAL_TOLERANCE)).item()
        N_j = np.sum(H_j).item()
        expected_rate = N_j / weighted_sum
        assert result[Exponential.PARAM_RATE] == expected_rate

    def test_all_H_j_below_tolerance(self):
        original_loc = self.component.loc
        original_rate = self.component.rate
        X = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
        H_j = np.array([NUMERICAL_TOLERANCE] * 5) / 10

        result = q_function_strategy(self.component, X, H_j, self.mock_optimizer)

        assert Exponential.PARAM_LOC in result
        assert Exponential.PARAM_RATE in result

        assert result[Exponential.PARAM_LOC] == original_loc
        assert abs(result[Exponential.PARAM_RATE] - original_rate) < COMPARISON_CONSTANT

    def test_weighted_sum_below_tolerance(self):
        X = np.array([1.001, 1.002, 1.003])
        H_j = np.array([0.8, 0.9, 1.0])

        result = q_function_strategy(self.component, X, H_j, self.mock_optimizer)

        assert Exponential.PARAM_LOC in result
        assert Exponential.PARAM_RATE in result

        relevant_X = X[H_j > NUMERICAL_TOLERANCE]
        expected_loc = np.min(relevant_X).item()
        assert result[Exponential.PARAM_LOC] == expected_loc

        weighted_sum = np.dot(H_j, np.maximum(X - result[Exponential.PARAM_LOC], NUMERICAL_TOLERANCE)).item()
        N_j = np.sum(H_j).item()
        expected_rate = N_j / weighted_sum
        assert result[Exponential.PARAM_RATE] == expected_rate

    def test_X_equals_loc_case(self):
        X = np.array([1.0, 1.0, 1.0])
        H_j = np.array([0.8, 0.9, 1.0])

        result = q_function_strategy(self.component, X, H_j, self.mock_optimizer)

        assert Exponential.PARAM_LOC in result
        assert Exponential.PARAM_RATE in result

        assert result[Exponential.PARAM_LOC] == 1.0

        N_j = np.sum(H_j).item()
        weighted_sum = np.dot(H_j, np.maximum(X - 1.0, NUMERICAL_TOLERANCE)).item()
        expected_rate = N_j / weighted_sum
        assert abs(result[Exponential.PARAM_RATE] - expected_rate) < COMPARISON_CONSTANT


class TestQFunctionStrategyGeneric:
    def setup_method(self):
        self.mock_optimizer = Mock(spec=Optimizer)

    def test_generic_strategy_called_for_non_exponential(self):
        class MockDistribution:
            params_to_optimize: ClassVar[set[str]] = {"param1", "param2"}

            def get_params_vector(self, params):
                return np.array([1.0, 2.0])

            def set_params_from_vector(self, params, vector):
                pass

            def q_function(self, X, H_j):
                return 0.5

        mock_component = MockDistribution()

        self.mock_optimizer.minimize.return_value = np.array([1.5, 2.5])

        result = q_function_strategy(mock_component, np.array([1.0, 2.0]), np.array([0.5, 0.5]), self.mock_optimizer)

        np.testing.assert_allclose(sorted(list(result.values())), [1.5, 2.5])
        self.mock_optimizer.minimize.assert_called_once()


class TestQFunctionStrategyIntegration:
    def test_dispatcher_registration(self):
        registry = q_function_strategy.registry
        assert Exponential in registry
        assert registry[object] != registry[Exponential]

    def test_correct_dispatcher_called(self):
        exponential = Exponential(loc=1.0, rate=2.0)

        mock_optimizer = Mock(spec=Optimizer)
        X = np.array([1.5, 2.0, 2.5])
        H_j = np.array([0.8, 0.9, 1.0])

        result = q_function_strategy(exponential, X, H_j, mock_optimizer)

        mock_optimizer.minimize.assert_not_called()

        assert Exponential.PARAM_LOC in result
        assert Exponential.PARAM_RATE in result
