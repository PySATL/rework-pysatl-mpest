"""Tests for utility functions used in cluster matching."""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from unittest.mock import Mock, patch

import numpy as np
import pytest
from pysatl_mpest.distributions import ContinuousDistribution
from pysatl_mpest.initializers.clusterize.utils import (
    _estimate_and_score_component,
    _precompute_fits,
    _validate_clusters_distributions,
)
from pysatl_mpest.optimizers import Optimizer


class TestUtils:
    def setup_method(self):
        self.mock_optimizer = Mock(spec=Optimizer)
        self.mock_model = Mock(spec=ContinuousDistribution)
        self.mock_model.__copy__ = Mock(return_value=self.mock_model)
        self.mock_model.set_params_from_vector = Mock()

    def test_validate_clusters_valid_input(self):
        weight_main = 0.8
        weight_minor = 0.2
        expected_sum_valid = 2.4
        expected_sum_minor = 0.6

        H = np.array([[weight_main, weight_minor], [weight_main, weight_minor], [weight_main, weight_minor]])
        valid_clusters, weights = _validate_clusters_distributions(
            H, models_count=2, estimation_strategies_count=2, min_samples=0
        )
        assert valid_clusters == [0, 1]
        np.testing.assert_allclose(weights, [expected_sum_valid, expected_sum_minor])

    def test_validate_clusters_sum_error(self):
        H = np.array([[0.8, 0.8], [0.1, 0.1]])
        with pytest.raises(ValueError, match="Sum of H matrix weights must be equal to 1"):
            _validate_clusters_distributions(H, models_count=2, estimation_strategies_count=2, min_samples=0)

    def test_validate_clusters_strategy_count_mismatch(self):
        H = np.array([[1.0], [1.0]])
        with pytest.raises(ValueError, match="Number of estimation functions must match number of models"):
            _validate_clusters_distributions(H, models_count=1, estimation_strategies_count=2, min_samples=0)

    def test_validate_clusters_insufficient_samples(self):
        H = np.array([[1.0, 0.0], [1.0, 0.0]])
        valid_clusters, weights = _validate_clusters_distributions(
            H, models_count=2, estimation_strategies_count=2, min_samples=1
        )
        assert valid_clusters == []
        assert weights == []

    def test_estimate_and_score_component(self):
        loc_val = 5.0
        score_val = 123.45
        array_size = 10
        expected_weight = 1.0

        est_func = Mock(return_value={"loc": loc_val})
        score_func = Mock(return_value=score_val)
        X = np.zeros(array_size)
        H_k = np.ones(array_size)

        result = _estimate_and_score_component(self.mock_model, est_func, score_func, X, H_k, self.mock_optimizer)

        est_func.assert_called_once()
        score_func.assert_called_once()
        assert result["params"] == {"loc": loc_val}
        assert result["score"] == score_val
        assert result["weight"] == expected_weight
        self.mock_model.set_params_from_vector.assert_called()

    def test_precompute_fits(self):
        array_size = 10
        expected_calls = 2

        context = {
            "models": [self.mock_model],
            "estimation_strategies": [Mock()],
            "valid_clusters": [0, 1],
            "score_func_component": Mock(),
            "X": np.zeros(array_size),
            "H": np.zeros((array_size, 2)),
            "optimizer": self.mock_optimizer,
        }

        with patch("pysatl_mpest.initializers.clusterize.utils._estimate_and_score_component") as mock_est:
            mock_est.return_value = {"params": {}, "score": 0, "weight": 0, "model": Mock()}

            fits = _precompute_fits(context)

            assert len(fits) == 1
            len_fits0 = 2
            assert len(fits[0]) == len_fits0
            assert mock_est.call_count == expected_calls
