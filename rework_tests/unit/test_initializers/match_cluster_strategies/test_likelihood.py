"""A module that provides tests for cluster match strategy with LikelyHood method"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from unittest.mock import Mock

import numpy as np
import pytest
from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.initializers.cluster_match_strategy import match_clusters_for_models_log_likelihood

COMPARISON_CONSTANT = 1e-10


class TestMatchClustersForModelsLogLikelihood:
    @pytest.fixture
    def mock_models(self, n=2):
        models = [Mock(spec=ContinuousDistribution) for _ in range(n)]
        for model in models:
            model.params = {"param1", "param2"}
            model.set_params_from_vector = Mock()
            model.lpdf = Mock(return_value=np.array([-1.0, -2.0, -3.0]))
        return models

    @pytest.fixture
    def estimation_info(self, n=2):
        funcs = [Mock() for _ in range(n)]
        for f in funcs:
            f.return_value = {"param1": 1.5, "param2": 2.5}
        return funcs

    @pytest.fixture
    def X(self):
        return np.array([1.0, 2.0, 3.0])

    @pytest.fixture
    def H_valid(self):
        H_raw = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
        return H_raw / H_raw.sum(axis=1, keepdims=True)

    def test_H_sum_not_1_raises(self, mock_models, estimation_info, X):
        H_invalid = np.array([[0.8, 0.1], [0.7, 0.2], [0.9, 0.05]])
        with pytest.raises(ValueError, match="Sum of H matrix weights must be equal to 1"):
            match_clusters_for_models_log_likelihood(mock_models, X, H_invalid, estimation_info, min_samples=1)

    def test_estimation_info_length_mismatch(self, mock_models, estimation_info, X, H_valid):
        with pytest.raises(ValueError, match="Number of estimation functions must match number of models"):
            match_clusters_for_models_log_likelihood(mock_models, X, H_valid, estimation_info[:1], min_samples=1)

    def test_insufficient_valid_clusters(self, mock_models, estimation_info, X):
        H_low = np.array([[0.999, 0.001], [0.998, 0.002], [0.997, 0.003]])
        H_norm = H_low / H_low.sum(axis=1, keepdims=True)
        models, params, weights = match_clusters_for_models_log_likelihood(
            mock_models, X, H_norm, estimation_info, min_samples=10
        )
        assert all(p == {} for p in params)
        assert all(abs(w - 0.5) < COMPARISON_CONSTANT for w in weights)

    def test_no_valid_clusters(self, mock_models, estimation_info, X):
        H_low = np.array([[0.999, 0.001], [0.998, 0.002], [0.997, 0.003]])
        H_norm = H_low / H_low.sum(axis=1, keepdims=True)
        models, params, weights = match_clusters_for_models_log_likelihood(
            mock_models, X, H_norm, estimation_info, min_samples=100
        )
        assert all(p == {} for p in params)
        assert all(abs(w - 0.5) < COMPARISON_CONSTANT for w in weights)

    def test_more_clusters_than_models(self, mock_models, estimation_info, X):
        models = [Mock(spec=ContinuousDistribution) for _ in range(2)]
        len_models = len(models)
        for model in models:
            model.params = {"param"}
            model.set_params_from_vector = Mock()
            model.lpdf.return_value = np.array([-0.5, -1.0, -1.5])

        X = np.array([1.0, 2.0, 3.0])
        H = np.array([[0.4, 0.4, 0.2], [0.4, 0.4, 0.2], [0.4, 0.4, 0.2]])
        H_norm = H / H.sum(axis=1, keepdims=True)

        est_funcs = [Mock(return_value={"param": 1.0}), Mock(return_value={"param": 2.0})]

        models_out, params, weights = match_clusters_for_models_log_likelihood(
            models, X, H_norm, est_funcs, min_samples=1
        )
        norm_weights = np.asarray(weights) / sum(weights)
        assert len(params) == len_models
        assert all(p != {} for p in params)
        assert abs(sum(norm_weights) - 1.0) < COMPARISON_CONSTANT

    def test_basic_assignment(self, mock_models, estimation_info, X, H_valid):
        len_models = len(mock_models)
        mock_models[0].lpdf.return_value = np.array([-0.1, -0.2, -0.3])
        mock_models[1].lpdf.return_value = np.array([-2.0, -3.0, -4.0])
        models, params, weights = match_clusters_for_models_log_likelihood(
            mock_models, X, H_valid, estimation_info, min_samples=1
        )
        assert len(params) == len_models
        assert len(weights) == len_models
        assert abs(sum(weights) - 1.0) < COMPARISON_CONSTANT

    def test_main_logic(self):
        models = [Mock(spec=ContinuousDistribution) for _ in range(2)]
        len_models = len(models)
        for m in models:
            m.params = {"a"}
            m.set_params_from_vector = Mock()
            m.lpdf.return_value = np.array([-0.1, -0.2, -0.3])

        X = np.array([1.0, 2.0, 3.0])
        H = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        est_funcs = [Mock(return_value={"a": 1.0}), Mock(return_value={"a": 2.0})]

        models_out, params, weights = match_clusters_for_models_log_likelihood(models, X, H, est_funcs, min_samples=1)

        assert len(params) == len_models
        assert all(isinstance(p, dict) and "a" in p for p in params)
        assert abs(sum(weights) - 1.0) < COMPARISON_CONSTANT

    def test_single_model_single_cluster(self):
        model = Mock(spec=ContinuousDistribution)
        model.params = {"p"}
        model.set_params_from_vector = Mock()
        model.lpdf.return_value = np.array([-1.0, -2.0])

        X = np.array([1.0, 2.0])
        H = np.array([[1.0], [1.0]])
        H_norm = H / H.sum(axis=1, keepdims=True)

        estimation = [Mock(return_value={"p": 5.0})]

        models, params, weights = match_clusters_for_models_log_likelihood(
            [model], X, H_norm, estimation, min_samples=1
        )
        assert len(params) == 1
        assert weights == [1.0]

    def test_extreme_log_probs_handled(self, mock_models, estimation_info, X):
        mock_models[0].lpdf.return_value = np.array([-1e20, -1e19, -1e18])
        mock_models[1].lpdf.return_value = np.array([-1e-20, -1e-19, -1e-18])
        len_models = len(mock_models)
        H = np.array([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]])
        H_norm = H / H.sum(axis=1, keepdims=True)
        models, params, weights = match_clusters_for_models_log_likelihood(
            mock_models, X, H_norm, estimation_info, min_samples=1
        )
        assert len(params) == len_models
        assert len(weights) == len_models
        assert abs(sum(weights) - 1.0) < COMPARISON_CONSTANT

    def test_zero_weight_cluster_handling(self):
        models = [Mock(spec=ContinuousDistribution) for _ in range(2)]
        for model in models:
            model.params = {"param"}
            model.set_params_from_vector = Mock()
            model.lpdf.return_value = np.array([-1.0, -2.0, -3.0])

        X = np.array([1.0, 2.0, 3.0])
        H = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        H_norm = H / H.sum(axis=1, keepdims=True)

        est_funcs = [Mock(return_value={"param": 1.0}), Mock(return_value={"param": 2.0})]

        models, params, weights = match_clusters_for_models_log_likelihood(models, X, H_norm, est_funcs, min_samples=1)
        assert all(p == {} for p in params)
        assert all(abs(w - 0.5) < COMPARISON_CONSTANT for w in weights)

    def test_equal_scores_different_clusters(self):
        models = [Mock(spec=ContinuousDistribution) for _ in range(2)]
        len_params = len(models)
        for model in models:
            model.params = {"param"}
            model.set_params_from_vector = Mock()
            model.lpdf.return_value = np.array([-0.5, -1.0, -1.5])

        X = np.array([1.0, 2.0, 3.0])
        H = np.array([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]])
        H_norm = H / H.sum(axis=1, keepdims=True)

        est_funcs = [Mock(return_value={"param": 1.0}), Mock(return_value={"param": 2.0})]

        models, params, weights = match_clusters_for_models_log_likelihood(models, X, H_norm, est_funcs, min_samples=1)
        assert len(params) == len_params
        assert all(p != {} for p in params)
        assert abs(sum(weights) - 1.0) < COMPARISON_CONSTANT
