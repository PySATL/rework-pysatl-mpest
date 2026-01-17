"""Tests for high-level cluster matching algorithms."""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from unittest.mock import Mock, patch

import numpy as np
import pytest
from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.initializers import ClusterizeInitializer, MatchingMethod, ScoringMethod
from rework_pysatl_mpest.initializers.clusterize._cluster_match_algorithms import (
    _match_greedy,
    _match_hungarian,
    _match_permutations,
)


class TestClusterMatchAlgorithms:
    @pytest.fixture
    def strategy_context(self):
        """Prepares a context with mocks for strategies."""
        models = [Mock(spec=ContinuousDistribution), Mock(spec=ContinuousDistribution)]
        for m in models:
            m.__copy__ = Mock(return_value=m)

        array_size = 10
        n_clusters = 2
        cluster_weight = 5.0

        return {
            "models": models,
            "X": np.zeros(array_size),
            "H": np.zeros((array_size, n_clusters)),
            "estimation_strategies": [Mock(), Mock()],
            "optimizer": Mock(),
            "valid_clusters": [0, 1],
            "cluster_weights": [cluster_weight, cluster_weight],
            "score_func_component": Mock(),
            "score_func_mixture": Mock(),
        }

    def test_match_greedy(self, strategy_context):
        # Setup:
        # Model 0 prefers Cluster 0 (score 10) over Cluster 1 (score 100)
        # Model 1 prefers Cluster 0 (score 20) over Cluster 1 (score 30)
        # Greedy behavior:
        # 1. Model 0 takes Cluster 0 (best available).
        # 2. Model 1 checks Cluster 0 (used), takes Cluster 1.

        weight_half = 0.5

        with patch(
            "rework_pysatl_mpest.initializers.clusterize._cluster_match_algorithms._estimate_and_score_component"
        ) as mock_est:
            # Calls are made: M0-C0, M0-C1 -> M0 picks. Then M1-C0 (skip), M1-C1.
            # Actually greedy implementation calculates score for ALL unused clusters.

            # Mock return values for calls.
            # Sequence: M0 checks C0, M0 checks C1. Finds C0 best.
            #           M1 checks C1 (C0 is used). Finds C1 best.
            mock_est.side_effect = [
                {"params": {"m": 0, "c": 0}, "score": 10.0, "weight": weight_half},  # M0 - C0
                {"params": {"m": 0, "c": 1}, "score": 100.0, "weight": weight_half},  # M0 - C1
                {"params": {"m": 1, "c": 1}, "score": 30.0, "weight": weight_half},  # M1 - C1
            ]

            models, params, weights = _match_greedy(strategy_context)

            expected_len = 2
            assert len(params) == expected_len
            assert params[0] == {"m": 0, "c": 0}
            assert params[1] == {"m": 1, "c": 1}

    def test_match_hungarian(self, strategy_context):
        # Scenario where Greedy fails but Hungarian succeeds.
        # Cost Matrix:
        #      C0    C1
        # M0   10    15
        # M1   12    100
        #
        # Greedy: M0 sees 10 vs 15 -> takes C0. M1 forced to C1 (100). Total = 110.
        # Hungarian: M1 takes C0 (12), M0 takes C1 (15). Total = 27.

        weight_half = 0.5

        with patch(
            "rework_pysatl_mpest.initializers.clusterize._cluster_match_algorithms._precompute_fits"
        ) as mock_precompute:
            fit_m0_c0 = {"params": {"p": "m0c0"}, "score": 10.0, "weight": weight_half}
            fit_m0_c1 = {"params": {"p": "m0c1"}, "score": 15.0, "weight": weight_half}
            fit_m1_c0 = {"params": {"p": "m1c0"}, "score": 12.0, "weight": weight_half}
            fit_m1_c1 = {"params": {"p": "m1c1"}, "score": 100.0, "weight": weight_half}

            mock_precompute.return_value = [[fit_m0_c0, fit_m0_c1], [fit_m1_c0, fit_m1_c1]]

            models, params, weights = _match_hungarian(strategy_context)

            assert params[0] == {"p": "m0c1"}
            assert params[1] == {"p": "m1c0"}

    def test_match_permutations(self, strategy_context):
        # Brute force optimization of the mixture score.
        # 2 Models, 2 Clusters.
        # Permutation 1: M0->C0, M1->C1. Mixture Score: 500
        # Permutation 2: M0->C1, M1->C0. Mixture Score: 100

        weight_half = 0.5
        expected_calls = 2

        with patch(
            "rework_pysatl_mpest.initializers.clusterize._cluster_match_algorithms._precompute_fits"
        ) as mock_precompute:
            fit_dummy = {"params": {}, "score": 0, "weight": weight_half, "model": Mock()}
            mock_precompute.return_value = [[fit_dummy, fit_dummy], [fit_dummy, fit_dummy]]

            strategy_context["score_func_mixture"].side_effect = [500.0, 100.0]

            models, params, weights = _match_permutations(strategy_context)

            assert strategy_context["score_func_mixture"].call_count == expected_calls

    def test_match_clusters_for_models_entry_point(self):
        """
        Check that main func calls greedy.
        """
        X = np.array([1, 2, 3])
        H = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        models = [Mock(spec=ContinuousDistribution), Mock(spec=ContinuousDistribution)]
        est_strategies = [Mock(), Mock()]

        expected_weights = [0.6, 0.4]

        mock_greedy_func = Mock()
        mock_greedy_func.return_value = (models, [{"p": 1}, {"p": 2}], expected_weights)

        initializer = ClusterizeInitializer(is_accurate=True, is_soft=True, clusterizer=Mock())

        with patch.dict(
            ClusterizeInitializer._MATCHING_METHOD,
            {MatchingMethod.GREEDY: mock_greedy_func},
        ):
            res_models, res_params, res_weights = initializer._match_clusters_for_models(
                models,
                X,
                H,
                est_strategies,
                method=MatchingMethod.GREEDY,
                score_func=ScoringMethod.LIKELIHOOD,
            )

            mock_greedy_func.assert_called_once()
            assert res_params == [{"p": 1}, {"p": 2}]

    def test_match_clusters_for_models_fallback_default(self):
        array_size = 100
        weight_half = 0.5

        X = np.zeros(array_size)
        H = np.zeros((array_size, 2))
        H[:, 0] = 1.0

        models = [Mock(spec=ContinuousDistribution), Mock(spec=ContinuousDistribution)]
        est_strategies = [Mock(), Mock()]

        initializer = ClusterizeInitializer(is_accurate=True, is_soft=True, clusterizer=Mock())

        res_models, res_params, res_weights = initializer._match_clusters_for_models(
            models, X, H, est_strategies, method=MatchingMethod.GREEDY, score_func=ScoringMethod.LIKELIHOOD
        )

        assert res_params == [{}, {}]
        assert res_weights == [weight_half, weight_half]


class TestThreeClustersMatching:
    @pytest.fixture
    def context_3x3(self):
        """
        Creates a context with 3 models and 3 clusters.
        """
        models = [Mock(spec=ContinuousDistribution) for _ in range(3)]
        for i, m in enumerate(models):
            m.name = f"Model_{i}"
            m.__copy__ = Mock(return_value=m)

        array_size = 30
        cluster_weight = 10.0

        X = np.arange(array_size)
        H = np.zeros((array_size, 3))
        H[0:10, 0] = 1.0
        H[10:20, 1] = 1.0
        H[20:30, 2] = 1.0

        cluster_weights = [cluster_weight, cluster_weight, cluster_weight]
        valid_clusters = [0, 1, 2]

        return {
            "models": models,
            "X": X,
            "H": H,
            "estimation_strategies": [Mock(), Mock(), Mock()],
            "optimizer": Mock(),
            "valid_clusters": valid_clusters,
            "cluster_weights": cluster_weights,
            "score_func_component": Mock(),
            "score_func_mixture": Mock(),
        }

    @staticmethod
    def _create_mock_fits(cost_matrix):
        """
        Helper to create the structure returned by _precompute_fits based on a simple cost matrix.
        cost_matrix[i][j] = score for Model i and Cluster j.
        """
        weight = 0.33
        fits = []
        for row_idx, row_scores in enumerate(cost_matrix):
            model_fits = []
            for col_idx, score in enumerate(row_scores):
                model_fits.append(
                    {"params": {"m": row_idx, "c": col_idx}, "score": float(score), "weight": weight, "model": Mock()}
                )
            fits.append(model_fits)
        return fits

    def test_greedy_vs_hungarian_3x3_scenario(self, context_3x3):
        cost_matrix = [[10.0, 15.0, 100.0], [11.0, 50.0, 100.0], [100.0, 100.0, 10.0]]
        weight_default = 0.33

        with patch(
            "rework_pysatl_mpest.initializers.clusterize._cluster_match_algorithms._estimate_and_score_component"
        ) as mock_est:

            def side_effect(model, est_func, score_func, X, H_k, opt):
                cluster_idx = -1
                if H_k[0] == 1.0:
                    cluster_idx = 0
                elif H_k[10] == 1.0:
                    cluster_idx = 1
                elif H_k[20] == 1.0:
                    cluster_idx = 2
                model_idx = int(model.name.split("_")[1])
                score = cost_matrix[model_idx][cluster_idx]
                return {"params": {"m": model_idx, "c": cluster_idx}, "score": score, "weight": weight_default}

            mock_est.side_effect = side_effect
            _, greedy_params, _ = _match_greedy(context_3x3)
            assert greedy_params[0] == {"m": 0, "c": 0}
            assert greedy_params[1] == {"m": 1, "c": 1}
            assert greedy_params[2] == {"m": 2, "c": 2}

        with patch(
            "rework_pysatl_mpest.initializers.clusterize._cluster_match_algorithms._precompute_fits"
        ) as mock_precompute:
            mock_precompute.return_value = self._create_mock_fits(cost_matrix)
            _, hungarian_params, _ = _match_hungarian(context_3x3)
            assert hungarian_params[0] == {"m": 0, "c": 1}
            assert hungarian_params[1] == {"m": 1, "c": 0}
            assert hungarian_params[2] == {"m": 2, "c": 2}

    def test_permutations_3x3_full_search(self, context_3x3):
        mock_mixture_scores = [1000.0, 900.0, 800.0, 700.0, 50.0, 600.0]
        expected_calls = 6

        with patch(
            "rework_pysatl_mpest.initializers.clusterize._cluster_match_algorithms._precompute_fits"
        ) as mock_precompute:
            cost_matrix_dummy = np.zeros((3, 3))
            mock_precompute.return_value = self._create_mock_fits(cost_matrix_dummy)
            context_3x3["score_func_mixture"].side_effect = mock_mixture_scores
            _, best_params, _ = _match_permutations(context_3x3)

            assert context_3x3["score_func_mixture"].call_count == expected_calls
            ruff_const_wtf = 2
            assert best_params[0]["c"] == ruff_const_wtf
            assert best_params[1]["c"] == 0
            assert best_params[2]["c"] == 1

    def test_integration_3x3_with_main_function(self, context_3x3):
        """
        Check hungarian call from main func.
        """
        weight_default = 0.33
        mock_hungarian_func = Mock()
        dummy_params = [{"p": 1}, {"p": 2}, {"p": 3}]
        mock_hungarian_func.return_value = (context_3x3["models"], dummy_params, [weight_default] * 3)

        initializer = ClusterizeInitializer(is_accurate=True, is_soft=True, clusterizer=Mock())

        with patch.dict(
            ClusterizeInitializer._MATCHING_METHOD,
            {MatchingMethod.HUNGARIAN: mock_hungarian_func},
        ):
            initializer._match_clusters_for_models(
                models=context_3x3["models"],
                X=context_3x3["X"],
                H=context_3x3["H"],
                estimation_strategies=context_3x3["estimation_strategies"],
                method=MatchingMethod.HUNGARIAN,
                score_func=ScoringMethod.AIC,
            )

            mock_hungarian_func.assert_called_once()
