"""A module that provides tests for Cluster Match Strategy"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from unittest.mock import Mock, patch

import numpy as np
import pytest
from rework_pysatl_mpest.core import MixtureModel
from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.initializers.cluster_match_strategy import (
    _calculate_component_aic,
    _calculate_component_log_likelihood,
    _calculate_mixture_aic,
    _calculate_mixture_log_likelihood,
    _estimate_and_score_component,
    _match_greedy,
    _match_hungarian,
    _match_permutations,
    _precompute_fits,
    _validate_clusters_distributions,
    match_clusters_for_models,
)
from rework_pysatl_mpest.initializers.strategies import MatchingMethod, ScoringMethod
from rework_pysatl_mpest.optimizers import Optimizer


class TestClusterMatchStrategy:
    def setup_method(self):
        self.mock_optimizer = Mock(spec=Optimizer)
        self.mock_models = [Mock(spec=ContinuousDistribution) for _ in range(2)]
        for m in self.mock_models:
            m.params = {"loc", "scale"}
            m.lpdf = Mock(return_value=np.array([-1.0, -2.0]))
            m.__copy__ = Mock(return_value=m)
            m.set_params_from_vector = Mock()

    def test_validate_clusters_valid_input(self):
        H = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
        valid_clusters, weights = _validate_clusters_distributions(
            H, models_count=2, estimation_strategies_count=2, min_samples=0
        )
        assert valid_clusters == [0, 1]
        np.testing.assert_allclose(weights, [2.4, 0.6])

    def test_validate_clusters_sum_error(self):
        H = np.array([[0.8, 0.8], [0.1, 0.1]])
        with pytest.raises(ValueError, match="Sum of H matrix weights must be equal to 1"):
            _validate_clusters_distributions(H, 2, 2, 0)

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

    def test_calculate_component_log_likelihood(self):
        model = self.mock_models[0]
        model.lpdf.return_value = np.array([-1.0, -2.0, -3.0])
        X = np.array([1, 2, 3])
        H_k = np.array([1.0, 0.5, 0.0])

        # Expected: 1*(-1) + 0.5*(-2) + 0*(-3) = -1 - 1 = -2.0
        ll = _calculate_component_log_likelihood(model, X, H_k)
        assert ll == pytest.approx(-2.0)

    def test_calculate_component_aic(self):
        model = self.mock_models[0]
        model.params = {"a", "b"}
        model.lpdf.return_value = np.array([-1.0])
        X = np.array([1])
        H_k = np.array([1.0])

        # LL = -1.0. k=2.
        # AIC = 2*k - 2*LL = 4 - 2*(-1) = 6.0
        aic = _calculate_component_aic(model, X, H_k)
        assert aic == pytest.approx(6.0)

    def test_calculate_mixture_aic_and_likelihood(self):
        mock_mixture = Mock(spec=MixtureModel)
        mock_mixture.loglikelihood.return_value = np.array([-2.0, -3.0])

        c1 = Mock()
        c1.params = {"a": 1}
        c2 = Mock()
        c2.params = {"b": 2}

        mock_mixture.components = [c1, c2]
        mock_mixture.n_components = 2

        X = np.array([1, 2])

        ll = _calculate_mixture_log_likelihood(mock_mixture, X)
        ll_const = -5.0
        assert ll == ll_const

        # AIC = 2k - 2LL = 2(3) - 2(-5) = 6 + 10 = 16.0
        aic = _calculate_mixture_aic(mock_mixture, X)
        aic_const = 16.0
        assert aic == aic_const

    def test_estimate_and_score_component(self):
        model = self.mock_models[0]
        est_func = Mock(return_value={"loc": 5.0})
        score_func = Mock(return_value=123.45)
        X = np.zeros(10)
        H_k = np.ones(10)

        result = _estimate_and_score_component(model, est_func, score_func, X, H_k, self.mock_optimizer)

        est_func.assert_called_once()
        score_func.assert_called_once()
        assert result["params"] == {"loc": 5.0}
        score_const = 123.45
        assert result["score"] == score_const
        assert result["weight"] == 1.0
        result["model"].set_params_from_vector.assert_called()

    @pytest.fixture
    def strategy_context(self):
        """Prepares a context with mocks for strategies."""
        models = [Mock(spec=ContinuousDistribution), Mock(spec=ContinuousDistribution)]
        for m in models:
            m.__copy__ = Mock(return_value=m)

        X = np.zeros(10)
        H = np.zeros((10, 2))
        est_strategies = [Mock(), Mock()]
        valid_clusters = [0, 1]
        cluster_weights = [5.0, 5.0]

        return {
            "models": models,
            "X": X,
            "H": H,
            "estimation_strategies": est_strategies,
            "optimizer": Mock(),
            "valid_clusters": valid_clusters,
            "cluster_weights": cluster_weights,
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

        with patch("rework_pysatl_mpest.initializers.cluster_match_strategy._estimate_and_score_component") as mock_est:
            # Calls are made: M0-C0, M0-C1 -> M0 picks. Then M1-C0 (skip), M1-C1.
            # Actually greedy implementation calculates score for ALL unused clusters.

            # Mock return values for calls.
            # Sequence: M0 checks C0, M0 checks C1. Finds C0 best.
            #           M1 checks C1 (C0 is used). Finds C1 best.
            mock_est.side_effect = [
                {"params": {"m": 0, "c": 0}, "score": 10.0, "weight": 0.5},  # M0 - C0
                {"params": {"m": 0, "c": 1}, "score": 100.0, "weight": 0.5},  # M0 - C1
                {"params": {"m": 1, "c": 1}, "score": 30.0, "weight": 0.5},  # M1 - C1
            ]

            models, params, weights = _match_greedy(strategy_context)
            params_len = 2
            assert len(params) == params_len
            assert params[0] == {"m": 0, "c": 0}
            assert params[1] == {"m": 1, "c": 1}
            call_const = 3
            assert mock_est.call_count == call_const

    def test_match_hungarian(self, strategy_context):
        # Scenario where Greedy fails but Hungarian succeeds.
        # Cost Matrix:
        #      C0    C1
        # M0   10    15
        # M1   12    100
        #
        # Greedy: M0 sees 10 vs 15 -> takes C0. M1 forced to C1 (100). Total = 110.
        # Hungarian: M1 takes C0 (12), M0 takes C1 (15). Total = 27.

        with patch("rework_pysatl_mpest.initializers.cluster_match_strategy._precompute_fits") as mock_precompute:
            fit_m0_c0 = {"params": {"p": "m0c0"}, "score": 10.0, "weight": 0.5}
            fit_m0_c1 = {"params": {"p": "m0c1"}, "score": 15.0, "weight": 0.5}
            fit_m1_c0 = {"params": {"p": "m1c0"}, "score": 12.0, "weight": 0.5}
            fit_m1_c1 = {"params": {"p": "m1c1"}, "score": 100.0, "weight": 0.5}

            mock_precompute.return_value = [[fit_m0_c0, fit_m0_c1], [fit_m1_c0, fit_m1_c1]]

            models, params, weights = _match_hungarian(strategy_context)

            assert params[0] == {"p": "m0c1"}
            assert params[1] == {"p": "m1c0"}

    def test_match_permutations(self, strategy_context):
        # Brute force optimization of the mixture score.
        # 2 Models, 2 Clusters.
        # Permutation 1: M0->C0, M1->C1. Mixture Score: 500
        # Permutation 2: M0->C1, M1->C0. Mixture Score: 100

        with patch("rework_pysatl_mpest.initializers.cluster_match_strategy._precompute_fits") as mock_precompute:
            fit_dummy = {"params": {}, "score": 0, "weight": 0.5, "model": Mock()}
            mock_precompute.return_value = [[fit_dummy, fit_dummy], [fit_dummy, fit_dummy]]

            strategy_context["score_func_mixture"].side_effect = [500.0, 100.0]

            models, params, weights = _match_permutations(strategy_context)

            call_const = 2
            assert strategy_context["score_func_mixture"].call_count == call_const

    def test_match_clusters_for_models_entry_point(self):
        """
        Check that main func calls greedy.
        """
        X = np.array([1, 2, 3])
        H = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        models = [Mock(spec=ContinuousDistribution), Mock(spec=ContinuousDistribution)]
        est_strategies = [Mock(), Mock()]

        mock_greedy_func = Mock()
        mock_greedy_func.return_value = (models, [{"p": 1}, {"p": 2}], [0.6, 0.4])

        with patch.dict(
            "rework_pysatl_mpest.initializers.cluster_match_strategy._MATCHING_METHOD",
            {MatchingMethod.GREEDY: mock_greedy_func},
        ):
            res_models, res_params, res_weights = match_clusters_for_models(
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
        X = np.zeros(100)
        H = np.zeros((100, 2))
        H[:, 0] = 1.0

        models = [Mock(spec=ContinuousDistribution), Mock(spec=ContinuousDistribution)]
        est_strategies = [Mock(), Mock()]

        res_models, res_params, res_weights = match_clusters_for_models(
            models, X, H, est_strategies, method=MatchingMethod.GREEDY, score_func=ScoringMethod.LIKELIHOOD
        )

        assert res_params == [{}, {}]
        assert res_weights == [0.5, 0.5]

    def test_precompute_fits_caching(self, strategy_context):
        with patch("rework_pysatl_mpest.initializers.cluster_match_strategy._estimate_and_score_component") as mock_est:
            mock_est.return_value = {"params": {}, "score": 0, "weight": 0, "model": Mock()}
            fits_len_const = 2
            fits = _precompute_fits(strategy_context)
            assert len(fits) == fits_len_const
            assert len(fits[0]) == fits_len_const
            call_const = 4
            assert mock_est.call_count == call_const


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

        X = np.arange(30)
        H = np.zeros((30, 3))
        H[0:10, 0] = 1.0
        H[10:20, 1] = 1.0
        H[20:30, 2] = 1.0

        cluster_weights = [10.0, 10.0, 10.0]
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
        fits = []
        for row_idx, row_scores in enumerate(cost_matrix):
            model_fits = []
            for col_idx, score in enumerate(row_scores):
                model_fits.append(
                    {"params": {"m": row_idx, "c": col_idx}, "score": float(score), "weight": 0.33, "model": Mock()}
                )
            fits.append(model_fits)
        return fits

    def test_greedy_vs_hungarian_3x3_scenario(self, context_3x3):
        cost_matrix = [[10.0, 15.0, 100.0], [11.0, 50.0, 100.0], [100.0, 100.0, 10.0]]

        with patch("rework_pysatl_mpest.initializers.cluster_match_strategy._estimate_and_score_component") as mock_est:

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
                return {"params": {"m": model_idx, "c": cluster_idx}, "score": score, "weight": 0.33}

            mock_est.side_effect = side_effect
            _, greedy_params, _ = _match_greedy(context_3x3)
            assert greedy_params[0] == {"m": 0, "c": 0}
            assert greedy_params[1] == {"m": 1, "c": 1}
            assert greedy_params[2] == {"m": 2, "c": 2}

        with patch("rework_pysatl_mpest.initializers.cluster_match_strategy._precompute_fits") as mock_precompute:
            mock_precompute.return_value = self._create_mock_fits(cost_matrix)
            _, hungarian_params, _ = _match_hungarian(context_3x3)
            assert hungarian_params[0] == {"m": 0, "c": 1}
            assert hungarian_params[1] == {"m": 1, "c": 0}
            assert hungarian_params[2] == {"m": 2, "c": 2}

    def test_permutations_3x3_full_search(self, context_3x3):
        mock_mixture_scores = [1000.0, 900.0, 800.0, 700.0, 50.0, 600.0]
        with patch("rework_pysatl_mpest.initializers.cluster_match_strategy._precompute_fits") as mock_precompute:
            cost_matrix_dummy = np.zeros((3, 3))
            mock_precompute.return_value = self._create_mock_fits(cost_matrix_dummy)
            context_3x3["score_func_mixture"].side_effect = mock_mixture_scores
            _, best_params, _ = _match_permutations(context_3x3)
            call_const = 6
            assert context_3x3["score_func_mixture"].call_count == call_const
            param1_const = 2
            param2_const = 0
            param3_const = 1
            assert best_params[0]["c"] == param1_const
            assert best_params[1]["c"] == param2_const
            assert best_params[2]["c"] == param3_const

    def test_integration_3x3_with_main_function(self, context_3x3):
        """
        Check hungarian call from main func.
        """
        mock_hungarian_func = Mock()
        dummy_params = [{"p": 1}, {"p": 2}, {"p": 3}]
        mock_hungarian_func.return_value = (context_3x3["models"], dummy_params, [0.33] * 3)

        with patch.dict(
            "rework_pysatl_mpest.initializers.cluster_match_strategy._MATCHING_METHOD",
            {MatchingMethod.HUNGARIAN: mock_hungarian_func},
        ):
            match_clusters_for_models(
                models=context_3x3["models"],
                X=context_3x3["X"],
                H=context_3x3["H"],
                estimation_strategies=context_3x3["estimation_strategies"],
                method=MatchingMethod.HUNGARIAN,
                score_func=ScoringMethod.AIC,
            )

            mock_hungarian_func.assert_called_once()

    def test_validate_clusters_3x3_one_too_small(self):
        H = np.zeros((30, 3))
        H[0:15, 0] = 1.0
        H[15:30, 1] = 1.0
        H[29, 2] = 0.0
        valid_clusters, _ = _validate_clusters_distributions(
            H, models_count=3, estimation_strategies_count=3, min_samples=5
        )
        assert valid_clusters == []
