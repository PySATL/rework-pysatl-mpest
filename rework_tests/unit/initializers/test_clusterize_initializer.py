# """A module that provides tests for ClusterizeInitializer"""
#
# __author__ = "Viktor Khanukaev"
# __copyright__ = "Copyright (c) 2025 PySATL project"
# __license__ = "SPDX-License-Identifier: MIT"
#
# from types import MappingProxyType
# from unittest.mock import Mock, patch
#
# import numpy as np
# import pytest
# from rework_pysatl_mpest.core.mixture import MixtureModel
# from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
# from rework_pysatl_mpest.initializers.clusterize_initializer import ClusterizeInitializer
# from rework_pysatl_mpest.initializers.strategies import EstimationStrategy
# from rework_pysatl_mpest.optimizers import ScipyNelderMead
#
#
# class TestClusterizeInitializer:
#     def setup_method(self):
#         self.mock_clusterizer = Mock()
#         self.mock_distributions = [Mock(spec=ContinuousDistribution) for _ in range(5)]
#
#         for dist in self.mock_distributions:
#             dist.params = {"mean", "std"}
#             dist.params_to_optimize = {"mean", "std"}
#             dist.set_params_from_vector = Mock()
#             dist.lpdf = Mock(return_value=np.array([-0.5, -1.0, -1.5]))
#             dist.get_params_vector = Mock(return_value=np.array([0.0, 1.0]))
#             dist.q_function = Mock(return_value=-10.0)
#
#     @pytest.mark.parametrize("is_accurate,is_soft", [(True, True), (True, False), (False, True), (False, False)])
#     def test_initialization_parameters(self, is_accurate, is_soft):
#         initializer = ClusterizeInitializer(is_accurate=is_accurate, is_soft=is_soft,
#                                             clusterizer=self.mock_clusterizer)
#
#         assert initializer.is_accurate == is_accurate
#         assert initializer.is_soft == is_soft
#         assert initializer.clusterizer == self.mock_clusterizer
#         assert initializer.n_components is None
#         assert initializer.cluster_match_strategy == ClusterMatchStrategy.LIKELIHOOD
#         assert initializer.estimation_strategies == []
#         assert initializer.models == []
#
#     def test_soft_clusterize(self):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=True, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0])
#         expected_weights = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
#
#         self.mock_clusterizer.fit_transform = Mock(return_value=expected_weights)
#         result = initializer._clusterize(X, self.mock_clusterizer)
#
#         np.testing.assert_array_equal(result, expected_weights)
#
#         assert self.mock_clusterizer.fit_transform.call_count == 1
#         called_arg = self.mock_clusterizer.fit_transform.call_args[0][0]
#         np.testing.assert_array_equal(called_arg, X)
#
#     def test_hard_clusterize_no_outliers(self):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=False, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0, 4.0])
#         labels = np.array([0, 1, 0, 1])
#
#         self.mock_clusterizer.fit_predict = Mock(return_value=labels)
#         result = initializer._clusterize(X, self.mock_clusterizer)
#
#         expected = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
#         np.testing.assert_array_equal(result, expected)
#
#     def test_hard_clusterize_single_cluster_no_outliers(self):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=False, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0])
#         labels = np.array([0, 0, 0])
#
#         self.mock_clusterizer.fit_predict = Mock(return_value=labels)
#         result = initializer._clusterize(X, self.mock_clusterizer)
#
#         expected = np.array([[1.0], [1.0], [1.0]])
#         np.testing.assert_array_equal(result, expected)
#
#     def test_hard_clusterize_with_outliers(self):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=False, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0])
#         labels = np.array([0, -1, 1])
#
#         self.mock_clusterizer.fit_predict = Mock(return_value=labels)
#         result = initializer._clusterize(X, self.mock_clusterizer)
#
#         expected = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
#         np.testing.assert_array_equal(result, expected)
#
#     def test_hard_clusterize_all_outliers(self):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=False, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0])
#         labels = np.array([-1, -1])
#
#         self.mock_clusterizer.fit_predict = Mock(return_value=labels)
#         result = initializer._clusterize(X, self.mock_clusterizer)
#
#         expected = np.array([[1.0], [1.0]])
#         np.testing.assert_array_equal(result, expected)
#
#     def test_hard_clusterize_non_consecutive_labels(self):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=False, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0, 4.0])
#         labels = np.array([2, 5, 2, 5])
#
#         self.mock_clusterizer.fit_predict = Mock(return_value=labels)
#         result = initializer._clusterize(X, self.mock_clusterizer)
#
#         expected = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
#         np.testing.assert_array_equal(result, expected)
#
#     @pytest.mark.parametrize("is_soft", [True, False])
#     def test_clusterize_failure(self, is_soft):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=is_soft, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0])
#
#         if is_soft:
#             self.mock_clusterizer.fit_transform = Mock(side_effect=Exception("Failed"))
#             expected_error = "Fuzzy clusterizer failed"
#         else:
#             self.mock_clusterizer.fit_predict = Mock(side_effect=Exception("Failed"))
#             expected_error = "Hard clusterizer failed"
#
#         with pytest.raises(ValueError, match=expected_error):
#             initializer._clusterize(X, self.mock_clusterizer)
#
#     def test_clusterizer_missing_methods(self):
#         class InvalidClusterizer:
#             pass
#
#         invalid_clusterizer = InvalidClusterizer()
#         X = np.array([1.0, 2.0, 3.0])
#
#         initializer_soft = ClusterizeInitializer(True, True, invalid_clusterizer)
#         with pytest.raises(ValueError, match="Clusterizer doesn't have required method"):
#             initializer_soft._clusterize(X, invalid_clusterizer)
#
#         initializer_hard = ClusterizeInitializer(True, False, invalid_clusterizer)
#         with pytest.raises(ValueError, match="Clusterizer doesn't have required method"):
#             initializer_hard._clusterize(X, invalid_clusterizer)
#
#     def test_perform_accurate_init_normal_path(self):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=True, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0])
#         H = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
#         dists = [self.mock_distributions[0], self.mock_distributions[1]]
#
#         with (
#             patch.object(initializer, "_clusterize", return_value=H),
#             patch(
#                 "rework_pysatl_mpest.initializers.cluster_match_strategy.match_clusters_for_models_akaike"
#             ) as mock_match,
#         ):
#             mock_match.return_value = (
#                 [dists[0], dists[1]],
#                 [{"mean": 1.0, "std": 0.5}, {"mean": 2.0, "std": 1.0}],
#                 [0.4, 0.6],
#             )
#
#             result = initializer.perform(
#                 X=X,
#                 dists=dists,
#                 cluster_match_strategy=ClusterMatchStrategy.AKAIKE,
#                 estimation_strategies=[EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION],
#             )
#             len_components = len(dists)
#
#             assert isinstance(result, MixtureModel)
#             assert len(result.components) == len_components
#             assert len(result.weights) == len_components
#             assert sum(result.weights) == pytest.approx(1.0)
#             np.testing.assert_array_equal(result.weights, [0.8, 0.2])
#
#     def test_perform_accurate_init_fallback_to_fast_init(self):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=True, clusterizer=self.mock_clusterizer)
#
#         X = np.array([[1.0], [2.0], [3.0]])
#         H = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
#         dists = [self.mock_distributions[0], self.mock_distributions[1]]
#
#         with (
#             patch.object(initializer, "_clusterize", return_value=H),
#             patch(
#                 "rework_pysatl_mpest.initializers.clusterize_initializer.match_clusters_for_models_log_likelihood"
#             ) as mock_match,
#         ):
#             mock_match.return_value = ([dists[0], dists[1]], [None, {"mean": 2.0, "std": 1.0}], [0.5, 0.5])
#
#             with patch.object(initializer, "_fast_init") as mock_fast_init:
#                 mock_fast_init.return_value = ([dists[0], dists[1]], [0.3, 0.7])
#                 optimizer = ScipyNelderMead()
#                 result = initializer.perform(
#                     X=X,
#                     dists=dists,
#                     cluster_match_strategy=ClusterMatchStrategy.LIKELIHOOD,
#                     estimation_strategies=[EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION],
#                     optimizer=optimizer,
#                 )
#
#                 mock_fast_init.assert_called_once_with(X, H, optimizer)
#                 np.testing.assert_array_equal(result.weights, [0.3, 0.7])
#
#     def test_perform_fast_init(self):
#         initializer = ClusterizeInitializer(is_accurate=False, is_soft=True, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0])
#         H = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
#         dists = [self.mock_distributions[0], self.mock_distributions[1]]
#
#         with (
#             patch.object(initializer, "_clusterize", return_value=H),
#             patch.object(initializer, "_estimation_strategies") as mock_est_strategies,
#         ):
#             mock_est_strategies.__getitem__.return_value = Mock(return_value={"mean": 0.0, "std": 1.0})
#
#             result = initializer.perform(
#                 X=X,
#                 dists=dists,
#                 cluster_match_strategy=ClusterMatchStrategy.LIKELIHOOD,
#                 estimation_strategies=[EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION],
#             )
#             len_components = len(dists)
#             assert isinstance(result, MixtureModel)
#             assert len(result.components) == len_components
#             assert len(result.weights) == len_components
#             assert sum(result.weights) == pytest.approx(1.0)
#
#             dists[0].set_params_from_vector.assert_called_once()
#             dists[1].set_params_from_vector.assert_called_once()
#
#     def test_weight_normalization(self):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=True, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0])
#         H = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
#
#         with (
#             patch.object(initializer, "_clusterize", return_value=H),
#             patch.object(initializer, "_accurate_init") as mock_acc_init,
#         ):
#             mock_acc_init.return_value = ([self.mock_distributions[0], self.mock_distributions[1]], [2.0, 3.0])
#
#             result = initializer.perform(
#                 X=X,
#                 dists=[self.mock_distributions[0], self.mock_distributions[1]],
#                 cluster_match_strategy=ClusterMatchStrategy.AKAIKE,
#                 estimation_strategies=[EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION],
#             )
#
#             expected_weights = [0.4, 0.6]
#             for i, expected in enumerate(expected_weights):
#                 assert result.weights[i] == pytest.approx(expected, abs=0.01)
#             assert sum(result.weights) == pytest.approx(1.0)
#
#     def test_validation_errors_accurate_init(self):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=True, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0])
#         H = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
#
#         initializer.n_components = None
#         initializer.models = [self.mock_distributions[0], self.mock_distributions[1]]
#         initializer.estimation_strategies = [EstimationStrategy.QFUNCTION] * 2
#
#         with pytest.raises(ValueError, match="n_components must be set before calling _accurate_init"):
#             initializer._accurate_init(X, H)
#
#         initializer.n_components = 2
#         initializer.models = [self.mock_distributions[0], self.mock_distributions[1]]
#         initializer.estimation_strategies = [EstimationStrategy.QFUNCTION]
#
#         with pytest.raises(ValueError, match="Count of models must match count of estimation strategies"):
#             initializer._accurate_init(X, H)
#
#     def test_validation_errors_fast_init(self):
#         initializer = ClusterizeInitializer(is_accurate=False, is_soft=True, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0])
#         H = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
#
#         initializer.n_components = None
#         initializer.models = [self.mock_distributions[0], self.mock_distributions[1]]
#         initializer.estimation_strategies = [EstimationStrategy.QFUNCTION] * 2
#
#         with pytest.raises(ValueError, match="n_components must be set before calling _fast_init"):
#             initializer._fast_init(X, H)
#
#     def test_different_distribution_types(self):
#         initializer = ClusterizeInitializer(is_accurate=False, is_soft=True, clusterizer=self.mock_clusterizer)
#
#         X = np.array([1.0, 2.0, 3.0, 4.0])
#         H = np.array([[0.8, 0.2, 0.1, 0.3], [0.7, 0.3, 0.2, 0.1], [0.9, 0.1, 0.3, 0.2], [0.6, 0.4, 0.5, 0.1]])
#
#         distributions = []
#         param_sets = [{"mean", "std"}, {"alpha", "beta"}, {"lambda"}, {"shape", "scale"}]
#
#         for i, params in enumerate(param_sets):
#             mock_dist = Mock(spec=ContinuousDistribution)
#             mock_dist.params = params
#             mock_dist.set_params_from_vector = Mock()
#             distributions.append(mock_dist)
#
#         self.mock_clusterizer.fit_transform = Mock(return_value=H)
#
#         with patch.object(initializer, "_estimation_strategies") as mock_est_strategies:
#
#             def mock_estimation_side_effect(model, X, H, optimizer):
#                 param_map = {
#                     frozenset({"mean", "std"}): {"mean": 2.0, "std": 1.0},
#                     frozenset({"alpha", "beta"}): {"alpha": 2.0, "beta": 2.0},
#                     frozenset({"lambda"}): {"lambda": 0.5},
#                     frozenset({"shape", "scale"}): {"shape": 2.0, "scale": 1.0},
#                 }
#                 return param_map.get(frozenset(model.params), {})
#
#             mock_est_strategies.__getitem__.return_value = Mock(side_effect=mock_estimation_side_effect)
#
#             result = initializer.perform(
#                 X=X,
#                 dists=distributions,
#                 cluster_match_strategy=ClusterMatchStrategy.LIKELIHOOD,
#                 estimation_strategies=[EstimationStrategy.QFUNCTION] * len(distributions),
#             )
#
#             assert isinstance(result, MixtureModel)
#             assert len(result.components) == len(distributions)
#             for dist in distributions:
#                 dist.set_params_from_vector.assert_called_once()
#
#     def test_edge_case_empty_data(self):
#         initializer = ClusterizeInitializer(is_accurate=True, is_soft=True, clusterizer=self.mock_clusterizer)
#
#         X = np.array([])
#         H = np.array([]).reshape(0, 2)
#         distributions = [Mock(spec=ContinuousDistribution) for _ in range(2)]
#
#         for dist in distributions:
#             dist.set_params_from_vector = Mock()
#
#         self.mock_clusterizer.fit_transform = Mock(return_value=H)
#
#         with patch.object(initializer, "_accurate_init") as mock_acc_init:
#             mock_acc_init.return_value = (distributions, [0.5, 0.5])
#
#             result = initializer.perform(
#                 X=X,
#                 dists=distributions,
#                 cluster_match_strategy=ClusterMatchStrategy.LIKELIHOOD,
#                 estimation_strategies=[EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION],
#             )
#             len_components = len(distributions)
#             assert isinstance(result, MixtureModel)
#             assert len(result.components) == len_components
#             np.testing.assert_array_equal(result.weights, [0.5, 0.5])
#
#     def test_class_variables_immutable(self):
#         assert isinstance(ClusterizeInitializer._estimation_strategies, MappingProxyType)
#         assert isinstance(ClusterizeInitializer._cluster_match_strategies, MappingProxyType)
#         assert EstimationStrategy.QFUNCTION in ClusterizeInitializer._estimation_strategies
#         assert ClusterMatchStrategy.LIKELIHOOD in ClusterizeInitializer._cluster_match_strategies
#         assert ClusterMatchStrategy.AKAIKE in ClusterizeInitializer._cluster_match_strategies
