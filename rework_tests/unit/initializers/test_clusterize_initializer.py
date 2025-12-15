"""
Detailed unit tests for ClusterizeInitializer.
Covers edge cases, parameter unpacking, outlier handling, and integration flows.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.initializers import Initializer
from rework_pysatl_mpest.initializers.clusterize_initializer import ClusterizeInitializer
from rework_pysatl_mpest.initializers.strategies import EstimationStrategy, MatchingMethod, ScoringMethod
from rework_pysatl_mpest.optimizers import Optimizer


class TestClusterizeInitializer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_clusterizer = Mock()
        self.mock_optimizer = Mock(spec=Optimizer)

        self.dist1 = Mock(spec=ContinuousDistribution)
        self.dist1.params_to_optimize = ["loc", "scale"]
        self.dist1.get_params_vector.return_value = [0.0, 1.0]

        self.dist2 = Mock(spec=ContinuousDistribution)
        self.dist2.params_to_optimize = ["loc", "scale"]
        self.dist2.get_params_vector.return_value = [5.0, 2.0]

        self.dists = [self.dist1, self.dist2]

    # 1. Tests for _clusterize (Hard & Soft Logic)

    def test_clusterize_soft_input_reshaping(self):
        """Test that fit_transform is called with correctly reshaped 2D array."""
        initializer = ClusterizeInitializer(is_accurate=True, is_soft=True)
        X_1d = np.array([1.0, 2.0, 3.0])

        self.mock_clusterizer.fit_transform.return_value = np.zeros((3, 2))

        initializer._clusterize(X_1d, self.mock_clusterizer)

        args, _ = self.mock_clusterizer.fit_transform.call_args
        np.testing.assert_array_equal(args[0], [[1.0], [2.0], [3.0]])

    def test_clusterize_hard_perfect_separation(self):
        """Test hard clustering conversion to one-hot encoding."""
        initializer = ClusterizeInitializer(is_accurate=True, is_soft=False)
        X = np.zeros((4, 1))
        self.mock_clusterizer.fit_predict.return_value = np.array([0, 1, 0, 1])

        H = initializer._clusterize(X, self.mock_clusterizer)

        expected_H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(H, expected_H)

    def test_clusterize_hard_with_outliers(self):
        """Test handling of -1 labels (noise) in hard clustering."""
        initializer = ClusterizeInitializer(is_accurate=True, is_soft=False)
        X = np.zeros((3, 1))
        self.mock_clusterizer.fit_predict.return_value = np.array([0, -1, 1])

        H = initializer._clusterize(X, self.mock_clusterizer)

        expected_H = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        np.testing.assert_array_equal(H, expected_H)

    def test_clusterize_hard_all_outliers(self):
        """Test edge case where ALL points are detected as noise."""
        initializer = ClusterizeInitializer(is_accurate=True, is_soft=False)
        X = np.zeros((3, 1))
        self.mock_clusterizer.fit_predict.return_value = np.array([-1, -1, -1])

        H = initializer._clusterize(X, self.mock_clusterizer)

        expected_H = np.array([[1.0], [1.0], [1.0]])
        np.testing.assert_array_equal(H, expected_H)

    def test_clusterize_hard_single_valid_cluster(self):
        """Test hard clustering where only 1 valid cluster is found."""
        initializer = ClusterizeInitializer(is_accurate=True, is_soft=False)
        X = np.zeros((3, 1))
        self.mock_clusterizer.fit_predict.return_value = np.array([0, 0, 0])

        H = initializer._clusterize(X, self.mock_clusterizer)

        expected_H = np.array([[1.0], [1.0], [1.0]])
        np.testing.assert_array_equal(H, expected_H)

    def test_clusterize_hard_non_contiguous_labels(self):
        """
        Test if labels are like [0, 5].
        Logic: valid_labels=[0, 5]. n_clusters=2.
        We expect indices 0 and 1 in H matrix mapping to labels 0 and 5.
        """
        initializer = ClusterizeInitializer(is_accurate=True, is_soft=False)
        X = np.zeros((2, 1))
        self.mock_clusterizer.fit_predict.return_value = np.array([0, 5])

        H = initializer._clusterize(X, self.mock_clusterizer)

        expected_H = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(H, expected_H)

    def test_clusterize_methods_missing(self):
        """Ensure correct error is raised if clusterizer lacks methods."""
        X = np.zeros((2, 1))

        init_soft = ClusterizeInitializer(is_accurate=True, is_soft=True)
        mock_bad = Mock(spec=[])
        with pytest.raises(ValueError, match="Clusterizer doesn't have required method"):
            init_soft._clusterize(X, mock_bad)

        init_hard = ClusterizeInitializer(is_accurate=True, is_soft=False)
        with pytest.raises(ValueError, match="Clusterizer doesn't have required method"):
            init_hard._clusterize(X, mock_bad)

    # 2. Tests for _fast_init (Parameter Estimation)

    def test_fast_init_parameter_setting(self):
        """
        Verify that _fast_init correctly calls estimation function
        and sets parameters using the zipped result (names, values).
        """
        initializer = ClusterizeInitializer(is_accurate=False, is_soft=True)
        initializer.n_components = 2
        initializer.models = self.dists
        initializer.estimation_strategies = [EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION]

        X = np.zeros(10)
        H = np.ones((10, 2)) * 0.5

        mock_est_func = Mock()
        mock_est_func.side_effect = [{"loc": 10.0, "scale": 1.5}, {"loc": -5.0, "scale": 0.5}]

        with patch.object(ClusterizeInitializer, "_estimation_strategies", {}) as strategies:
            strategies[EstimationStrategy.QFUNCTION] = mock_est_func

            res_dists, res_weights = initializer._fast_init(X, H, self.mock_optimizer)

        call_const = 2
        assert mock_est_func.call_count == call_const

        self.dist1.set_params_from_vector.assert_called_once()
        args0 = self.dist1.set_params_from_vector.call_args[0]
        assert "loc" in args0[0] and "scale" in args0[0]
        arg1, arg2 = 10.0, 1.5
        assert arg1 in args0[1] and arg2 in args0[1]

        self.dist2.set_params_from_vector.assert_called_once()
        args1 = self.dist2.set_params_from_vector.call_args[0]
        arg3 = -5.0
        assert arg3 in args1[1]

        assert res_weights == [0.5, 0.5]

    def test_fast_init_requires_n_components(self):
        """Ensure n_components is validated."""
        init = ClusterizeInitializer(is_accurate=False, is_soft=True)
        init.n_components = None
        with pytest.raises(ValueError, match="n_components must be set"):
            init._fast_init(np.zeros(1), np.zeros((1, 1)))

    # 3. Tests for _accurate_init (Matching Strategy Integration)

    def test_accurate_init_success_flow(self):
        """Test successful matching and parameter application."""
        initializer = ClusterizeInitializer(is_accurate=True, is_soft=True)
        initializer.n_components = 2
        initializer.models = self.dists
        initializer.estimation_strategies = [EstimationStrategy.QFUNCTION] * 2

        X = np.zeros(10)
        H = np.zeros((10, 2))

        with patch("rework_pysatl_mpest.initializers.clusterize_initializer.match_clusters_for_models") as mock_match:
            mock_match.return_value = (self.dists, [{"loc": 100.0}, {"loc": 200.0}], [0.1, 0.9])

            res_dists, res_weights = initializer._accurate_init(X, H, self.mock_optimizer)

        assert res_weights == [0.1, 0.9]

        self.dist1.set_params_from_vector.assert_called()
        arg1 = 100.0
        assert arg1 in self.dist1.set_params_from_vector.call_args[0][1]

        self.dist2.set_params_from_vector.assert_called()
        arg2 = 200.0
        assert arg2 in self.dist2.set_params_from_vector.call_args[0][1]

    def test_accurate_init_fallback(self):
        """Test that empty params from matching triggers fast_init."""
        initializer = ClusterizeInitializer(is_accurate=True, is_soft=True)
        initializer.n_components = 2
        initializer.models = self.dists
        initializer.estimation_strategies = [EstimationStrategy.QFUNCTION] * 2

        X = np.zeros(10)
        H = np.zeros((10, 2))

        with (
            patch("rework_pysatl_mpest.initializers.clusterize_initializer.match_clusters_for_models") as mock_match,
            patch.object(initializer, "_fast_init") as mock_fast,
        ):
            mock_match.return_value = (self.dists, [{}, {}], [0.5, 0.5])

            mock_fast.return_value = (self.dists, [0.33, 0.67])

            res_dists, res_weights = initializer._accurate_init(X, H, self.mock_optimizer)

            mock_fast.assert_called_once_with(X, H, self.mock_optimizer)
            assert res_weights == [0.33, 0.67]

    def test_accurate_init_validation(self):
        """Check validation of strategies count matching model count."""
        initializer = ClusterizeInitializer(is_accurate=True, is_soft=True)
        initializer.n_components = 2
        initializer.models = [self.dist1, self.dist2]
        initializer.estimation_strategies = [EstimationStrategy.QFUNCTION]

        with pytest.raises(ValueError, match="Count of models must match"):
            initializer._accurate_init(np.zeros(1), np.zeros((1, 1)))

    # 4. Tests for perform (Main Entry Point)

    def test_perform_dependency_injection(self):
        """Test that optimizer can be passed in perform or use default from init."""
        initializer = ClusterizeInitializer(
            is_accurate=True,
            is_soft=True,
            clusterizer=self.mock_clusterizer,
        )

        X = np.array([1])
        dists = self.dists

        with (
            patch.object(initializer, "_clusterize", return_value=np.zeros((1, 2))),
            patch.object(initializer, "_accurate_init", return_value=(dists, [0.5, 0.5])),
            patch("rework_pysatl_mpest.initializers.clusterize_initializer.MixtureModel"),
        ):
            initializer.perform(
                X,
                dists,
                MatchingMethod.GREEDY,
                ScoringMethod.AIC,
                [EstimationStrategy.QFUNCTION] * 2,
                optimizer=self.mock_optimizer,
            )

    def test_perform_flow_integration(self):
        """
        Full integration check of perform method:
        - Setup variables
        - Call clusterize
        - Call init strategy
        - Create MixtureModel
        """
        initializer = ClusterizeInitializer(
            is_accurate=False, is_soft=True, clusterizer=self.mock_clusterizer, optimizer=self.mock_optimizer
        )

        X = np.array([10.0, 20.0])
        dists = self.dists
        est_strategies = [EstimationStrategy.QFUNCTION, EstimationStrategy.QFUNCTION]

        with (
            patch.object(initializer, "_clusterize") as mock_clust,
            patch.object(initializer, "_fast_init") as mock_fast,
            patch("rework_pysatl_mpest.initializers.clusterize_initializer.MixtureModel") as MockMixtureClass,
        ):
            mock_clust.return_value = "H_Matrix"
            mock_fast.return_value = (dists, [0.2, 0.8])

            MockMixtureClass.return_value = "Final_Mixture"

            result = initializer.perform(
                X, dists, MatchingMethod.GREEDY, ScoringMethod.LIKELIHOOD, est_strategies, optimizer=self.mock_optimizer
            )

            assert result == "Final_Mixture"
            n_components = 2
            assert initializer.n_components == n_components
            assert initializer.models == dists
            assert initializer.method == MatchingMethod.GREEDY
            assert initializer.score_func == ScoringMethod.LIKELIHOOD
            assert initializer.estimation_strategies == est_strategies

            mock_clust.assert_called_once()
            mock_fast.assert_called_once_with(X, "H_Matrix", self.mock_optimizer)

            call_args = MockMixtureClass.call_args
            passed_weights = call_args[0][1]
            np.testing.assert_array_almost_equal(passed_weights, [0.2, 0.8])

    def test_initializer_abstract_perform(self):
        class ConcreteInitializer(Initializer):
            def perform(self, X, dists, method, score_func, estimation_strategies):
                return super().perform(X, dists, method, score_func, estimation_strategies)

        init = ConcreteInitializer()
        with pytest.raises(NotImplementedError):
            init.perform(None, [], None, None, [])

    # 5. Other tests

    def test_clusterize_hard_exception(self):
        init = ClusterizeInitializer(is_accurate=True, is_soft=False)
        mock_clusterizer = Mock()
        mock_clusterizer.fit_predict.side_effect = Exception("Fail")

        with pytest.raises(ValueError, match="Hard clusterizer failed: Fail"):
            init._clusterize(np.array([1]), mock_clusterizer)

    def test_clusterize_soft_exception(self):
        init = ClusterizeInitializer(is_accurate=True, is_soft=True)
        mock_clusterizer = Mock()
        mock_clusterizer.fit_transform.side_effect = Exception("Fail")

        with pytest.raises(ValueError, match="Fuzzy clusterizer failed: Fail"):
            init._clusterize(np.array([1]), mock_clusterizer)

    def test_clusterize_perform_no_clusterizer(self):
        init = ClusterizeInitializer(is_accurate=True, is_soft=True, clusterizer=None)
        with pytest.raises(TypeError, match="Clusterizer not found"):
            init.perform([1], [], None, None, [], clusterizer=None)

    def test_accurate_init_fallback_on_partial_params(self):
        init = ClusterizeInitializer(is_accurate=True, is_soft=True)
        init.n_components = 2
        init.models = [Mock(), Mock()]
        init.estimation_strategies = [EstimationStrategy.QFUNCTION] * 2

        with (
            patch("rework_pysatl_mpest.initializers.clusterize_initializer.match_clusters_for_models") as mock_match,
            patch.object(init, "_fast_init") as mock_fast,
        ):
            mock_match.return_value = (init.models, [{"loc": 1}, {}], [0.5, 0.5])
            mock_fast.return_value = (init.models, [0.5, 0.5])

            init._accurate_init(np.zeros(1), np.zeros((1, 2)))
            mock_fast.assert_called_once()

    def test_clusterize_hard_zero_clusters_edge_case(self):
        init = ClusterizeInitializer(is_accurate=True, is_soft=False)
        mock_clusterizer = Mock()
        mock_clusterizer.fit_predict.return_value = np.array([])
        H = init._clusterize(np.array([1]), mock_clusterizer)
        assert H.shape[1] == 1
