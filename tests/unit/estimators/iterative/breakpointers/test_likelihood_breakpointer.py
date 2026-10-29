"""Tests for LikelihoodBreakpointer"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from unittest.mock import Mock

import numpy as np
import pytest
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import PipelineState
from pysatl_mpest.estimators.iterative.breakpointers import LikelihoodBreakpointer


@pytest.fixture
def mock_mixture_with_likelihood():
    """Returns a mock mixture that returns predefined log-likelihoods."""

    def make_mixture(ll_values):
        mixture = Mock(spec=MixtureModel)
        gen = iter(ll_values)

        def lpdf(X):
            values = np.zeros_like(X, dtype=np.float64)
            values[0] = next(gen)
            return values

        mixture.lpdf.side_effect = lpdf
        mixture.loglikelihood.side_effect = lambda X: float(np.sum(mixture.lpdf(X)))
        return mixture

    return make_mixture


@pytest.fixture
def mock_mixture_with_log_densities():
    """Return a factory of mixtures with one log-density per input observation."""

    def make_mixture(log_densities):
        mixture = Mock(spec=MixtureModel)
        values = iter(log_densities)

        def lpdf(X):
            result = np.asarray(next(values), dtype=np.float64)
            assert result.shape == np.asarray(X).shape
            return result

        mixture.lpdf.side_effect = lpdf
        mixture.loglikelihood.side_effect = lambda X: float(np.sum(mixture.lpdf(X)))
        return mixture

    return make_mixture


@pytest.fixture
def dummy_state_factory():
    """Factory to create PipelineState with custom mixture."""

    def _make_state(mixture, X=None):
        if X is None:
            X = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        return PipelineState(X, None, None, mixture, None)

    return _make_state


# --- Initialization Tests ---


class TestInitialization:
    @pytest.mark.parametrize("threshold", [0.01, 0.5, 10.0])
    def test_initialization_with_valid_threshold(self, threshold):
        bp = LikelihoodBreakpointer(threshold)
        assert bp.threshold == threshold
        assert bp._likelihood_old is None
        assert bp._support_mask_old is None

    def test_initialization_rejects_non_positive_threshold(self):
        with pytest.raises(ValueError, match="The threshold must be greater than 0"):
            LikelihoodBreakpointer(0.0)
        with pytest.raises(ValueError, match="The threshold must be greater than 0"):
            LikelihoodBreakpointer(-1.0)


# --- Core Logic Tests ---


class TestCheckLogic:
    def test_first_call_never_stops(self, mock_mixture_with_likelihood, dummy_state_factory):
        FIRST_CALL = 5.0
        mixture = mock_mixture_with_likelihood([5.0])
        state = dummy_state_factory(mixture)
        bp = LikelihoodBreakpointer(0.1)
        assert not bp.check(state)
        assert bp._likelihood_old == FIRST_CALL

    def test_convergence_detected(self, mock_mixture_with_likelihood, dummy_state_factory):
        mixture = mock_mixture_with_likelihood([10.0, 10.05])
        state = dummy_state_factory(mixture)
        bp = LikelihoodBreakpointer(0.1)

        assert not bp.check(state)
        assert bp.check(state)

    def test_no_convergence_continues(self, mock_mixture_with_likelihood, dummy_state_factory):
        mixture = mock_mixture_with_likelihood([5.0, 6.0])
        state = dummy_state_factory(mixture)
        bp = LikelihoodBreakpointer(0.5)

        assert not bp.check(state)
        assert not bp.check(state)

    def test_reset_after_convergence_enables_reuse(self, mock_mixture_with_likelihood, dummy_state_factory):
        mixture = mock_mixture_with_likelihood([10.0, 10.01, 20.0, 20.005])
        state = dummy_state_factory(mixture)
        bp = LikelihoodBreakpointer(0.02)

        assert not bp.check(state)
        assert bp.check(state)

        assert not bp.check(state)
        assert bp.check(state)
        assert bp._likelihood_old is None


@pytest.mark.filterwarnings("error")
class TestSupportFiltering:
    """Convergence compares sums over the same supported observations."""

    def test_points_outside_the_support_are_ignored(self, mock_mixture_with_log_densities, dummy_state_factory):
        mixture = mock_mixture_with_log_densities([[-np.inf, -1.0, -2.0], [-np.inf, -1.0, -2.00001]])
        state = dummy_state_factory(mixture)
        bp = LikelihoodBreakpointer(0.1)

        assert not bp.check(state)
        assert bp.check(state)

    def test_change_in_the_supported_points_is_not_convergence(
        self, mock_mixture_with_log_densities, dummy_state_factory
    ):
        mixture = mock_mixture_with_log_densities([[-np.inf, -1.0], [-5.0, -1.0]])
        state = dummy_state_factory(mixture, X=np.array([1.0, 2.0]))
        bp = LikelihoodBreakpointer(0.1)

        assert not bp.check(state)
        assert not bp.check(state)

    @pytest.mark.parametrize(
        ("before", "after"),
        [([0.0, -np.inf, -np.inf], [0.0, 0.0, 0.0]), ([0.0, 0.0, 0.0], [0.0, -np.inf, -np.inf])],
        ids=["expanding", "contracting"],
    )
    def test_support_change_with_the_same_likelihood_is_not_convergence(
        self, mock_mixture_with_log_densities, dummy_state_factory, before, after
    ):
        mixture = mock_mixture_with_log_densities([before, after, after])
        state = dummy_state_factory(mixture)
        bp = LikelihoodBreakpointer(0.1)

        assert not bp.check(state)
        assert not bp.check(state)
        assert bp.check(state)

    def test_supported_points_change_with_the_same_count_is_not_convergence(
        self, mock_mixture_with_log_densities, dummy_state_factory
    ):
        mixture = mock_mixture_with_log_densities([[-1.0, -np.inf], [-np.inf, -1.0], [-np.inf, -1.0]])
        state = dummy_state_factory(mixture, X=np.array([1.0, 2.0]))
        bp = LikelihoodBreakpointer(0.1)

        assert not bp.check(state)
        assert not bp.check(state)
        assert bp.check(state)

    def test_support_change_updates_the_likelihood_baseline(self, mock_mixture_with_log_densities, dummy_state_factory):
        mixture = mock_mixture_with_log_densities([[-np.inf, -1.0], [0.0, -5.0], [0.0, -5.00001]])
        state = dummy_state_factory(mixture, X=np.array([1.0, 2.0]))
        bp = LikelihoodBreakpointer(0.1)

        assert not bp.check(state)
        assert not bp.check(state)
        assert bp.check(state)

    def test_support_mask_is_reset_after_convergence(self, mock_mixture_with_log_densities, dummy_state_factory):
        mixture = mock_mixture_with_log_densities([[-np.inf, -1.0]] * 3)
        state = dummy_state_factory(mixture, X=np.array([1.0, 2.0]))
        bp = LikelihoodBreakpointer(0.1)

        assert not bp.check(state)
        assert bp.check(state)
        assert bp._likelihood_old is None
        assert bp._support_mask_old is None
        assert not bp.check(state)
        assert bp._likelihood_old == -1.0
        np.testing.assert_array_equal(bp._support_mask_old, [False, True])


class TestNonFiniteStops:
    """Invalid sums and empty support stop with a warning and clear the baseline."""

    @pytest.mark.parametrize("invalid", [np.nan, np.inf], ids=["nan", "positive-infinity"])
    def test_non_finite_log_density_stops_with_a_warning(
        self, mock_mixture_with_log_densities, dummy_state_factory, invalid
    ):
        mixture = mock_mixture_with_log_densities([[-np.inf, -1.0, invalid]])
        state = dummy_state_factory(mixture)
        bp = LikelihoodBreakpointer(0.1)

        with pytest.warns(RuntimeWarning, match="log-likelihood cannot be evaluated"):
            assert bp.check(state)

        assert bp._likelihood_old is None
        assert bp._support_mask_old is None

    @pytest.mark.parametrize("log_densities", [[-np.inf, -np.inf], []], ids=["outside-support", "empty-sample"])
    def test_no_point_in_the_support_stops_with_a_warning(
        self, mock_mixture_with_log_densities, dummy_state_factory, log_densities
    ):
        mixture = mock_mixture_with_log_densities([log_densities])
        state = dummy_state_factory(mixture, X=np.arange(len(log_densities), dtype=np.float64))
        bp = LikelihoodBreakpointer(0.1)

        with pytest.warns(RuntimeWarning, match="no data point is inside the support"):
            assert bp.check(state)

        assert bp._likelihood_old is None
        assert bp._support_mask_old is None

    @pytest.mark.parametrize("invalid", [[np.nan, -1.0], [np.inf, -1.0], [-np.inf, -np.inf]])
    def test_state_is_reset_after_a_non_finite_stop(
        self, mock_mixture_with_log_densities, dummy_state_factory, invalid
    ):
        mixture = mock_mixture_with_log_densities([[-np.inf, -1.0], invalid, [-np.inf, -1.0]])
        state = dummy_state_factory(mixture, X=np.array([1.0, 2.0]))
        bp = LikelihoodBreakpointer(0.1)

        assert not bp.check(state)
        with pytest.warns(RuntimeWarning, match="log-likelihood cannot be evaluated"):
            assert bp.check(state)

        assert bp._likelihood_old is None
        assert bp._support_mask_old is None
        assert not bp.check(state)
        assert bp._likelihood_old == -1.0
        np.testing.assert_array_equal(bp._support_mask_old, [False, True])
