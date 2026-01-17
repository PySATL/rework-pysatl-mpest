"""Tests for LikelihoodBreakpointer"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from unittest.mock import Mock

import numpy as np
import pytest
from rework_pysatl_mpest.estimators.iterative import PipelineState
from rework_pysatl_mpest.estimators.iterative.breakpointers.likelihood_breakpointer import LikelihoodBreakpointer

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]


@pytest.fixture
def mock_mixture_with_likelihood():
    """Returns a mock mixture that returns predefined log-likelihoods."""

    def make_mixture(ll_values):
        mixture = Mock()
        gen = iter(ll_values)
        mixture.loglikelihood = lambda X: next(gen)
        return mixture

    return make_mixture


@pytest.fixture
def dummy_state_factory():
    """Factory to create PipelineState with custom mixture."""

    def _make_state(mixture, X=None):
        if X is None:
            X = np.array([1.0, 2.0, 3.0])
        return PipelineState(X, None, None, mixture, None)

    return _make_state


# --- Initialization Tests ---


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestInitialization:
    @pytest.mark.parametrize("threshold", [0.01, 0.5, 10.0])
    def test_initialization_with_valid_threshold(self, dtype, threshold):
        bp = LikelihoodBreakpointer[dtype](threshold)
        assert bp.threshold == threshold
        assert bp._likelihood_old is None

    def test_initialization_rejects_non_positive_threshold(self, dtype):
        with pytest.raises(ValueError, match="The threshold must be greater than 0"):
            LikelihoodBreakpointer[dtype](0.0)
        with pytest.raises(ValueError, match="The threshold must be greater than 0"):
            LikelihoodBreakpointer[dtype](-1.0)


# --- Core Logic Tests ---


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestCheckLogic:
    def test_first_call_never_stops(self, mock_mixture_with_likelihood, dummy_state_factory, dtype):
        FIRST_CALL = 5.0
        mixture = mock_mixture_with_likelihood([5.0])
        state = dummy_state_factory(mixture)
        bp = LikelihoodBreakpointer[dtype](0.1)
        assert not bp.check(state)
        assert bp._likelihood_old == FIRST_CALL

    def test_convergence_detected(self, mock_mixture_with_likelihood, dummy_state_factory, dtype):
        mixture = mock_mixture_with_likelihood([10.0, 10.05])
        state = dummy_state_factory(mixture)
        bp = LikelihoodBreakpointer[dtype](0.1)

        assert not bp.check(state)
        assert bp.check(state)

    def test_no_convergence_continues(self, mock_mixture_with_likelihood, dummy_state_factory, dtype):
        mixture = mock_mixture_with_likelihood([5.0, 6.0])
        state = dummy_state_factory(mixture)
        bp = LikelihoodBreakpointer[dtype](0.5)

        assert not bp.check(state)
        assert not bp.check(state)

    def test_reset_after_convergence_enables_reuse(self, mock_mixture_with_likelihood, dummy_state_factory, dtype):
        mixture = mock_mixture_with_likelihood([10.0, 10.01, 20.0, 20.005])
        state = dummy_state_factory(mixture)
        bp = LikelihoodBreakpointer[dtype](0.02)

        # First cycle
        assert not bp.check(state)
        assert bp.check(state)

        # After reset, should behave like new
        assert not bp.check(state)
        assert bp.check(state)

        # Internal state reset
        assert bp._likelihood_old is None
