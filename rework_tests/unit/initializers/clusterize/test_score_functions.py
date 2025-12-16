"""Tests for scoring functions (AIC, LogLikelihood)."""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from unittest.mock import Mock

import numpy as np
import pytest
from rework_pysatl_mpest.core import MixtureModel
from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.initializers.clusterize.score_functions import (
    _calculate_component_aic,
    _calculate_component_log_likelihood,
    _calculate_mixture_aic,
    _calculate_mixture_log_likelihood,
)


class TestScoreFunctions:
    def setup_method(self):
        self.mock_model = Mock(spec=ContinuousDistribution)
        self.mock_model.params = {"loc", "scale"}

    @pytest.mark.parametrize(
        "log_probs, weights, expected_ll",
        [
            (np.array([-1.0, -2.0, -3.0]), np.array([1.0, 0.5, 0.0]), -2.0),
        ],
    )
    def test_calculate_component_log_likelihood(self, log_probs, weights, expected_ll):
        self.mock_model.lpdf = Mock(return_value=log_probs)
        X = np.array([1, 2, 3])  # Dummy data

        # Expected: 1*(-1) + 0.5*(-2) + 0*(-3) = -1 - 1 = -2.0
        ll = _calculate_component_log_likelihood(self.mock_model, X, weights)
        assert ll == pytest.approx(expected_ll)

    @pytest.mark.parametrize(
        "log_prob, n_params, expected_aic",
        [
            (np.array([-1.0]), 2, 6.0),
        ],
    )
    def test_calculate_component_aic(self, log_prob, n_params, expected_aic):
        # Настраиваем модель
        self.mock_model.params = {"p" + str(i) for i in range(n_params)}
        self.mock_model.lpdf.return_value = log_prob

        X = np.array([1])
        H_k = np.array([1.0])

        # LL = -1.0. k=2.
        # AIC = 2*k - 2*LL = 4 - 2*(-1) = 6.0
        aic = _calculate_component_aic(self.mock_model, X, H_k)
        assert aic == pytest.approx(expected_aic)

    def test_calculate_mixture_aic_and_likelihood(self):
        mock_mixture = Mock(spec=MixtureModel)
        expected_ll = -5.0
        expected_aic = 16.0

        mock_mixture.loglikelihood.return_value = np.array([-2.0, -3.0])

        c1 = Mock()
        c1.params = {"a": 1}
        c2 = Mock()
        c2.params = {"b": 2}

        mock_mixture.components = [c1, c2]
        mock_mixture.n_components = 2

        X = np.array([1, 2])

        ll = _calculate_mixture_log_likelihood(mock_mixture, X)
        assert ll == expected_ll

        # AIC = 2k - 2LL = 2(3) - 2(-5) = 6 + 10 = 16.0
        aic = _calculate_mixture_aic(mock_mixture, X)
        assert aic == expected_aic
