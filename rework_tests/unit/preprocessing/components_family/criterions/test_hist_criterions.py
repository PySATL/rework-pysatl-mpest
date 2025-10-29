import numpy as np
import pytest
from rework_pysatl_mpest.preprocessing.components_family.criterions.hist_criterions import (
    CHistEnergy,
    CHistEntropy,
    CHistFlat,
    CHistLength,
    CHistUniform,
    CSobelCount,
    CSobelMax,
    CSobelMean,
    CSobelMin,
)


class TestHistCriterions:
    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0.2),
            ([1, 2, 3, 4, 5], 0.24444),
            ([1, 2, 4, 3, 0], 0.3),
        ],
    )
    def test_hist_energy(self, hist, expected):
        error_rate = 1e-5
        result = CHistEnergy().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 2.32192),
            ([1, 2, 3, 4, 5], 2.14925),
            ([1, 2, 4, 3, 0], 1.84643),
        ],
    )
    def test_hist_entropy(self, hist, expected):
        error_rate = 1e-5
        result = CHistEntropy().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 1),
            ([1, 2, 3, 4, 5], 0),
            ([1, 2, 4, 3, 0], 0),
        ],
    )
    def test_hist_flat(self, hist, expected):
        error_rate = 1e-5
        result = CHistFlat().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.26666),
            ([1, 2, 4, 3, 0], 0.7),
        ],
    )
    def test_hist_length(self, hist, expected):
        error_rate = 1e-5
        result = CHistLength().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.17778),
            ([1, 2, 4, 3, 0], 0.31596),
        ],
    )
    def test_hist_uniform(self, hist, expected):
        error_rate = 1e-5
        result = CHistUniform().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 1),
            ([1, 2, 4, 3, 0], 1),
        ],
    )
    def test_sobel_count(self, hist, expected):
        error_rate = 1e-5
        result = CSobelCount().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.13333),
            ([1, 2, 4, 3, 0], 0.4),
        ],
    )
    def test_sobel_max(self, hist, expected):
        error_rate = 1e-5
        result = CSobelMax().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.06666),
            ([1, 2, 4, 3, 0], 0.09999),
        ],
    )
    def test_sobel_min(self, hist, expected):
        error_rate = 1e-5
        result = CSobelMin().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.10666),
            ([1, 2, 4, 3, 0], 0.24),
        ],
    )
    def test_sobel_mean(self, hist, expected):
        error_rate = 1e-5
        result = CSobelMean().score(hist)
        assert np.abs(expected - result) < error_rate
