import numpy as np
import pytest
from rework_pysatl_mpest.preprocessing.components_family.criterions.frequency_criterions import (
    CDct,
    CDctEnergy,
    CSpecBandwidth,
    CSpecCentroid,
    CSpecDecrease,
    CSpecEnergy,
    CSpecEntropy,
    CSpecFlatness,
    CSpecRolloff,
    CSpecSlope,
    CWaveletEnergy,
    CWaveletEntropy,
    CWaveletLarge,
    CWaveletMean,
    CWaveletStd,
)


class TestFrequencyCriterions:
    @pytest.mark.parametrize(
        "dct_type,hist,expected",
        [
            (1, [1, 1, 1, 1, 1], 0),
            (1, [1, 2, 3, 4, 5], -0.20996),
            (1, [1, 2, 4, 3, 0], 0.02297),
            (2, [1, 1, 1, 1, 1], 0),
            (2, [1, 2, 3, 4, 5], 0),
            (2, [1, 2, 4, 3, 0], -0.29953),
            (3, [1, 1, 1, 1, 1], 0),
            (3, [1, 2, 3, 4, 5], -0.01893),
            (3, [1, 2, 4, 3, 0], 0.09732),
        ],
    )
    def test_cdct(self, dct_type, hist, expected):
        error_rate = 1e-5
        result = CDct(dct_type).score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0),
            ([1, 2, 4, 3, 0], 0.00027),
        ],
    )
    def test_cdct_energy(self, hist, expected):
        error_rate = 1e-5
        result = CDctEnergy().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.69712),
            ([1, 2, 4, 3, 0], 0.66418),
        ],
    )
    def test_spec_bandwidth(self, hist, expected):
        error_rate = 1e-5
        result = CSpecBandwidth().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.43463),
            ([1, 2, 4, 3, 0], 0.48264),
        ],
    )
    def test_spec_centroid(self, hist, expected):
        error_rate = 1e-5
        result = CSpecCentroid().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0.00002),
            ([1, 2, 3, 4, 5], -0.23606),
            ([1, 2, 4, 3, 0], -0.50523),
        ],
    )
    def test_spec_decrease(self, hist, expected):
        error_rate = 1e-5
        result = CSpecDecrease().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 1),
            ([1, 2, 3, 4, 5], 1.11111),
            ([1, 2, 4, 3, 0], 1.24999),
        ],
    )
    def test_spec_energy(self, hist, expected):
        error_rate = 1e-5
        result = CSpecEnergy().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.85048),
            ([1, 2, 4, 3, 0], 0.46104),
        ],
    )
    def test_spec_entropy(self, hist, expected):
        error_rate = 1e-5
        result = CSpecEntropy().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.75605),
            ([1, 2, 4, 3, 0], 0.77272),
        ],
    )
    def test_spec_flatness(self, hist, expected):
        error_rate = 1e-5
        result = CSpecFlatness().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 1),
            ([1, 2, 4, 3, 0], 1),
        ],
    )
    def test_spec_rolloff(self, hist, expected):
        error_rate = 1e-5
        result = CSpecRolloff().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], -0.49999),
            ([1, 2, 3, 4, 5], -0.41237),
            ([1, 2, 4, 3, 0], -0.42193),
        ],
    )
    def test_spec_slope(self, hist, expected):
        error_rate = 1e-5
        result = CSpecSlope().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "level,hist,expected",
        [
            (1, [1, 1, 1, 1, 1], 1),
            (1, [1, 2, 3, 4, 5], 0.86538),
            (1, [1, 2, 4, 3, 0], 0.41666),
            (2, [1, 1, 1, 1, 1], 0),
            (2, [1, 2, 3, 4, 5], 0.09615),
            (2, [1, 2, 4, 3, 0], 0.41666),
            (3, [1, 1, 1, 1, 1], 0),
            (3, [1, 2, 3, 4, 5], 0.03076),
            (3, [1, 2, 4, 3, 0], 0.13333),
        ],
    )
    def test_wavelet_energy(self, level, hist, expected):
        error_rate = 1e-5
        result = CWaveletEnergy(level=level).score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "level,hist,expected",
        [
            (1, [1, 1, 1, 1, 1], 0),
            (1, [1, 2, 3, 4, 5], 0),
            (1, [1, 2, 4, 3, 0], 0),
            (2, [1, 1, 1, 1, 1], 0),
            (2, [1, 2, 3, 4, 5], 0),
            (2, [1, 2, 4, 3, 0], 0),
            (3, [1, 1, 1, 1, 1], 0),
            (3, [1, 2, 3, 4, 5], 0),
            (3, [1, 2, 4, 3, 0], 0),
        ],
    )
    def test_wavelet_entropy(self, level, hist, expected):
        error_rate = 1e-5
        result = CWaveletEntropy(level=level).score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "level,hist,expected",
        [
            (1, [1, 1, 1, 1, 1], 0.56568),
            (1, [1, 2, 3, 4, 5], 0.70710),
            (1, [1, 2, 4, 3, 0], 0.35355),
            (2, [1, 1, 1, 1, 1], 0),
            (2, [1, 2, 3, 4, 5], -0.23570),
            (2, [1, 2, 4, 3, 0], 0.35355),
            (3, [1, 1, 1, 1, 1], 0),
            (3, [1, 2, 3, 4, 5], -0.06666),
            (3, [1, 2, 4, 3, 0], -0.1),
        ],
    )
    def test_wavelet_mean(self, level, hist, expected):
        error_rate = 1e-5
        result = CWaveletMean(level=level).score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "level,hist,expected",
        [
            (1, [1, 1, 1, 1, 1], 0),
            (1, [1, 2, 3, 4, 5], 0),
            (1, [1, 2, 4, 3, 0], 0),
            (2, [1, 1, 1, 1, 1], 0),
            (2, [1, 2, 3, 4, 5], 0),
            (2, [1, 2, 4, 3, 0], 0),
            (3, [1, 1, 1, 1, 1], 0),
            (3, [1, 2, 3, 4, 5], 0.06666),
            (3, [1, 2, 4, 3, 0], 0.1),
        ],
    )
    def test_wavelet_std(self, level, hist, expected):
        error_rate = 1e-5
        result = CWaveletStd(level=level).score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "level,hist,expected",
        [
            (1, [1, 1, 1, 1, 1], 1),
            (1, [1, 2, 3, 4, 5], 1),
            (1, [1, 2, 4, 3, 0], 1),
            (2, [1, 1, 1, 1, 1], 0),
            (2, [1, 2, 3, 4, 5], 1),
            (2, [1, 2, 4, 3, 0], 1),
            (3, [1, 1, 1, 1, 1], 0),
            (3, [1, 2, 3, 4, 5], 0.5),
            (3, [1, 2, 4, 3, 0], 0.5),
        ],
    )
    def test_wavelet_large(self, level, hist, expected):
        error_rate = 1e-5
        result = CWaveletLarge(level=level).score(hist)
        assert np.abs(expected - result) < error_rate
