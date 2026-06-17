import numpy as np
import pytest
from pysatl_mpest.preprocessing.components_family.criterions.peaks_criterions import (
    CPeaksCount,
    CPeaksDistMax,
    CPeaksDistMean,
    CPeaksDistMin,
    CPeaksFirst,
    CPeaksLast,
    CPeaksMax,
    CPeaksMean,
    CPeaksMin,
    CPeaksWidthMax,
    CPeaksWidthMean,
    CPeaksWidthMin,
    CValleysDistMax,
    CValleysDistMean,
    CValleysDistMin,
    CValleysMax,
    CValleysMean,
    CValleysMin,
    CValleysWidthMax,
    CValleysWidthMean,
    CValleysWidthMin,
)


class TestPeaksCriterions:
    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 1),
            ([1, 2, 0, 4, 3, 3, 4, 2], 3),
            ([1, 2, 0, 0, 1, 3, 4, 2], 2),
        ],
    )
    def test_peaks_count(self, hist, expected):
        result = CPeaksCount().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0),
            ([2, 1, 0, 0, 1, 3, 4, 2], 1),
        ],
    )
    def test_peaks_first(self, hist, expected):
        result = CPeaksFirst().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0),
            ([1, 2, 0, 0, 1, 3, 4, 5], 1),
        ],
    )
    def test_peaks_last(self, hist, expected):
        result = CPeaksLast().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 1),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.5625),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.3125),
        ],
    )
    def test_peaks_width_max(self, hist, expected):
        result = CPeaksWidthMax().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 1),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.4375),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.25),
        ],
    )
    def test_peaks_width_mean(self, hist, expected):
        result = CPeaksWidthMean().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 1),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.1875),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.1875),
        ],
    )
    def test_peaks_width_min(self, hist, expected):
        result = CPeaksWidthMin().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.25),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.3125),
        ],
    )
    def test_valleys_width_max(self, hist, expected):
        result = CValleysWidthMax().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.171875),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.3125),
        ],
    )
    def test_valleys_width_mean(self, hist, expected):
        result = CValleysWidthMean().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.09375),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.3125),
        ],
    )
    def test_valleys_width_min(self, hist, expected):
        result = CValleysWidthMin().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.25),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.5),
        ],
    )
    def test_peaks_dist_max(self, hist, expected):
        result = CPeaksDistMax().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.1875),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.5),
        ],
    )
    def test_peaks_dist_mean(self, hist, expected):
        result = CPeaksDistMean().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.125),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.5),
        ],
    )
    def test_peaks_dist_min(self, hist, expected):
        result = CPeaksDistMin().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.125),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0),
        ],
    )
    def test_valleys_dist_max(self, hist, expected):
        result = CValleysDistMax().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.125),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0),
        ],
    )
    def test_valleys_dist_mean(self, hist, expected):
        result = CValleysDistMean().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.125),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0),
        ],
    )
    def test_valleys_dist_min(self, hist, expected):
        result = CValleysDistMin().score(hist)
        assert expected == result

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0.125),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.21052),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.30769),
        ],
    )
    def test_peaks_max(self, hist, expected):
        error_rate = 1e-5
        result = CPeaksMax().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0.125),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.17543),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.23076),
        ],
    )
    def test_peaks_mean(self, hist, expected):
        error_rate = 1e-5
        result = CPeaksMean().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0.125),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.10526),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0.15384),
        ],
    )
    def test_peaks_min(self, hist, expected):
        error_rate = 1e-5
        result = CPeaksMin().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.15789),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0),
        ],
    )
    def test_valleys_max(self, hist, expected):
        error_rate = 1e-5
        result = CValleysMax().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0.07894),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0),
        ],
    )
    def test_valleys_mean(self, hist, expected):
        error_rate = 1e-5
        result = CValleysMean().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1, 1, 1, 1], 0),
            ([1, 2, 0, 4, 3, 3, 4, 2], 0),
            ([1, 2, 0, 0, 1, 3, 4, 2], 0),
        ],
    )
    def test_valleys_min(self, hist, expected):
        error_rate = 1e-5
        result = CValleysMin().score(hist)
        assert np.abs(expected - result) < error_rate
