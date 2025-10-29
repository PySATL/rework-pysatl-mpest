import numpy as np
import pytest
from rework_pysatl_mpest.preprocessing.components_family.criterions.sample_criterions import (
    CBootKurt,
    CBootMean,
    CBootVar,
    CGmean,
    CHillAbs,
    CIqr,
    CKurt,
    CKurtMoors,
    CLogRatio,
    CMad,
    CMaxZscore,
    CMean,
    CMedian,
    CNegativeValue,
    COutlierFraction,
    CPercentileExtreme,
    CPercentileMedian,
    CPercentileRange,
    CPercentileTail,
    CRange,
    CSkew,
    CSkewBowley,
    CSpacingGap,
    CSpacingGini,
    CSpacingVar,
    CStd,
)


class TestSampleCriterions:
    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.31044),
            ([1, 2, 4, 3, 0], 0.29186),
        ],
    )
    def test_boot_kurt(self, hist, expected):
        error_rate = 1e-5
        result = CBootKurt(state=52).score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.43727),
            ([1, 2, 4, 3, 0], 0.42539),
        ],
    )
    def test_boot_mean(self, hist, expected):
        error_rate = 1e-5
        result = CBootMean(state=52).score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.62914),
            ([1, 2, 4, 3, 0], 0.6622),
        ],
    )
    def test_boot_var(self, hist, expected):
        error_rate = 1e-5
        result = CBootVar(state=52).score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 1),
            ([1, 2, 3, 4, 5], 2.60517),
            ([1, 2, 4, 3, 0], 0),
        ],
    )
    def test_gmean(self, hist, expected):
        error_rate = 1e-5
        result = CGmean().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.39925),
            ([1, 2, 4, 3, 0], 0.5493),
        ],
    )
    def test_hill_abs(self, hist, expected):
        error_rate = 1e-5
        result = CHillAbs().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 1),
            ([1, 2, 3, 4, 5], 3),
            ([1, 2, 4, 3, 0], 2),
        ],
    )
    def test_median(self, hist, expected):
        error_rate = 1e-5
        result = CMedian().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 1),
            ([1, 2, 3, 4, 5], 3),
            ([1, 2, 4, 3, 0], 2),
        ],
    )
    def test_mean(self, hist, expected):
        error_rate = 1e-5
        result = CMean().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 1),
            ([1, 2, 4, 3, 0], 1),
        ],
    )
    def test_mad(self, hist, expected):
        error_rate = 1e-5
        result = CMad().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0),
            ([-1, 2, 4, 3, 0], 1),
        ],
    )
    def test_negative_value(self, hist, expected):
        error_rate = 1e-5
        result = CNegativeValue().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], -1.3),
            ([1, 2, 4, 3, 0], -1.3),
        ],
    )
    def test_kurt(self, hist, expected):
        error_rate = 1e-5
        result = CKurt().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.66666),
            ([1, 2, 4, 3, 0], 1),
        ],
    )
    def test_iqr(self, hist, expected):
        error_rate = 1e-5
        result = CIqr().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0),
            ([1, 2, 4, 3, 0], 0),
        ],
    )
    def test_log_ration(self, hist, expected):
        error_rate = 1e-5
        result = CLogRatio().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0),
            ([1, 2, 4, 3, 0], 0),
        ],
    )
    def test_outlier_fraction(self, hist, expected):
        error_rate = 1e-5
        result = COutlierFraction().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 4),
            ([1, 2, 4, 3, 0], 4),
        ],
    )
    def test_range(self, hist, expected):
        error_rate = 1e-5
        result = CRange().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0),
            ([-1, 2, 4, 3, 0], -0.15798),
        ],
    )
    def test_skew(self, hist, expected):
        error_rate = 1e-5
        result = CSkew().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0),
            ([-1, 2, 4, 3, 0], -0.33333),
        ],
    )
    def test_skew_bowley(self, hist, expected):
        error_rate = 1e-5
        result = CSkewBowley().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0),
            ([1, 2, 4, 3, 0], 0),
        ],
    )
    def test_spacing_gap(self, hist, expected):
        error_rate = 1e-5
        result = CSpacingGap().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0),
            ([-1, 2, 4, 3, 0], 0.15),
        ],
    )
    def test_spacing_gini(self, hist, expected):
        error_rate = 1e-5
        result = CSpacingGini().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0),
            ([-1, 2, 4, 3, 0], 0.1875),
        ],
    )
    def test_spacing_var(self, hist, expected):
        error_rate = 1e-5
        result = CSpacingVar().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 1.41421),
            ([-1, 2, 4, 3, 0], 1.85472),
        ],
    )
    def test_std(self, hist, expected):
        error_rate = 1e-5
        result = CStd().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 1),
            ([1, 2, 4, 3, 0], 1),
        ],
    )
    def test_percentile_median(self, hist, expected):
        error_rate = 1e-5
        result = CPercentileMedian().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 3.59999),
            ([1, 2, 4, 3, 0], 3.59999),
        ],
    )
    def test_percentile_range(self, hist, expected):
        error_rate = 1e-5
        result = CPercentileRange().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0),
            ([-1, 2, 4, 3, 0], -1),
        ],
    )
    def test_percentile_tail(self, hist, expected):
        error_rate = 1e-5
        result = CPercentileTail().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 0.036),
            ([-1, 2, 4, 3, 0], 0.036),
        ],
    )
    def test_percentile_extreme(self, hist, expected):
        error_rate = 1e-5
        result = CPercentileExtreme().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 1.41421),
            ([1, 2, 4, 3, 0], 1.41421),
        ],
    )
    def test_zscore(self, hist, expected):
        error_rate = 1e-5
        result = CMaxZscore().score(hist)
        assert np.abs(expected - result) < error_rate

    @pytest.mark.parametrize(
        "hist,expected",
        [
            ([1, 1, 1, 1, 1], 0),
            ([1, 2, 3, 4, 5], 2.66666),
            ([1, 2, 4, 3, 0], 2.2),
        ],
    )
    def test_kurt_moors(self, hist, expected):
        error_rate = 1e-5
        result = CKurtMoors().score(hist)
        assert np.abs(expected - result) < error_rate
