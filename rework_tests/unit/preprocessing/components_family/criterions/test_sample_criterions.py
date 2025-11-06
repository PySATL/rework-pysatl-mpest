import numpy as np
import pytest
from rework_pysatl_mpest.preprocessing.components_family.criterions.sample_criterions import (
    CBootKurt,
    CHillAbs,
    CIqr,
    CKurt,
    CKurtMoors,
    CLogRatio,
    CMaxZscore,
    CNegativeValue,
    COutlierFraction,
    CRange,
    CSkew,
    CSkewBowley,
    CSpacingGap,
    CSpacingGini,
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
            ([1, 2, 4, 3, 0], 2.66666),
        ],
    )
    def test_kurt_moors(self, hist, expected):
        error_rate = 1e-5
        result = CKurtMoors().score(hist)
        assert np.abs(expected - result) < error_rate
