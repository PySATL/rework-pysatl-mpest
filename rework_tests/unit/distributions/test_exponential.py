"""Tests for Exponential class"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import random

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from rework_pysatl_mpest.distributions import Exponential
from rework_tests.unit.distributions.test_continuous_distribution import DTypeHandlingMixin
from scipy.integrate import quad
from scipy.stats import expon, kstest

st_rate = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
st_loc = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


class TestExponentialInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self):
        """Tests that the instance is initialized correctly with valid parameters."""

        loc, rate = 0.5, 2.0
        dist = Exponential(loc=loc, rate=rate)
        assert isinstance(dist.loc, float)
        assert isinstance(dist.rate, float)
        assert dist.loc == loc
        assert dist.rate == rate

    def test_name_property(self):
        """Tests that the name property returns the correct string."""

        dist = Exponential(loc=0.0, rate=1.0)
        assert dist.name == "Exponential"

    def test_params_property(self):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Exponential(loc=0.0, rate=1.0)
        assert dist.params == {"loc", "rate"}

    def test_rate_invariant_violation(self):
        """Tests that initializing with a non-positive rate raises a ValueError."""

        with pytest.raises(ValueError, match="Rate parameter must be a positive"):
            Exponential(loc=0.0, rate=0.0)
        with pytest.raises(ValueError, match="Rate parameter must be a positive"):
            Exponential(loc=0.0, rate=-1.0)

    def test_rate_assignment_violation(self):
        """Tests that assigning a non-positive rate after initialization raises a ValueError."""

        dist = Exponential(loc=0.0, rate=1.0)
        with pytest.raises(ValueError, match="Rate parameter must be a positive"):
            dist.rate = 0.0
        with pytest.raises(ValueError, match="Rate parameter must be a positive"):
            dist.rate = -10.0

    def test_repr_method(self):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Exponential(loc=1.23, rate=4.56)
        repr_str = repr(dist)
        assert repr_str == "Exponential(loc=1.23, rate=4.56)"

        recreated_dist = eval(repr_str)
        assert recreated_dist == dist


class TestExponentialPDF:
    """Tests for the pdf method using hypothesis."""

    @given(loc=st_loc, rate=st_rate, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_pdf_properties(self, loc, rate, x):
        """Tests that the PDF is non-negative and has the correct return type and shape."""

        dist = Exponential(loc=loc, rate=rate)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

    @given(loc=st_loc, rate=st_rate, x=st.floats(1e-6, 1e6))
    def test_pdf_against_scipy(self, loc, rate, x):
        """Compares the custom PDF implementation against scipy's implementation."""

        dist = Exponential(loc=loc, rate=rate)
        custom_pdf = dist.pdf(x)
        scipy_pdf = expon.pdf(x, loc=loc, scale=1 / rate)
        np.testing.assert_allclose(custom_pdf, scipy_pdf, atol=1e-9)

    @given(loc=st_loc, rate=st_rate)
    def test_pdf_integral_is_one(self, loc, rate):
        """Tests that the integral of the PDF over its support is equal to 1."""

        dist = Exponential(loc=loc, rate=rate)
        integral, error = quad(dist.pdf, loc, np.inf)
        np.testing.assert_allclose(1.0, integral)

    @given(loc=st_loc, rate=st_rate, x=st.floats(max_value=-1e6, allow_infinity=False))
    def test_pdf_outside_support(self, loc, rate, x):
        """Tests that the PDF is zero for values less than the location parameter."""

        x_val = loc - abs(x)
        dist = Exponential(loc=loc, rate=rate)
        assert dist.pdf(x_val) == 0.0


class TestExponentialLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @given(loc=st_loc, rate=st_rate, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_return_type_and_shape(self, loc, rate, x):
        """Tests the return type and shape of the lpdf method."""

        dist = Exponential(loc=loc, rate=rate)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.shape == x.shape

    @given(loc=st_loc, rate=st_rate, x=st.floats(1e-6, 1e6))
    def test_lpdf_against_scipy(self, loc, rate, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        dist = Exponential(loc=loc, rate=rate)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = expon.logpdf(x, loc=loc, scale=1 / rate)
        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-9)

    @given(loc=st_loc, rate=st_rate, x=st.floats(max_value=-1e6, allow_infinity=False))
    def test_lpdf_outside_support(self, loc, rate, x):
        """Tests that the LPDF is -inf for values less than the location parameter."""

        x_val = loc - abs(x)
        dist = Exponential(loc=loc, rate=rate)
        assert dist.lpdf(x_val) == -np.inf


class TestExponentialPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @given(
        loc=st_loc, rate=st_rate, p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True))
    )
    def test_ppf_return_type_and_shape(self, loc, rate, p):
        """Tests the return type and shape of the ppf method."""

        dist = Exponential(loc=loc, rate=rate)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.shape == p.shape

    @given(loc=st_loc, rate=st_rate, p=st.floats(0, 1))
    def test_ppf_against_scipy(self, loc, rate, p):
        """Compares the custom PPF implementation against scipy's implementation."""

        dist = Exponential(loc=loc, rate=rate)
        custom_ppf = dist.ppf(p)
        scipy_ppf = expon.ppf(p, loc=loc, scale=1 / rate)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-9)

    @pytest.mark.parametrize("p_val", [-0.5, 1.1, 1.5])
    def test_ppf_invalid_input(self, p_val):
        """Tests that PPF returns NaN for probabilities outside the [0, 1) range."""

        dist = Exponential(loc=0.0, rate=1.0)
        assert np.isnan(dist.ppf(p_val))


class TestExponentialGradients:
    """Tests for gradient calculation methods."""

    h = 1e-6

    @given(loc=st_loc, rate=st_rate, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)))
    def test_dlog_loc_numerical(self, loc, rate, x):
        """Checks the analytical gradient for 'loc' against a numerical approximation."""

        assume(np.all(x > (loc + self.h)))

        dist = Exponential(loc=loc, rate=rate)

        lpdf_plus_h = Exponential(loc=loc + self.h, rate=rate).lpdf(x)
        lpdf_minus_h = Exponential(loc=loc - self.h, rate=rate).lpdf(x)

        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        analytical_grad = dist._dlog_loc(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.shape == x.shape
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(loc=st_loc, rate=st_rate, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)))
    def test_dlog_rate_numerical(self, loc, rate, x):
        """Checks the analytical gradient for 'rate' against a numerical approximation."""

        assume(np.all(x > (loc + self.h)))

        dist = Exponential(loc=loc, rate=rate)

        lpdf_plus_h = Exponential(loc=loc, rate=rate + self.h).lpdf(x)
        lpdf_minus_h = Exponential(loc=loc, rate=rate - self.h).lpdf(x)

        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        analytical_grad = dist._dlog_rate(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.shape == x.shape
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "fixed_params, expected_shape_col, expected_params",
        [([], 2, ["loc", "rate"]), (["loc"], 1, ["rate"]), (["rate"], 1, ["loc"]), (["loc", "rate"], 0, [])],
    )
    def test_log_gradients_structure(self, fixed_params, expected_shape_col, expected_params):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Exponential(loc=1.0, rate=2.0)
        for param in fixed_params:
            dist.fix_param(param)

        x = np.array([1.5, 2.0, 3.0])
        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == (len(x), expected_shape_col)

        if "loc" in expected_params:
            idx = sorted(expected_params).index("loc")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_loc(x))
        if "rate" in expected_params:
            idx = sorted(expected_params).index("rate")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_rate(x))


class TestExponentialGenerate:
    """Tests for the generate method."""

    def test_generate_type_and_shape(self):
        """Tests that generated samples have the correct type and shape."""

        np.random.seed(42)
        random.seed(42)
        dist = Exponential(loc=0.0, rate=2.0)
        size = 100
        samples = dist.generate(size=size)
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float64
        assert samples.shape == (size,)

    def test_generate_zero_size(self):
        """Tests if the generating 0 number of samples returns an empty array"""

        dist = Exponential(loc=0.0, rate=1.0)
        assert len(dist.generate(size=0)) == 0

    @pytest.mark.parametrize("size", [-1, -10])
    def test_generate_negative_size(self, size):
        """Tests that generating a negative number of samples raises ValueError."""

        dist = Exponential(loc=0.0, rate=1.0)

        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        np.random.seed(123)
        random.seed(123)
        loc, rate = 5.0, 0.5
        dist = Exponential(loc=loc, rate=rate)
        size = 20000

        samples = dist.generate(size=size)

        theoretical_mean = loc + 1 / rate
        theoretical_var = (1 / rate) ** 2

        assert np.mean(samples) == pytest.approx(theoretical_mean, rel=0.1)
        assert np.var(samples) == pytest.approx(theoretical_var, rel=0.1)

    def test_generate_kolmogorov_smirnov(self):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        loc, rate = 10.0, 2.0
        dist = Exponential(loc=loc, rate=rate)
        size = 1000

        samples = dist.generate(size=size)

        ks_statistic, p_value = kstest(samples, "expon", args=(loc, 1 / rate))
        lower_bound = 0.05
        assert p_value > lower_bound


class TestExponentialDType(DTypeHandlingMixin):
    distribution_class = Exponential

    def __init__(self):
        self.default_params = {"loc": 0.0, "rate": 1.0}
