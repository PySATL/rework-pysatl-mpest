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
from scipy.integrate import quad
from scipy.stats import expon, kstest

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]

st_rate = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
st_loc = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestExponentialInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self, dtype):
        """Tests that the instance is initialized correctly with valid parameters."""

        loc, rate = 0.5, 2.0
        dist = Exponential(loc=loc, rate=rate, dtype=dtype)
        assert dist.loc.dtype == dtype
        assert dist.rate.dtype == dtype
        assert dist.loc == dtype(loc)
        assert dist.rate == dtype(rate)

    def test_name_property(self, dtype):
        """Tests that the name property returns the correct string."""

        dist = Exponential(loc=0.0, rate=1.0, dtype=dtype)
        assert dist.name == "Exponential"

    def test_params_property(self, dtype):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Exponential(loc=0.0, rate=1.0, dtype=dtype)
        assert dist.params == {"loc", "rate"}

    def test_rate_invariant_violation(self, dtype):
        """Tests that initializing with a non-positive rate raises a ValueError."""

        with pytest.raises(ValueError, match="Rate parameter must be a positive"):
            Exponential(loc=0.0, rate=0.0, dtype=dtype)
        with pytest.raises(ValueError, match="Rate parameter must be a positive"):
            Exponential(loc=0.0, rate=-1.0, dtype=dtype)

    def test_rate_assignment_violation(self, dtype):
        """Tests that assigning a non-positive rate after initialization raises a ValueError."""

        dist = Exponential(loc=0.0, rate=1.0, dtype=dtype)
        with pytest.raises(ValueError, match="Rate parameter must be a positive"):
            dist.rate = 0.0
        with pytest.raises(ValueError, match="Rate parameter must be a positive"):
            dist.rate = -10.0

    def test_repr_method(self, dtype):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Exponential(loc=1.23, rate=4.56, dtype=dtype)
        repr_str = repr(dist)
        assert repr_str == f"Exponential(loc={dist.loc}, rate={dist.rate}, dtype=np.{dtype.__name__})"

        recreated_dist = eval(repr_str)
        assert recreated_dist == dist


class TestExponentialPDF:
    """Tests for the pdf method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, rate=st_rate, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_pdf_properties_for_array_input(self, loc, rate, x, dtype):
        """Tests that for an array input, the PDF returns a non-negative array with the correct type and shape."""

        dist = Exponential(loc=loc, rate=rate, dtype=dtype)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.dtype == dtype
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, rate=st_rate, x=st.floats(-1e6, 1e6))
    def test_pdf_properties_for_scalar_input(self, loc, rate, x, dtype):
        """Tests that for a scalar input, the PDF returns a non-negative scalar with the correct type."""

        dist = Exponential(loc, rate, dtype=dtype)
        pdf_value = dist.pdf(x)
        assert np.isscalar(pdf_value)
        assert isinstance(pdf_value, dtype)
        assert pdf_value >= 0

    @given(loc=st_loc, rate=st_rate, x=st.floats(1e-6, 1e6))
    def test_pdf_against_scipy(self, loc, rate, x):
        """Compares the custom PDF implementation against scipy's implementation."""

        assume(x > loc)

        dist = Exponential(loc=loc, rate=rate)
        custom_pdf = dist.pdf(x)
        scipy_pdf = expon.pdf(x, loc=loc, scale=1 / rate)
        np.testing.assert_allclose(custom_pdf, scipy_pdf, atol=1e-9)

    @given(loc=st_loc, rate=st_rate)
    def test_pdf_integral_is_one(self, loc, rate):
        """Tests that the integral of the PDF over its support is equal to 1."""

        dist = Exponential(loc=loc, rate=rate)
        integral, error = quad(lambda x: dist.pdf(x).item(), loc, np.inf)
        np.testing.assert_allclose(1.0, integral)

    @given(loc=st_loc, rate=st_rate, x=st.floats(max_value=-1e6, allow_infinity=False))
    def test_pdf_outside_support(self, loc, rate, x):
        """Tests that the PDF is zero for values less than the location parameter."""

        x_val = loc - abs(x)
        dist = Exponential(loc=loc, rate=rate)
        assert dist.pdf(x_val) == 0.0


class TestExponentialLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, rate=st_rate, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_return_type_and_shape_for_array_input(self, loc, rate, x, dtype):
        """Tests the return type and shape of the lpdf method for array input."""

        dist = Exponential(loc=loc, rate=rate, dtype=dtype)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.dtype == dtype
        assert lpdf_values.shape == x.shape

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, rate=st_rate, x=st.floats(-1e6, 1e6))
    def test_lpdf_return_type_and_shape_for_scalar_input(self, loc, rate, x, dtype):
        """Tests the return type and shape of the lpdf method for scalar input."""

        dist = Exponential(loc, rate, dtype=dtype)
        lpdf_value = dist.lpdf(x)
        assert np.isscalar(lpdf_value)
        assert isinstance(lpdf_value, dtype)

    @given(loc=st_loc, rate=st_rate, x=st.floats(1e-6, 1e6))
    def test_lpdf_against_scipy(self, loc, rate, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        assume(x > loc)

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

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(
        loc=st_loc, rate=st_rate, p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True))
    )
    def test_ppf_return_type_and_shape_for_array_input(self, loc, rate, p, dtype):
        """Tests the return type and shape of the ppf method for array input."""

        dist = Exponential(loc=loc, rate=rate, dtype=dtype)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.dtype == dtype
        assert ppf_values.shape == p.shape

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, rate=st_rate, p=st.floats(0, 1, exclude_max=True))
    def test_ppf_return_type_and_shape_for_scalar_input(self, loc, rate, p, dtype):
        """Tests the return type and shape of the ppf method for scalar input."""

        dist = Exponential(loc=loc, rate=rate, dtype=dtype)
        ppf_value = dist.ppf(p)
        assert np.isscalar(ppf_value)
        assert isinstance(ppf_value, dtype)

    @given(loc=st_loc, rate=st_rate, p=st.floats(0, 1, exclude_max=True, exclude_min=True))
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


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestExponentialGradients:
    """Tests for gradient calculation methods."""

    h = 1e-6

    @given(loc=st_loc, rate=st_rate, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)))
    def test_dlog_loc_numerical_for_array_input(self, loc, rate, x, dtype):
        """Checks the analytical gradient for 'loc' against a numerical approximation for array input."""

        assume(np.all(x > (loc + self.h)))

        dist = Exponential(loc, rate, dtype=dtype)
        analytical_grad = dist._dlog_loc(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Exponential(loc + self.h, rate).lpdf(x)
            lpdf_minus_h = Exponential(loc - self.h, rate).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(loc=st_loc, rate=st_rate, x=st.floats(1e-3, 1e3))
    def test_dlog_loc_for_scalar_input(self, loc, rate, x, dtype):
        """Checks that the gradient for 'loc' for a scalar input returns a scalar."""

        assume(x > (loc + self.h))

        dist = Exponential(loc, rate, dtype=dtype)
        analytical_grad = dist._dlog_loc(x)
        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @given(loc=st_loc, rate=st_rate, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)))
    def test_dlog_rate_numerical_for_array_input(self, loc, rate, x, dtype):
        """Checks the analytical gradient for 'rate' against a numerical approximation for array input."""

        assume(np.all(x > (loc + self.h)))

        dist = Exponential(loc, rate, dtype=dtype)
        analytical_grad = dist._dlog_rate(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Exponential(loc, rate + self.h).lpdf(x)
            lpdf_minus_h = Exponential(loc, rate - self.h).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3)

    @given(loc=st_loc, rate=st_rate, x=st.floats(1e-3, 1e3))
    def test_dlog_rate_for_scalar_input(self, loc, rate, x, dtype):
        """Checks that the gradient for 'rate' for a scalar input returns a scalar."""

        assume(x > (loc + self.h))

        dist = Exponential(loc, rate, dtype=dtype)
        analytical_grad = dist._dlog_rate(x)
        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @pytest.mark.parametrize(
        "fixed_params, expected_shape_col, expected_params",
        [([], 2, ["loc", "rate"]), (["loc"], 1, ["rate"]), (["rate"], 1, ["loc"]), (["loc", "rate"], 0, [])],
    )
    def test_log_gradients_structure(self, fixed_params, expected_shape_col, expected_params, dtype):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Exponential(loc=1.0, rate=2.0, dtype=dtype)
        for param in fixed_params:
            dist.fix_param(param)

        x = np.array([1.5, 2.0, 3.0])
        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.dtype == dtype
        assert gradients.shape == (len(x), expected_shape_col)

        if "loc" in expected_params:
            idx = sorted(expected_params).index("loc")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_loc(x))
        if "rate" in expected_params:
            idx = sorted(expected_params).index("rate")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_rate(x))

    @given(loc=st_loc, rate=st_rate, x=st.floats(1e-3, 1e3))
    def test_log_gradients_for_scalar_input(self, loc, rate, x, dtype):
        """Checks that the log_gradients for a scalar input returns a 1D-array."""

        dist = Exponential(loc, rate, dtype=dtype)
        gradients = dist.log_gradients(x)
        assert isinstance(gradients, np.ndarray)
        assert gradients.dtype == dtype
        assert gradients.ndim == 1


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestExponentialGenerate:
    """Tests for the generate method."""

    @pytest.mark.parametrize(
        "size, expected_shape, is_scalar",
        [
            (None, (), True),
            (0, (0,), False),
            (10, (10,), False),
            ((5,), (5,), False),
            ((2, 3), (2, 3), False),
        ],
    )
    def test_generate_type_and_shape(self, dtype, size, expected_shape, is_scalar):
        """Tests that generated samples have the correct type and shape."""

        np.random.seed(42)
        random.seed(42)
        dist = Exponential(loc=0.0, rate=2.0, dtype=dtype)
        samples = dist.generate(size=size)

        if is_scalar:
            assert np.isscalar(samples)
            assert isinstance(samples, dtype)
        else:
            assert isinstance(samples, np.ndarray)
            assert samples.shape == expected_shape
            assert samples.dtype == dtype

    @pytest.mark.parametrize("size", [-1, -10])
    def test_generate_negative_size(self, size, dtype):
        """Tests that generating a negative number of samples raises ValueError."""

        dist = Exponential(loc=0.0, rate=1.0, dtype=dtype)

        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self, dtype):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        np.random.seed(123)
        random.seed(123)
        loc, rate = 5.0, 0.5
        dist = Exponential(loc=loc, rate=rate, dtype=dtype)
        size = 20000

        samples = dist.generate(size=size)

        theoretical_mean = loc + 1 / rate
        theoretical_var = (1 / rate) ** 2

        assert np.mean(samples, dtype=np.float64) == pytest.approx(theoretical_mean, rel=0.1)
        assert np.var(samples, dtype=np.float64) == pytest.approx(theoretical_var, rel=0.1)

    def test_generate_kolmogorov_smirnov(self, dtype):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        loc, rate = 10.0, 2.0
        dist = Exponential(loc=loc, rate=rate, dtype=dtype)
        size = 1000

        samples = dist.generate(size=size)

        ks_statistic, p_value = kstest(samples, "expon", args=(loc, 1 / rate))
        lower_bound = 0.05
        assert p_value > lower_bound
