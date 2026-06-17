"""Tests for cauchy class"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import random

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pysatl_mpest.distributions import Cauchy
from scipy.integrate import quad
from scipy.stats import cauchy, kstest

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]

st_scale = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
st_loc = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestCauchyInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self, dtype):
        """Tests that the instance is initialized correctly with valid parameters."""

        loc, scale = 0.5, 2.0
        dist = Cauchy(loc=loc, scale=scale, dtype=dtype)
        assert dist.loc.dtype == dtype
        assert dist.scale.dtype == dtype
        assert dist.loc == dtype(loc)
        assert dist.scale == dtype(scale)

    def test_name_property(self, dtype):
        """Tests that the name property returns the correct string."""

        dist = Cauchy(loc=0.0, scale=1.0, dtype=dtype)
        assert dist.name == "Cauchy"

    def test_params_property(self, dtype):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Cauchy(loc=0.0, scale=1.0, dtype=dtype)
        assert dist.params == {"loc", "scale"}

    def test_scale_invariant_violation(self, dtype):
        """Tests that assigning a non-positive rate after initialization raises a ValueError."""

        with pytest.raises(ValueError, match="Scale parameter should be positive"):
            Cauchy(loc=0.0, scale=-10.0, dtype=dtype)
        with pytest.raises(ValueError, match="Scale parameter should be positive"):
            Cauchy(loc=0.0, scale=-0.02, dtype=dtype)

    def test_scale_assignment_violation(self, dtype):
        """Tests that assigning a non-positive rate after initialization raises a ValueError."""

        dist = Cauchy(loc=0.0, scale=1.0, dtype=dtype)
        with pytest.raises(ValueError, match="Scale parameter should be positive"):
            dist.scale = 0.0
        with pytest.raises(ValueError, match="Scale parameter should be positive"):
            dist.scale = -10.0

    def test_repr_method(self, dtype):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Cauchy(loc=1.23, scale=4.56, dtype=dtype)
        repr_str = repr(dist)
        assert repr_str == f"Cauchy(loc={dist.loc}, scale={dist.scale}, dtype=np.{dtype.__name__})"

        recreated_dist = eval(repr_str)
        assert dist == recreated_dist


class TestCauchyPDF:
    """Tests for the pdf method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_pdf_properties_for_array_input(self, loc, scale, x, dtype):
        """Tests that for an array input, the PDF returns a non-negative array with the correct type and shape."""

        dist = Cauchy(loc=loc, scale=scale, dtype=dtype)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.dtype == dtype
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, scale=st_scale, x=st.floats(-1e6, 1e6))
    def test_pdf_properties_for_scalar_input(self, loc, scale, x, dtype):
        """Tests that for a scalar input, the PDF returns a non-negative scalar with the correct type."""

        dist = Cauchy(loc, scale, dtype=dtype)
        pdf_value = dist.pdf(x)
        assert np.isscalar(pdf_value)
        assert isinstance(pdf_value, dtype)
        assert pdf_value >= 0

    @given(loc=st_loc, scale=st_scale, x=st.floats(1e-6, 1e6))
    def test_pdf_against_scipy(self, loc, scale, x):
        """Compares the custom PDF implementation against scipy's implementation."""

        dist = Cauchy(loc=loc, scale=scale)
        custom_pdf = dist.pdf(x)
        scipy_pdf = cauchy.pdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_pdf, scipy_pdf, atol=1e-7)

    @given(loc=st_loc, scale=st_scale)
    def test_pdf_integral_is_one(self, loc, scale):
        """Tests that the integral of the PDF over its support is equal to 1."""

        dist = Cauchy(loc=loc, scale=scale)
        integral, error = quad(lambda x: dist.pdf(x).item(), loc - 186_124 * scale, loc + 186_124 * scale)
        print(f"integral = {integral}, loc = {loc}, scale = {scale}")
        np.testing.assert_allclose(1.0, integral, rtol=1e-5)


class TestCauchyLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_return_type_and_shape_for_array_input(self, loc, scale, x, dtype):
        """Tests the return type and shape of the lpdf method for array input."""

        dist = Cauchy(loc=loc, scale=scale, dtype=dtype)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.dtype == dtype
        assert lpdf_values.shape == x.shape

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, scale=st_scale, x=st.floats(-1e6, 1e6))
    def test_lpdf_return_type_and_shape_for_scalar_input(self, loc, scale, x, dtype):
        """Tests the return type and shape of the lpdf method for scalar input."""

        dist = Cauchy(loc=loc, scale=scale, dtype=dtype)
        lpdf_value = dist.lpdf(x)
        assert np.isscalar(lpdf_value)
        assert isinstance(lpdf_value, dtype)

    @given(loc=st_loc, scale=st_scale, x=st.floats(1e-6, 1e6))
    def test_lpdf_against_scipy(self, loc, scale, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        dist = Cauchy(loc=loc, scale=scale)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = cauchy.logpdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-9)


class TestCauchyPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(
        loc=st_loc, scale=st_scale, p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True))
    )
    def test_ppf_return_type_and_shape_for_array_input(self, loc, scale, p, dtype):
        """Tests the return type and shape of the ppf method for array input."""

        dist = Cauchy(loc=loc, scale=scale, dtype=dtype)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.dtype == dtype
        assert ppf_values.shape == p.shape

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, scale=st_scale, p=st.floats(0, 1, exclude_max=True))
    def test_ppf_return_type_and_shape_for_scalar_input(self, loc, scale, p, dtype):
        """Tests the return type and shape of the ppf method for scalar input."""

        dist = Cauchy(loc=loc, scale=scale, dtype=dtype)
        ppf_value = dist.ppf(p)
        assert np.isscalar(ppf_value)
        assert isinstance(ppf_value, dtype)

    @given(loc=st_loc, scale=st_scale, p=st.floats(0, 1, exclude_max=True, exclude_min=True))
    def test_ppf_against_scipy(self, loc, scale, p):
        """Compares the custom PPF implementation against scipy's implementation."""
        p = round(p, 5)
        dist = Cauchy(loc=loc, scale=scale)
        custom_ppf = dist.ppf(p)
        scipy_ppf = cauchy.ppf(p, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-5)

    @pytest.mark.parametrize("p_val", [-0.5, 1.1, 1.5])
    def test_ppf_invalid_input(self, p_val):
        """Tests that PPF returns NaN for probabilities outside the [0, 1) range."""

        dist = Cauchy(loc=0.0, scale=1.0)
        assert np.isnan(dist.ppf(p_val))


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestCauchyGradients:
    """Tests for gradient calculation methods."""

    h = 1e-6

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)))
    def test_dlog_loc_numerical_for_array_input(self, loc, scale, x, dtype):
        """Checks the analytical gradient for 'loc' against a numerical approximation for array input."""

        assume(np.all(x > (loc + self.h)))

        dist = Cauchy(loc, scale, dtype=dtype)
        analytical_grad = dist._dlog_loc(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Cauchy(loc + self.h, scale).lpdf(x)
            lpdf_minus_h = Cauchy(loc - self.h, scale).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(loc=st_loc, scale=st_scale, x=st.floats(1e-3, 1e3))
    def test_dlog_loc_for_scalar_input(self, loc, scale, x, dtype):
        """Checks that the gradient for 'loc' for a scalar input returns a scalar."""

        assume(x > loc + self.h)

        dist = Cauchy(loc, scale, dtype=dtype)
        analytical_grad = dist._dlog_loc(x)

        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)))
    def test_dlog_scale_numerical_for_array_input(self, loc, scale, x, dtype):
        """Checks the analytical gradient for 'scale' against a numerical approximation for array input."""

        assume(np.all(x > (loc + self.h)))

        dist = Cauchy(loc, scale, dtype=dtype)
        analytical_grad = dist._dlog_scale(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Cauchy(loc, scale + self.h).lpdf(x)
            lpdf_minus_h = Cauchy(loc, scale - self.h).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3)

    @given(loc=st_loc, scale=st_scale, x=st.floats(1e-3, 1e3))
    def test_dlog_scale_for_scalar_input(self, loc, scale, x, dtype):
        """Checks that the gradient for 'scale' for a scalar input returns a scalar."""

        assume(x > loc + self.h)

        dist = Cauchy(loc, scale, dtype=dtype)
        analytical_grad = dist._dlog_scale(x)

        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @pytest.mark.parametrize(
        "fixed_params, expected_shape_col, expected_params",
        [([], 2, ["loc", "scale"]), (["loc"], 1, ["scale"]), (["scale"], 1, ["loc"]), (["loc", "scale"], 0, [])],
    )
    def test_log_gradients_structure(self, fixed_params, expected_shape_col, expected_params, dtype):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Cauchy(loc=1.0, scale=2.0, dtype=dtype)
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
        if "scale" in expected_params:
            idx = sorted(expected_params).index("scale")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_scale(x))

    @given(loc=st_loc, scale=st_scale, x=st.floats(1e-3, 1e3))
    def test_log_gradients_for_scalar_input(self, loc, scale, x, dtype):
        """Checks that the log_gradients for a scalar input returns a 1D-array."""

        dist = Cauchy(loc, scale, dtype=dtype)
        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.dtype == dtype
        assert gradients.ndim == 1


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestCauchyGenerate:
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
        dist = Cauchy(loc=0.0, scale=2.0, dtype=dtype)
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

        dist = Cauchy(loc=0.0, scale=1.0, dtype=dtype)

        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self, dtype):
        """Tests if the generated samples have the correct statistical properties (median)."""

        np.random.seed(123)
        random.seed(123)
        loc, scale = 5.0, 2.0
        dist = Cauchy(loc=loc, scale=scale, dtype=dtype)
        size = 5000

        samples = dist.generate(size=size)

        # The mean and variance of the Cauchy distribution are undefined.
        # The median is equal to the location parameter 'loc'.
        theoretical_median = loc

        assert np.median(samples) == pytest.approx(theoretical_median, rel=0.1)

    def test_generate_kolmogorov_smirnov(self, dtype):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        loc, scale = 10.0, 2.0
        dist = Cauchy(loc=loc, scale=scale, dtype=dtype)
        size = 1000

        samples = dist.generate(size=size)

        ks_statistic, p_value = kstest(samples, "cauchy", args=(loc, scale))
        lower_bound = 0.05
        assert p_value > lower_bound
