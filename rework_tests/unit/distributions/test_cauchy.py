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
from rework_pysatl_mpest.distributions import Cauchy
from scipy.integrate import quad
from scipy.stats import cauchy, kstest

st_scale = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
st_loc = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


class TestCauchyInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self):
        """Tests that the instance is initialized correctly with valid parameters."""

        loc, scale = 0.5, 2.0
        dist = Cauchy(loc=loc, scale=scale)
        assert isinstance(dist.loc, float)
        assert isinstance(dist.scale, float)
        assert dist.loc == loc
        assert dist.scale == scale

    def test_name_property(self):
        """Tests that the name property returns the correct string."""

        dist = Cauchy(loc=0.0, scale=1.0)
        assert dist.name == "Cauchy"

    def test_params_property(self):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Cauchy(loc=0.0, scale=1.0)
        assert dist.params == {"loc", "scale"}

    def test_scale_invariant_violation(self):
        """Tests that assigning a non-positive rate after initialization raises a ValueError."""

        with pytest.raises(ValueError, match="Scale parameter should be positive"):
            Cauchy(loc=0.0, scale=-10.0)
        with pytest.raises(ValueError, match="Scale parameter should be positive"):
            Cauchy(loc=0.0, scale=-0.02)

    def test_scale_assignment_violation(self):
        """Tests that assigning a non-positive rate after initialization raises a ValueError."""

        dist = Cauchy(loc=0.0, scale=1.0)
        with pytest.raises(ValueError, match="Scale parameter should be positive"):
            dist.scale = 0.0
        with pytest.raises(ValueError, match="Scale parameter should be positive"):
            dist.scale = -10.0

    def test_repr_method(self):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Cauchy(loc=1.23, scale=4.56)
        repr_str = repr(dist)
        assert repr_str == "Cauchy(loc=1.23, scale=4.56)"

        recreated_dist = eval(repr_str)
        assert isinstance(recreated_dist, Cauchy)
        assert recreated_dist.loc == dist.loc
        assert recreated_dist.scale == dist.scale

        assert dist == recreated_dist


class TestCauchyPDF:
    """Tests for the pdf method using hypothesis."""

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_pdf_properties(self, loc, scale, x):
        """Tests that the PDF is non-negative and has the correct return type and shape."""

        dist = Cauchy(loc=loc, scale=scale)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

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
        integral, error = quad(dist.pdf, loc - 186_124 * scale, loc + 186_124 * scale)
        print(f"integral = {integral}, loc = {loc}, scale = {scale}")
        np.testing.assert_allclose(1.0, integral, rtol=1e-5)


class TestLogCauchyPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_return_type_and_shape(self, loc, scale, x):
        """Tests the return type and shape of the lpdf method."""

        dist = Cauchy(loc=loc, scale=scale)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.shape == x.shape

    @given(loc=st_loc, scale=st_scale, x=st.floats(1e-6, 1e6))
    def test_lpdf_against_scipy(self, loc, scale, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        dist = Cauchy(loc=loc, scale=scale)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = cauchy.logpdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-9)


class TestCauchyPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @given(
        loc=st_loc, scale=st_scale, p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True))
    )
    def test_ppf_return_type_and_shape(self, loc, scale, p):
        """Tests the return type and shape of the ppf method."""

        dist = Cauchy(loc=loc, scale=scale)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.shape == p.shape

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


class TestCauchyGradients:
    """Tests for gradient calculation methods."""

    h = 1e-6

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)))
    def test_dlog_loc_numerical(self, loc, scale, x):
        """Checks the analytical gradient for 'loc' against a numerical approximation."""

        assume(np.all(x > (loc + self.h)))

        dist = Cauchy(loc=loc, scale=scale)

        lpdf_plus_h = Cauchy(loc=loc + self.h, scale=scale).lpdf(x)
        lpdf_minus_h = Cauchy(loc=loc - self.h, scale=scale).lpdf(x)

        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        analytical_grad = dist._dlog_loc(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.shape == x.shape
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)))
    def test_dlog_scale_numerical(self, loc, scale, x):
        """Checks the analytical gradient for 'scale' against a numerical approximation."""

        assume(np.all(x > (loc + self.h)))

        dist = Cauchy(loc=loc, scale=scale)

        lpdf_plus_h = Cauchy(loc=loc, scale=scale + self.h).lpdf(x)
        lpdf_minus_h = Cauchy(loc=loc, scale=scale - self.h).lpdf(x)

        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        analytical_grad = dist._dlog_scale(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.shape == x.shape
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "fixed_params, expected_shape_col, expected_params",
        [([], 2, ["loc", "scale"]), (["loc"], 1, ["scale"]), (["scale"], 1, ["loc"]), (["loc", "scale"], 0, [])],
    )
    def test_log_gradients_structure(self, fixed_params, expected_shape_col, expected_params):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Cauchy(loc=1.0, scale=2.0)
        for param in fixed_params:
            dist.fix_param(param)

        x = np.array([1.5, 2.0, 3.0])
        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == (len(x), expected_shape_col)

        if "loc" in expected_params:
            idx = sorted(expected_params).index("loc")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_loc(x))
        if "scale" in expected_params:
            idx = sorted(expected_params).index("scale")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_scale(x))


class TestCauchyGenerate:
    """Tests for the generate method."""

    def test_generate_type_and_shape(self):
        """Tests that generated samples have the correct type and shape."""

        np.random.seed(42)
        random.seed(42)
        dist = Cauchy(loc=0.0, scale=2.0)
        size = 100
        samples = dist.generate(size=size)
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float64
        assert samples.shape == (size,)

    def test_generate_zero_size(self):
        """Tests if the generating 0 number of samples returns an empty array"""

        dist = Cauchy(loc=0.0, scale=1.0)
        assert len(dist.generate(size=0)) == 0

    @pytest.mark.parametrize("size", [-1, -10])
    def test_generate_negative_size(self, size):
        """Tests that generating a negative number of samples raises ValueError."""

        dist = Cauchy(loc=0.0, scale=1.0)

        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_kolmogorov_smirnov(self):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        loc, scale = 10.0, 2.0
        dist = Cauchy(loc=loc, scale=scale)
        size = 1000

        samples = dist.generate(size=size)

        ks_statistic, p_value = kstest(samples, "cauchy", args=(loc, scale))
        lower_bound = 0.05
        assert p_value > lower_bound
