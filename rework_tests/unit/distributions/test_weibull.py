"""Tests for Weibull class"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import random

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from rework_pysatl_mpest.distributions import Weibull
from scipy.integrate import quad
from scipy.special import gamma
from scipy.stats import kstest, weibull_min

# Strategies for generating valid Weibull parameters
st_shape = st.floats(min_value=0.5, max_value=10, allow_nan=False, allow_infinity=False)
st_loc = st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)
st_scale = st.floats(min_value=0.5, max_value=10, allow_nan=False, allow_infinity=False)


class TestWeibullInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self):
        """Tests that the instance is initialized correctly with valid parameters."""

        shape, loc, scale = 2.0, 0.5, 1.5
        dist = Weibull(shape=shape, loc=loc, scale=scale)
        assert isinstance(dist.shape, float)
        assert isinstance(dist.loc, float)
        assert isinstance(dist.scale, float)
        assert dist.shape == shape
        assert dist.loc == loc
        assert dist.scale == scale

    def test_name_property(self):
        """Tests that the name property returns the correct string."""

        dist = Weibull(shape=2.0, loc=0.0, scale=1.0)
        assert dist.name == "Weibull"

    def test_params_property(self):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Weibull(shape=2.0, loc=0.0, scale=1.0)
        assert dist.params == {"shape", "loc", "scale"}

    @pytest.mark.parametrize("invalid_shape", [0.0, -1.0, -10.0])
    def test_shape_invariant_violation(self, invalid_shape):
        """Tests that initializing with a non-positive shape raises a ValueError."""

        with pytest.raises(ValueError, match="Shape parameter must be positive"):
            Weibull(shape=invalid_shape, loc=0.0, scale=1.0)

    @pytest.mark.parametrize("invalid_scale", [0.0, -1.0, -10.0])
    def test_scale_invariant_violation(self, invalid_scale):
        """Tests that initializing with a non-positive scale raises a ValueError."""

        with pytest.raises(ValueError, match="Scale parameter must be positive"):
            Weibull(shape=1.0, loc=0.0, scale=invalid_scale)

    def test_repr_method(self):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Weibull(shape=1.23, loc=4.56, scale=7.89)
        repr_str = repr(dist)
        assert repr_str == "Weibull(shape=1.23, loc=4.56, scale=7.89)"

        recreated_dist = eval(repr_str)
        assert isinstance(recreated_dist, Weibull)
        assert recreated_dist.shape == dist.shape
        assert recreated_dist.loc == dist.loc
        assert recreated_dist.scale == dist.scale


class TestWeibullPDF:
    """Tests for the pdf method using hypothesis."""

    @given(
        shape=st_shape,
        loc=st_loc,
        scale=st_scale,
        x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)),
    )
    def test_pdf_properties(self, shape, loc, scale, x):
        """Tests that the PDF is non-negative and has the correct return type and shape."""

        dist = Weibull(shape=shape, loc=loc, scale=scale)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

    @given(shape=st_shape, loc=st_loc, scale=st_scale, x=st.floats(1e-6, 1e6))
    def test_pdf_against_scipy(self, shape, loc, scale, x):
        """Compares the custom PDF implementation against scipy's implementation."""

        dist = Weibull(shape=shape, loc=loc, scale=scale)
        custom_pdf = dist.pdf(x)
        scipy_pdf = weibull_min.pdf(x, c=shape, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_pdf, scipy_pdf, atol=1e-9)

    @given(shape=st_shape, loc=st_loc, scale=st_scale)
    def test_pdf_integral_is_one(self, shape, loc, scale):
        """Tests that the integral of the PDF over its support is equal to 1."""

        dist = Weibull(shape=shape, loc=loc, scale=scale)
        integral, error = quad(dist.pdf, loc, np.inf)
        np.testing.assert_allclose(1.0, integral, atol=1e-6)

    @given(shape=st_shape, loc=st_loc, scale=st_scale, x=st.floats(max_value=-1e6, allow_infinity=False))
    def test_pdf_outside_support(self, shape, loc, scale, x):
        """Tests that the PDF is zero for values less than the location parameter."""

        x_val = loc - abs(x) if x != 0 else loc - 1
        dist = Weibull(shape=shape, loc=loc, scale=scale)
        assert dist.pdf(x_val) == 0.0


class TestWeibullLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @given(
        shape=st_shape,
        loc=st_loc,
        scale=st_scale,
        x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)),
    )
    def test_lpdf_return_type_and_shape(self, shape, loc, scale, x):
        """Tests the return type and shape of the lpdf method."""

        dist = Weibull(shape=shape, loc=loc, scale=scale)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.shape == x.shape

    @given(shape=st_shape, loc=st_loc, scale=st_scale, x=st.floats(1e-6, 1e6))
    def test_lpdf_against_scipy(self, shape, loc, scale, x):
        """Compares the custom LPDF implementation against scipy's implementation."""
        assume(x > loc)

        dist = Weibull(shape=shape, loc=loc, scale=scale)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = weibull_min.logpdf(x, c=shape, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-9)

    @given(shape=st_shape, loc=st_loc, scale=st_scale, x=st.floats(max_value=-1e6, allow_infinity=False))
    def test_lpdf_outside_support(self, shape, loc, scale, x):
        """Tests that the LPDF is -inf for values less than or equal to the location."""

        x_val = loc - abs(x) if x != 0 else loc
        dist = Weibull(shape=shape, loc=loc, scale=scale)
        assert dist.lpdf(x_val) == -np.inf


class TestWeibullPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @given(
        shape=st_shape,
        loc=st_loc,
        scale=st_scale,
        p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True)),
    )
    def test_ppf_return_type_and_shape(self, shape, loc, scale, p):
        """Tests the return type and shape of the ppf method."""

        dist = Weibull(shape=shape, loc=loc, scale=scale)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.shape == p.shape

    @given(shape=st_shape, loc=st_loc, scale=st_scale, p=st.floats(0.01, 1))
    def test_ppf_against_scipy(self, shape, loc, scale, p):
        """Compares the custom PPF implementation against scipy's implementation."""

        dist = Weibull(shape=shape, loc=loc, scale=scale)
        custom_ppf = dist.ppf(p)
        scipy_ppf = weibull_min.ppf(p, c=shape, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-6)


class TestWeibullGradients:
    """Tests for gradient calculation methods."""

    h = 1e-9

    @given(
        shape=st_shape,
        loc=st_loc,
        scale=st_scale,
        x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)),
    )
    def test_dlog_shape_numerical(self, shape, loc, scale, x):
        """Checks the analytical gradient for 'shape' against a numerical approximation."""

        assume(np.all(x > loc + self.h))
        dist = Weibull(shape=shape, loc=loc, scale=scale)
        lpdf_plus_h = Weibull(shape=shape + self.h, loc=loc, scale=scale).lpdf(x)
        lpdf_minus_h = Weibull(shape=shape - self.h, loc=loc, scale=scale).lpdf(x)
        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        analytical_grad = dist._dlog_shape(x)
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(
        shape=st_shape,
        loc=st_loc,
        scale=st_scale,
        x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)),
    )
    def test_dlog_loc_numerical(self, shape, loc, scale, x):
        """Checks the analytical gradient for 'loc' against a numerical approximation."""

        assume(np.all(x > (loc + self.h)))
        dist = Weibull(shape=shape, loc=loc, scale=scale)
        lpdf_plus_h = Weibull(shape=shape, loc=loc + self.h, scale=scale).lpdf(x)
        lpdf_minus_h = Weibull(shape=shape, loc=loc - self.h, scale=scale).lpdf(x)
        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        analytical_grad = dist._dlog_loc(x)
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(
        shape=st_shape,
        loc=st_loc,
        scale=st_scale,
        x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)),
    )
    def test_dlog_scale_numerical(self, shape, loc, scale, x):
        """Checks the analytical gradient for 'scale' against a numerical approximation."""

        assume(np.all(x > loc + self.h))
        dist = Weibull(shape=shape, loc=loc, scale=scale)
        lpdf_plus_h = Weibull(shape=shape, loc=loc, scale=scale + self.h).lpdf(x)
        lpdf_minus_h = Weibull(shape=shape, loc=loc, scale=scale - self.h).lpdf(x)
        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        analytical_grad = dist._dlog_scale(x)
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "fixed_params, expected_cols, expected_params",
        [
            ([], 3, ["loc", "scale", "shape"]),
            (["loc"], 2, ["scale", "shape"]),
            (["scale"], 2, ["loc", "shape"]),
            (["shape"], 2, ["loc", "scale"]),
            (["loc", "scale"], 1, ["shape"]),
            (["loc", "shape"], 1, ["scale"]),
            (["scale", "shape"], 1, ["loc"]),
            (["loc", "scale", "shape"], 0, []),
        ],
    )
    def test_log_gradients_structure(self, fixed_params, expected_cols, expected_params):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Weibull(shape=2.0, loc=1.0, scale=3.0)
        for param in fixed_params:
            dist.fix_param(param)

        x = np.array([1.5, 2.0, 3.0])
        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == (len(x), expected_cols)

        sorted_params = sorted(expected_params)
        if "loc" in expected_params:
            np.testing.assert_allclose(gradients[:, sorted_params.index("loc")], dist._dlog_loc(x))
        if "scale" in expected_params:
            np.testing.assert_allclose(gradients[:, sorted_params.index("scale")], dist._dlog_scale(x))
        if "shape" in expected_params:
            np.testing.assert_allclose(gradients[:, sorted_params.index("shape")], dist._dlog_shape(x))


class TestWeibullGenerate:
    """Tests for the generate method."""

    def test_generate_type_and_shape(self):
        """Tests that generated samples have the correct type and shape."""

        dist = Weibull(shape=2.0, loc=0.0, scale=1.0)
        size = 100
        samples = dist.generate(size=size)
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float64
        assert samples.shape == (size,)

    def test_generate_zero_size(self):
        """Tests if generating 0 number of samples returns an empty array"""

        dist = Weibull(shape=2.0, loc=0.0, scale=1.0)
        assert len(dist.generate(size=0)) == 0

    @pytest.mark.parametrize("size", [-1, -10])
    def test_generate_negative_size(self, size):
        """Tests that generating a negative number of samples raises ValueError."""

        dist = Weibull(shape=2.0, loc=0.0, scale=1.0)
        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        np.random.seed(123)
        random.seed(123)
        shape, loc, scale = 2.5, 5.0, 3.0
        dist = Weibull(shape=shape, loc=loc, scale=scale)
        size = 50000

        samples = dist.generate(size=size)

        # Theoretical mean and variance for 3-parameter Weibull
        g1 = gamma(1 + 1 / shape)
        g2 = gamma(1 + 2 / shape)
        theoretical_mean = loc + scale * g1
        theoretical_var = (scale**2) * (g2 - g1**2)

        assert np.mean(samples) == pytest.approx(theoretical_mean, rel=0.05)
        assert np.var(samples) == pytest.approx(theoretical_var, rel=0.05)

    def test_generate_kolmogorov_smirnov(self):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        shape, loc, scale = 3.0, 10.0, 2.0
        dist = Weibull(shape=shape, loc=loc, scale=scale)
        size = 1000
        expected_p_value = 0.05

        samples = dist.generate(size=size)

        # args for scipy's weibull_min are (shape, loc, scale)
        ks_statistic, p_value = kstest(samples, "weibull_min", args=(shape, loc, scale))
        assert p_value > expected_p_value
