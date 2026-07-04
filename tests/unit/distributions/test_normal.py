"""Tests for Normal class"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pysatl_mpest.distributions import Normal
from scipy.integrate import quad
from scipy.stats import kstest, norm

# Strategies for hypothesis
st_loc = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
st_scale = st.floats(min_value=0.01, max_value=1e3, allow_nan=False, allow_infinity=False)


class TestNormalInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self):
        """Tests that the instance is initialized correctly with valid parameters."""

        loc, scale = 10.0, 2.5
        dist = Normal(loc=loc, scale=scale)
        assert isinstance(dist.loc, float)
        assert isinstance(dist.scale, float)
        assert dist.loc == float(loc)
        assert dist.scale == float(scale)

    def test_name_property(self):
        """Tests that the name property returns the correct string."""

        dist = Normal(loc=0.0, scale=1.0)
        assert dist.name == "Normal"

    def test_params_property(self):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Normal(loc=0.0, scale=1.0)
        assert dist.params == {"loc", "scale"}

    def test_scale_invariant_violation(self):
        """Tests that initializing with a non-positive scale raises a ValueError."""

        with pytest.raises(ValueError, match="Scale parameter must be positive"):
            Normal(loc=0.0, scale=0.0)
        with pytest.raises(ValueError, match="Scale parameter must be positive"):
            Normal(loc=0.0, scale=-1.0)

    def test_scale_assignment_violation(self):
        """Tests that assigning a non-positive rate after initialization raises a ValueError."""

        dist = Normal(loc=0.0, scale=1.0)
        with pytest.raises(ValueError, match="Scale parameter must be positive"):
            dist.scale = 0.0
        with pytest.raises(ValueError, match="Scale parameter must be positive"):
            dist.scale = -10.0

    def test_repr_method(self):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Normal(loc=1.23, scale=4.56)
        repr_str = repr(dist)
        assert repr_str == f"Normal(loc={dist.loc}, scale={dist.scale})"

        recreated_dist = eval(repr_str)
        assert dist == recreated_dist


class TestNormalPDF:
    """Tests for the pdf method using hypothesis."""

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_pdf_properties_for_array_input(self, loc, scale, x):
        """Tests that for an array input, the PDF returns a non-negative array with the correct type and shape."""

        dist = Normal(loc=loc, scale=scale)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.dtype == np.float64
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

    @given(loc=st_loc, scale=st_scale, x=st.floats(-1e6, 1e6))
    def test_pdf_properties_for_scalar_input(self, loc, scale, x):
        """Tests that for a scalar input, the PDF returns a non-negative scalar with the correct type."""

        dist = Normal(loc, scale)
        pdf_value = dist.pdf(x)
        assert np.isscalar(pdf_value)
        assert isinstance(pdf_value, np.float64)
        assert pdf_value >= 0

    @given(loc=st_loc, scale=st_scale, x=st.floats(-1e6, 1e6))
    def test_pdf_against_scipy(self, loc, scale, x):
        """Compares the custom PDF implementation against scipy's implementation."""

        dist = Normal(loc=loc, scale=scale)
        custom_pdf = dist.pdf(x)
        scipy_pdf = norm.pdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_pdf, scipy_pdf, atol=1e-9)

    @given(loc=st_loc, scale=st_scale)
    def test_pdf_integral_is_one(self, loc, scale):
        """Tests that the integral of the PDF over its support is equal to 1."""

        dist = Normal(loc=loc, scale=scale)
        integral, _ = quad(lambda x: dist.pdf(x).item(), loc - scale * 6, loc + scale * 6)
        np.testing.assert_allclose(1.0, integral, atol=1e-7)


class TestNormalLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_return_type_and_shape_for_array_input(self, loc, scale, x):
        """Tests the return type and shape of the lpdf method for array input."""

        dist = Normal(loc=loc, scale=scale)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.dtype == np.float64
        assert lpdf_values.shape == x.shape

    @given(loc=st_loc, scale=st_scale, x=st.floats(-1e6, 1e6))
    def test_lpdf_return_type_and_shape_for_scalar_input(self, loc, scale, x):
        """Tests the return type and shape of the lpdf method for scalar input."""

        dist = Normal(loc=loc, scale=scale)
        lpdf_value = dist.lpdf(x)
        assert np.isscalar(lpdf_value)
        assert isinstance(lpdf_value, np.float64)

    @given(loc=st_loc, scale=st_scale, x=st.floats(-1e6, 1e6))
    def test_lpdf_against_scipy(self, loc, scale, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        dist = Normal(loc=loc, scale=scale)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = norm.logpdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-9)


class TestNormalPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @given(loc=st_loc, scale=st_scale, p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1)))
    def test_ppf_return_type_and_shape_for_array_input(self, loc, scale, p):
        """Tests the return type and shape of the ppf method for array input."""

        dist = Normal(loc=loc, scale=scale)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.dtype == np.float64
        assert ppf_values.shape == p.shape

    @given(loc=st_loc, scale=st_scale, p=st.floats(0, 1))
    def test_ppf_return_type_and_shape_for_scalar_input(self, loc, scale, p):
        """Tests the return type and shape of the ppf method for scalar input."""

        dist = Normal(loc=loc, scale=scale)
        ppf_value = dist.ppf(p)
        assert np.isscalar(ppf_value)
        assert isinstance(ppf_value, np.float64)

    @given(loc=st_loc, scale=st_scale, p=st.floats(1e-6, 1.0 - 1e-6))
    def test_ppf_against_scipy(self, loc, scale, p):
        """Compares the custom PPF implementation against scipy's implementation."""

        dist = Normal(loc=loc, scale=scale)
        custom_ppf = dist.ppf(p)
        scipy_ppf = norm.ppf(p, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-9)

    @pytest.mark.parametrize("p_val", [-0.5, 1.1, 1.5])
    def test_ppf_invalid_input(self, p_val):
        """Tests that PPF returns NaN for probabilities outside the [0, 1] range."""

        dist = Normal(loc=0.0, scale=1.0)
        assert np.isnan(dist.ppf(p_val))


class TestNormalGradients:
    """Tests for gradient calculation methods."""

    h = 1e-6

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(-1e3, 1e3)))
    def test_dlog_loc_numerical_for_array_input(self, loc, scale, x):
        """Checks the analytical gradient for 'loc' against a numerical approximation for array input."""

        dist = Normal(loc, scale)
        analytical_grad = dist._dlog_loc(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == np.float64
        assert analytical_grad.shape == x.shape

        lpdf_plus_h = Normal(loc + self.h, scale).lpdf(x)
        lpdf_minus_h = Normal(loc - self.h, scale).lpdf(x)

        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(loc=st_loc, scale=st_scale, x=st.floats(-1e3, 1e3))
    def test_dlog_loc_for_scalar_input(self, loc, scale, x):
        """Checks that the gradient for 'loc' for a scalar input returns a scalar."""

        dist = Normal(loc, scale)
        analytical_grad = dist._dlog_loc(x)

        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, np.float64)

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(-1e3, 1e3)))
    def test_dlog_scale_numerical_for_array_input(self, loc, scale, x):
        """Checks the analytical gradient for 'scale' against a numerical approximation for array input."""

        dist = Normal(loc, scale)
        analytical_grad = dist._dlog_scale(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == np.float64
        assert analytical_grad.shape == x.shape

        lpdf_plus_h = Normal(loc, scale + self.h).lpdf(x)
        lpdf_minus_h = Normal(loc, scale - self.h).lpdf(x)

        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3)

    @given(loc=st_loc, scale=st_scale, x=st.floats(-1e3, 1e3))
    def test_dlog_scale_for_scalar_input(self, loc, scale, x):
        """Checks that the gradient for 'scale' for a scalar input returns a scalar."""

        dist = Normal(loc, scale)
        analytical_grad = dist._dlog_scale(x)

        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, np.float64)

    @pytest.mark.parametrize(
        "fixed_params, expected_cols, expected_params",
        [([], 2, ["loc", "scale"]), (["loc"], 1, ["scale"]), (["scale"], 1, ["loc"]), (["loc", "scale"], 0, [])],
    )
    def test_log_gradients_structure_for_array_input(self, fixed_params, expected_cols, expected_params):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Normal(loc=1.0, scale=2.0)
        for param in fixed_params:
            dist.fix_param(param)

        x = np.array([-1.0, 1.0, 3.0])
        gradients = dist.log_gradients(x)
        assert gradients.shape == (len(x), expected_cols)

        sorted_params = sorted(expected_params)
        if "loc" in expected_params:
            idx = sorted_params.index("loc")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_loc(x))
        if "scale" in expected_params:
            idx = sorted_params.index("scale")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_scale(x))

    @given(loc=st_loc, scale=st_scale, x=st.floats(-1e3, 1e3))
    def test_log_gradients_for_scalar_input(self, loc, scale, x):
        """Checks that the log_gradients for a scalar input returns a 1D-array."""

        dist = Normal(loc, scale)
        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.dtype == np.float64
        assert gradients.ndim == 1


class TestNormalGenerate:
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
    def test_generate_type_and_shape(self, size, expected_shape, is_scalar):
        """Tests that generated samples have the correct type and shape."""

        dist = Normal(loc=0.0, scale=1.0)
        samples = dist.generate(size=size)

        if is_scalar:
            assert np.isscalar(samples)
            assert isinstance(samples, np.float64)
        else:
            assert isinstance(samples, np.ndarray)
            assert samples.shape == expected_shape
            assert samples.dtype == np.float64

    @pytest.mark.parametrize("size", [-1, -10])
    def test_generate_negative_size(self, size):
        """Tests that generating a negative number of samples raises ValueError."""

        dist = Normal(loc=0.0, scale=1.0)

        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        np.random.seed(123)
        random.seed(123)
        loc, scale = 15.0, 3.0
        dist = Normal(loc=loc, scale=scale)
        size = 50000

        samples = dist.generate(size=size)
        assert np.mean(samples) == pytest.approx(loc, rel=0.05)
        assert np.var(samples) == pytest.approx(scale**2, rel=0.05)

    def test_generate_kolmogorov_smirnov(self):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        loc, scale = -10.0, 5.0
        dist = Normal(loc=loc, scale=scale)
        size = 1000

        samples = dist.generate(size=size)
        ks_statistic, p_value = kstest(samples, norm(loc=loc, scale=scale).cdf)
        expected_p_value = 0.05
        assert p_value > expected_p_value
