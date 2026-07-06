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
from tests.helpers import assert_is_array_type, assert_is_scalar_type

# Strategies for hypothesis
st_mu = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
st_sigma = st.floats(min_value=0.01, max_value=1e3, allow_nan=False, allow_infinity=False)


class TestNormalInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self):
        """Tests that the instance is initialized correctly with valid parameters."""

        mu, sigma = 10.0, 2.5
        dist = Normal(mu=mu, sigma=sigma)
        assert isinstance(dist.mu, float)
        assert isinstance(dist.sigma, float)
        assert dist.mu == float(mu)
        assert dist.sigma == float(sigma)

    def test_name_property(self):
        """Tests that the name property returns the correct string."""

        dist = Normal(mu=0.0, sigma=1.0)
        assert dist.name == "Normal"

    def test_params_property(self):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Normal(mu=0.0, sigma=1.0)
        assert dist.params == {"mu", "sigma"}

    def test_repr_method(self):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Normal(mu=1.23, sigma=4.56)
        repr_str = repr(dist)
        assert repr_str == f"Normal(mu={dist.mu}, sigma={dist.sigma})"

        recreated_dist = eval(repr_str)
        assert dist == recreated_dist


class TestNormalPDF:
    """Tests for the pdf method using hypothesis."""

    @given(mu=st_mu, sigma=st_sigma, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_pdf_properties_for_array_input(self, mu, sigma, x):
        """Tests that for an array input, the PDF returns a non-negative array with the correct type and shape."""

        dist = Normal(mu=mu, sigma=sigma)
        pdf_values = dist.pdf(x)
        assert_is_array_type(pdf_values, x.shape)
        assert np.all(pdf_values >= 0)

    @given(mu=st_mu, sigma=st_sigma, x=st.floats(-1e6, 1e6))
    def test_pdf_properties_for_scalar_input(self, mu, sigma, x):
        """Tests that for a scalar input, the PDF returns a non-negative scalar with the correct type."""

        dist = Normal(mu, sigma)
        pdf_value = dist.pdf(x)
        assert_is_scalar_type(pdf_value)
        assert pdf_value >= 0

    @given(mu=st_mu, sigma=st_sigma, x=st.floats(-1e6, 1e6))
    def test_pdf_against_scipy(self, mu, sigma, x):
        """Compares the custom PDF implementation against scipy's implementation."""

        dist = Normal(mu=mu, sigma=sigma)
        custom_pdf = dist.pdf(x)
        scipy_pdf = norm.pdf(x, loc=mu, scale=sigma)
        np.testing.assert_allclose(custom_pdf, scipy_pdf, atol=1e-9)

    @given(mu=st_mu, sigma=st_sigma)
    def test_pdf_integral_is_one(self, mu, sigma):
        """Tests that the integral of the PDF over its support is equal to 1."""

        dist = Normal(mu=mu, sigma=sigma)
        integral, _ = quad(lambda x: dist.pdf(x).item(), mu - sigma * 6, mu + sigma * 6)
        np.testing.assert_allclose(1.0, integral, atol=1e-7)


class TestNormalLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @given(mu=st_mu, sigma=st_sigma, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_return_type_and_shape_for_array_input(self, mu, sigma, x):
        """Tests the return type and shape of the lpdf method for array input."""

        dist = Normal(mu=mu, sigma=sigma)
        lpdf_values = dist.lpdf(x)
        assert_is_array_type(lpdf_values, x.shape)

    @given(mu=st_mu, sigma=st_sigma, x=st.floats(-1e6, 1e6))
    def test_lpdf_return_type_and_shape_for_scalar_input(self, mu, sigma, x):
        """Tests the return type and shape of the lpdf method for scalar input."""

        dist = Normal(mu=mu, sigma=sigma)
        lpdf_value = dist.lpdf(x)
        assert_is_scalar_type(lpdf_value)

    @given(mu=st_mu, sigma=st_sigma, x=st.floats(-1e6, 1e6))
    def test_lpdf_against_scipy(self, mu, sigma, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        dist = Normal(mu=mu, sigma=sigma)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = norm.logpdf(x, loc=mu, scale=sigma)
        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-9)


class TestNormalPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @given(mu=st_mu, sigma=st_sigma, p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1)))
    def test_ppf_return_type_and_shape_for_array_input(self, mu, sigma, p):
        """Tests the return type and shape of the ppf method for array input."""

        dist = Normal(mu=mu, sigma=sigma)
        ppf_values = dist.ppf(p)
        assert_is_array_type(ppf_values, p.shape)

    @given(mu=st_mu, sigma=st_sigma, p=st.floats(0, 1))
    def test_ppf_return_type_and_shape_for_scalar_input(self, mu, sigma, p):
        """Tests the return type and shape of the ppf method for scalar input."""

        dist = Normal(mu=mu, sigma=sigma)
        ppf_value = dist.ppf(p)
        assert_is_scalar_type(ppf_value)

    @given(mu=st_mu, sigma=st_sigma, p=st.floats(1e-6, 1.0 - 1e-6))
    def test_ppf_against_scipy(self, mu, sigma, p):
        """Compares the custom PPF implementation against scipy's implementation."""

        dist = Normal(mu=mu, sigma=sigma)
        custom_ppf = dist.ppf(p)
        scipy_ppf = norm.ppf(p, loc=mu, scale=sigma)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-9)


class TestNormalGradients:
    """Tests for gradient calculation methods."""

    @pytest.mark.parametrize(
        "fixed_params, expected_cols, expected_params",
        [([], 2, ["mu", "sigma"]), (["mu"], 1, ["sigma"]), (["sigma"], 1, ["mu"]), (["mu", "sigma"], 0, [])],
    )
    def test_log_gradients_structure_for_array_input(self, fixed_params, expected_cols, expected_params):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Normal(mu=1.0, sigma=2.0)
        for param in fixed_params:
            dist.fix_param(param)

        x = np.array([-1.0, 1.0, 3.0])
        gradients = dist.log_gradients(x)
        assert gradients.shape == (len(x), expected_cols)

    @given(mu=st_mu, sigma=st_sigma, x=st.floats(-1e3, 1e3))
    def test_log_gradients_for_scalar_input(self, mu, sigma, x):
        """Checks that the log_gradients for a scalar input returns a 1D-array."""

        dist = Normal(mu, sigma)
        gradients = dist.log_gradients(x)
        assert_is_array_type(gradients, (2,))

    @pytest.mark.parametrize(
        "fixed_params, expected_cols",
        [([], 2), (["mu"], 1), (["sigma"], 1), (["mu", "sigma"], 0)],
    )
    def test_log_gradients_structure_for_scalar_input_parametrized(self, fixed_params, expected_cols):
        """Tests the structure of log_gradients with scalar input and various fixed parameters."""

        dist = Normal(mu=1.0, sigma=2.0)
        for param in fixed_params:
            dist.fix_param(param)

        x = 0.5
        gradients = dist.log_gradients(x)
        assert gradients.shape == (expected_cols,)


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

        dist = Normal(mu=0.0, sigma=1.0)
        samples = dist.generate(size=size)

        if is_scalar:
            assert_is_scalar_type(samples)
        else:
            assert_is_array_type(samples, expected_shape)

    @pytest.mark.parametrize("size", [-1, -10])
    def test_generate_negative_size(self, size):
        """Tests that generating a negative number of samples raises ValueError."""

        dist = Normal(mu=0.0, sigma=1.0)

        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        np.random.seed(123)
        random.seed(123)
        mu, sigma = 15.0, 3.0
        dist = Normal(mu=mu, sigma=sigma)
        size = 50000

        samples = dist.generate(size=size)
        assert np.mean(samples) == pytest.approx(mu, rel=0.05)
        assert np.var(samples) == pytest.approx(sigma**2, rel=0.05)

    def test_generate_kolmogorov_smirnov(self):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        mu, sigma = -10.0, 5.0
        dist = Normal(mu=mu, sigma=sigma)
        size = 10000

        samples = dist.generate(size=size)
        ks_statistic, p_value = kstest(samples, norm(loc=mu, scale=sigma).cdf)
        expected_p_value = 0.05
        assert p_value > expected_p_value
