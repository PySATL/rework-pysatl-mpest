"""Tests for Exponential class"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pysatl_mpest.distributions import Exponential
from scipy.integrate import quad
from scipy.stats import expon, kstest
from tests.helpers import assert_is_array_type, assert_is_scalar_type

st_lambda_ = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)


class TestExponentialInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self):
        """Tests that the instance is initialized correctly with valid parameters."""

        lambda_ = 2.0
        dist = Exponential(lambda_=lambda_)
        assert dist.lambda_ == lambda_

    def test_name_property(self):
        """Tests that the name property returns the correct string."""

        dist = Exponential(lambda_=1.0)
        assert dist.name == "Exponential"

    def test_params_property(self):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Exponential(lambda_=1.0)
        assert dist.params == {"lambda_"}

    def test_repr_method(self):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Exponential(lambda_=4.56)
        repr_str = repr(dist)
        assert repr_str == f"Exponential(lambda_={dist.lambda_})"

        recreated_dist = eval(repr_str)
        assert recreated_dist == dist


class TestExponentialPDF:
    """Tests for the pdf method using hypothesis."""

    @given(lambda_=st_lambda_, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_pdf_properties_for_array_input(self, lambda_, x):
        """Tests that for an array input, the PDF returns a non-negative array with the correct shape."""

        dist = Exponential(lambda_=lambda_)
        pdf_values = dist.pdf(x)
        assert_is_array_type(pdf_values, x.shape)
        assert np.all(pdf_values >= 0)

    @given(lambda_=st_lambda_, x=st.floats(-1e6, 1e6))
    def test_pdf_properties_for_scalar_input(self, lambda_, x):
        """Tests that for a scalar input, the PDF returns a non-negative scalar."""

        dist = Exponential(lambda_)
        pdf_value = dist.pdf(x)
        assert_is_scalar_type(pdf_value)
        assert pdf_value >= 0

    @given(lambda_=st_lambda_, x=st.floats(1e-6, 1e6))
    def test_pdf_against_scipy(self, lambda_, x):
        """Compares the custom PDF implementation against scipy's implementation."""

        dist = Exponential(lambda_=lambda_)
        custom_pdf = dist.pdf(x)
        scipy_pdf = expon.pdf(x, scale=1 / lambda_)
        np.testing.assert_allclose(custom_pdf, scipy_pdf, atol=1e-9)

    @given(lambda_=st_lambda_)
    def test_pdf_integral_is_one(self, lambda_):
        """Tests that the integral of the PDF over its support is equal to 1."""

        dist = Exponential(lambda_=lambda_)
        integral, _ = quad(lambda x: dist.pdf(x).item(), 0, np.inf)
        np.testing.assert_allclose(1.0, integral)

    @given(lambda_=st_lambda_, x=st.floats(max_value=-1e6, allow_infinity=False))
    def test_pdf_outside_support(self, lambda_, x):
        """Tests that the PDF is zero for values less than the location parameter."""

        dist = Exponential(lambda_=lambda_)
        assert dist.pdf(x) == 0.0


class TestExponentialLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @given(lambda_=st_lambda_, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_return_type_and_shape_for_array_input(self, lambda_, x):
        """Tests the return type and shape of the lpdf method for array input."""

        dist = Exponential(lambda_=lambda_)
        lpdf_values = dist.lpdf(x)
        assert_is_array_type(lpdf_values, x.shape)

    @given(lambda_=st_lambda_, x=st.floats(-1e6, 1e6))
    def test_lpdf_return_type_and_shape_for_scalar_input(self, lambda_, x):
        """Tests the return type and shape of the lpdf method for scalar input."""

        dist = Exponential(lambda_)
        lpdf_value = dist.lpdf(x)
        assert_is_scalar_type(lpdf_value)

    @given(lambda_=st_lambda_, x=st.floats(1e-6, 1e6))
    def test_lpdf_against_scipy(self, lambda_, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        dist = Exponential(lambda_=lambda_)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = expon.logpdf(x, scale=1 / lambda_)
        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-9)

    @given(lambda_=st_lambda_, x=st.floats(max_value=-1e6, allow_infinity=False))
    def test_lpdf_outside_support(self, lambda_, x):
        """Tests that the LPDF is -inf for values less than the location parameter."""

        dist = Exponential(lambda_=lambda_)
        assert dist.lpdf(x) == -np.inf


class TestExponentialPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @given(lambda_=st_lambda_, p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True)))
    def test_ppf_return_type_and_shape_for_array_input(self, lambda_, p):
        """Tests the return type and shape of the ppf method for array input."""

        dist = Exponential(lambda_=lambda_)
        ppf_values = dist.ppf(p)
        assert_is_array_type(ppf_values, p.shape)

    @given(lambda_=st_lambda_, p=st.floats(0, 1, exclude_max=True))
    def test_ppf_return_type_and_shape_for_scalar_input(self, lambda_, p):
        """Tests the return type and shape of the ppf method for scalar input."""

        dist = Exponential(lambda_=lambda_)
        ppf_value = dist.ppf(p)
        assert_is_scalar_type(ppf_value)

    @given(lambda_=st_lambda_, p=st.floats(0, 1, exclude_max=True, exclude_min=True))
    def test_ppf_against_scipy(self, lambda_, p):
        """Compares the custom PPF implementation against scipy's implementation."""

        dist = Exponential(lambda_=lambda_)
        custom_ppf = dist.ppf(p)
        scipy_ppf = expon.ppf(p, scale=1 / lambda_)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-9)


class TestExponentialGradients:
    """Tests for gradient calculation methods."""

    @given(lambda_=st_lambda_, x=st.floats(1e-3, 1e3))
    def test_log_gradients_for_scalar_input(self, lambda_, x):
        """Checks that the log_gradients for a scalar input returns a 1D-array."""

        dist = Exponential(lambda_)
        gradients = dist.log_gradients(x)
        assert_is_array_type(gradients, (1,))


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
    def test_lambda_type_and_shape(self, size, expected_shape, is_scalar):
        """Tests that generated samples have the correct type and shape."""

        np.random.seed(42)
        random.seed(42)
        dist = Exponential(lambda_=2.0)
        samples = dist.generate(size=size)

        if is_scalar:
            assert_is_scalar_type(samples)
        else:
            assert_is_array_type(samples, expected_shape)

    @pytest.mark.parametrize("size", [-1, -10])
    def test_lambda_negative_size(self, size):
        """Tests that generating a negative number of samples raises ValueError."""

        dist = Exponential(lambda_=1.0)

        with pytest.raises(ValueError):
            dist.generate(size=size, random_state=42)

    def test_lambda_statistical_properties(self):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        lambda_ = 0.5
        dist = Exponential(lambda_=lambda_)
        size = 20000

        samples = dist.generate(size=size, random_state=42)

        theoretical_mean = 1 / lambda_
        theoretical_var = (1 / lambda_) ** 2

        assert np.mean(samples, dtype=np.float64) == pytest.approx(theoretical_mean, rel=0.1)
        assert np.var(samples, dtype=np.float64) == pytest.approx(theoretical_var, rel=0.1)

    def test_lambda_kolmogorov_smirnov(self):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        lambda_ = 2.0
        dist = Exponential(lambda_=lambda_)
        size = 10000

        samples = dist.generate(size=size, random_state=42)

        _, p_value = kstest(
            samples,
            "expon",
            args=(
                0,
                1 / lambda_,
            ),
        )
        lower_bound = 0.05
        assert p_value > lower_bound
