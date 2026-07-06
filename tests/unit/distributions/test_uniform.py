"""Tests for Uniform class"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pysatl_mpest.distributions import Uniform
from scipy.integrate import quad
from scipy.stats import kstest, uniform


@st.composite
def st_valid_border(draw):
    """Generates valid borders"""

    lower_bound = draw(st.floats(min_value=-1e3, max_value=1e3 - 1, allow_nan=False, allow_infinity=False))
    upper_bound = draw(
        st.floats(min_value=lower_bound + 1e-6, max_value=lower_bound + 1e3, allow_nan=False, allow_infinity=False)
    )
    return lower_bound, upper_bound


class TestUniformInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self):
        """Tests that the instance is initialized correctly with valid parameters."""

        l_border, r_border = 0.5, 2.0
        dist = Uniform(lower_bound=l_border, upper_bound=r_border)
        assert dist.lower_bound == l_border
        assert dist.upper_bound == r_border

    def test_name_property(self):
        """Tests that the name property returns the correct string."""

        dist = Uniform(lower_bound=0.0, upper_bound=1.0)
        assert dist.name == "ContinuousUniform"

    def test_params_property(self):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Uniform(lower_bound=0.0, upper_bound=1.0)
        assert dist.params == {"lower_bound", "upper_bound"}

    def test_repr_method(self):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Uniform(lower_bound=1.23, upper_bound=4.56)
        repr_str = repr(dist)
        assert repr_str == f"Uniform(lower_bound={dist.lower_bound}, upper_bound={dist.upper_bound})"

        recreated_dist = eval(repr_str)
        assert dist == recreated_dist


class TestUniformPDF:
    """Tests for the pdf method using hypothesis."""

    @given(
        borders=st_valid_border(),
        x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)),
    )
    def test_pdf_properties_for_array_input(self, borders, x):
        """Tests that for an array input, the PDF returns a non-negative array with the correct type and shape."""

        lower_bound, upper_bound = borders

        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

    @given(x=st.floats(-1e6, 1e6))
    def test_pdf_properties_for_scalar_input(self, x):
        """Tests that for a scalar input, the PDF returns a non-negative scalar with the correct type."""

        lower_bound, upper_bound = -1.0, 12.0
        dist = Uniform(lower_bound, upper_bound)
        pdf_value = dist.pdf(x)
        assert isinstance(pdf_value, float)
        assert pdf_value >= 0

    @given(borders=st_valid_border(), x=st.floats(1e-6, 1e6))
    def test_pdf_against_scipy(self, borders, x):
        """Compares the custom PDF implementation against scipy's implementation."""

        lower_bound, upper_bound = borders
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        custom_pdf = dist.pdf(x)
        scipy_pdf = uniform.pdf(x, loc=lower_bound, scale=upper_bound - lower_bound)
        np.testing.assert_allclose(custom_pdf, scipy_pdf, atol=1e-9)

    @given(borders=st_valid_border())
    def test_pdf_integral_is_one(self, borders):
        """Tests that the integral of the PDF over its support is equal to 1."""

        lower_bound, upper_bound = borders
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        integral, error = quad(lambda x: dist.pdf(x).item(), lower_bound, upper_bound)
        np.testing.assert_allclose(1.0, integral)

    @given(borders=st_valid_border(), x=st.floats(max_value=-1e9, allow_infinity=False))
    def test_pdf_outside_support(self, borders, x):
        """Tests that the PDF is zero for values not in range of parameters."""

        lower_bound, upper_bound = borders
        x_val = lower_bound - abs(x)
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        assert dist.pdf(x_val) == 0.0


class TestUniformLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @given(
        borders=st_valid_border(),
        x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)),
    )
    def test_lpdf_return_type_and_shape_for_array_input(self, borders, x):
        """Tests the return type and shape of the lpdf method for array input."""

        lower_bound, upper_bound = borders
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.shape == x.shape

    @given(borders=st_valid_border(), x=st.floats(-1e6, 1e6))
    def test_lpdf_return_type_and_shape_for_scalar_input(self, borders, x):
        """Tests the return type and shape of the lpdf method for scalar input."""

        lower_bound, upper_bound = borders
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        lpdf_value = dist.lpdf(x)
        assert isinstance(lpdf_value, float)

    @given(borders=st_valid_border(), x=st.floats(1e-6, 1e6))
    def test_lpdf_against_scipy(self, borders, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        lower_bound, upper_bound = borders
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = uniform.logpdf(x, loc=lower_bound, scale=upper_bound - lower_bound)
        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-9)

    @given(borders=st_valid_border(), x=st.floats(min_value=1e-6))
    def test_lpdf_outside_support(self, borders, x):
        """Tests that the LPDF is -inf for values outside the support."""

        lower_bound, upper_bound = borders
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        assert dist.lpdf(lower_bound - x) == -np.inf
        assert dist.lpdf(upper_bound + x) == -np.inf


class TestUniformPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @given(
        borders=st_valid_border(),
        p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True)),
    )
    def test_ppf_return_type_and_shape_for_array_input(self, borders, p):
        """Tests the return type and shape of the ppf method for array input."""

        lower_bound, upper_bound = borders
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.shape == p.shape

    @given(borders=st_valid_border(), p=st.floats(0, 1, exclude_max=True))
    def test_ppf_return_type_and_shape_for_scalar_input(self, borders, p):
        """Tests the return type and shape of the ppf method for scalar input."""

        lower_bound, upper_bound = borders
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        ppf_value = dist.ppf(p)
        assert isinstance(ppf_value, float)

    @given(borders=st_valid_border(), p=st.floats(0, 1))
    def test_ppf_against_scipy(self, borders, p):
        """Compares the custom PPF implementation against scipy's implementation."""

        lower_bound, upper_bound = borders
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        custom_ppf = dist.ppf(p)
        scipy_ppf = uniform.ppf(p, loc=lower_bound, scale=upper_bound - lower_bound)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-9)


@st.composite
def st_valid_grad_input_array(draw):
    """Generates valid borders to calculate gradient for an array of x."""

    lower_bound = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    upper_bound = draw(
        st.floats(min_value=lower_bound + 0.1, max_value=lower_bound + 20.0, allow_nan=False, allow_infinity=False)
    )

    margin = 0.01
    x_values = draw(
        arrays(
            np.float64,
            st.integers(1, 5),
            elements=st.floats(
                min_value=lower_bound + margin, max_value=upper_bound - margin, allow_nan=False, allow_infinity=False
            ),
        )
    )

    return (lower_bound, upper_bound), x_values


@st.composite
def st_valid_grad_input_scalar(draw):
    """Generates valid borders to calculate gradient for a scalar x."""

    lower_bound = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    upper_bound = draw(
        st.floats(min_value=lower_bound + 0.1, max_value=lower_bound + 20.0, allow_nan=False, allow_infinity=False)
    )

    margin = 0.01
    x_value = draw(
        st.floats(min_value=lower_bound + margin, max_value=upper_bound - margin, allow_nan=False, allow_infinity=False)
    )

    return (lower_bound, upper_bound), x_value


class TestUniformGradients:
    """Tests for gradient calculation methods."""

    @given(input_data=st_valid_grad_input_scalar())
    def test_log_gradients_for_scalar_input(self, input_data):
        """Checks that the log_gradients for a scalar input returns a 1D-array."""

        borders, x = input_data
        lower_bound, upper_bound = borders
        dist = Uniform(lower_bound, upper_bound)
        gradients = dist.log_gradients(x)
        assert isinstance(gradients, np.ndarray)
        assert gradients.ndim == 1


class TestUniformGenerate:
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

        np.random.seed(42)
        random.seed(42)
        dist = Uniform(lower_bound=0.0, upper_bound=2.0)
        samples = dist.generate(size=size, random_state=42)

        if is_scalar:
            assert isinstance(samples, float)
        else:
            assert isinstance(samples, np.ndarray)
            assert samples.shape == expected_shape

    @pytest.mark.parametrize("size", [-1, -10])
    def test_generate_negative_size(self, size):
        """Tests that generating a negative number of samples raises ValueError."""

        dist = Uniform(lower_bound=0.0, upper_bound=1.0)

        with pytest.raises(ValueError):
            dist.generate(size=size, random_state=42)

    def test_generate_statistical_properties(self):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        lower_bound, upper_bound = 5.0, 5.5
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        size = 20000

        samples = dist.generate(size=size, random_state=42)

        theoretical_mean = (upper_bound + lower_bound) / 2
        theoretical_var = (upper_bound - lower_bound) ** 2 / 12

        assert np.mean(samples, dtype=np.float64) == pytest.approx(theoretical_mean, rel=0.1)
        assert np.var(samples, dtype=np.float64) == pytest.approx(theoretical_var, rel=0.1)

    def test_generate_kolmogorov_smirnov(self):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        lower_bound, upper_bound = 10.0, 12.0
        dist = Uniform(lower_bound=lower_bound, upper_bound=upper_bound)
        size = 10000

        samples = dist.generate(size=size, random_state=42)

        _, p_value = kstest(samples, "uniform", args=(lower_bound, upper_bound - lower_bound))
        lower_bound = 0.05
        assert p_value > lower_bound
