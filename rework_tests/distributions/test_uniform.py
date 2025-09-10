"""Tests for Uniform class"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from rework_pysatl_mpest.distributions.uniform import Uniform
from scipy.integrate import quad
from scipy.stats import kstest, uniform


@st.composite
def st_valid_border(draw):
    """Generates valid borders"""
    left_border = draw(st.floats(min_value=-1e3, max_value=1e3 - 1, allow_nan=False, allow_infinity=False))
    right_border = draw(
        st.floats(min_value=left_border + 1e-6, max_value=left_border + 1e3, allow_nan=False, allow_infinity=False)
    )
    return left_border, right_border


class TestUniformInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self):
        """Tests that the instance is initialized correctly with valid parameters."""

        l_border, r_border = 0.5, 2.0
        dist = Uniform(left_border=l_border, right_border=r_border)
        assert isinstance(dist.left_border, float)
        assert isinstance(dist.right_border, float)
        assert dist.left_border == l_border
        assert dist.right_border == r_border

    def test_name_property(self):
        """Tests that the name property returns the correct string."""

        dist = Uniform(left_border=0.0, right_border=1.0)
        assert dist.name == "Uniform"

    def test_params_property(self):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Uniform(left_border=0.0, right_border=1.0)
        assert dist.params == {"left_border", "right_border"}

    def test_invariant_violation(self):
        """Tests that initializing with a infinite borders or left border bigger right border  raises a ValueError."""
        with pytest.raises(ValueError, match="right_border parameter must be strictly greater than left_border"):
            Uniform(0.0, -1.0)
        with pytest.raises(ValueError, match="right_border parameter must be strictly greater than left_border"):
            Uniform(0.0, -2.0)
        with pytest.raises(ValueError, match="right_border parameter must be strictly greater than left_border"):
            Uniform(0.0, 0.0)
        with pytest.raises(ValueError, match="Both borders should be finite values"):
            Uniform(-np.inf, 0.0)
        with pytest.raises(ValueError, match="Both borders should be finite values"):
            Uniform(0.0, np.inf)

    def test_repr_method(self):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Uniform(left_border=1.23, right_border=4.56)
        repr_str = repr(dist)
        assert repr_str == "Uniform(left_border=1.23, right_border=4.56)"

        recreated_dist = eval(repr_str)
        assert isinstance(recreated_dist, Uniform)
        assert recreated_dist.left_border == dist.left_border
        assert recreated_dist.right_border == dist.right_border


class TestUniformPDF:
    """Tests for the pdf method using hypothesis."""

    @given(
        borders=st_valid_border(),
        x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)),
    )
    def test_pdf_properties(self, borders, x):
        """Tests that the PDF is non-negative and has the correct return type and shape."""
        left_border, right_border = borders

        dist = Uniform(left_border=left_border, right_border=right_border)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

    @given(borders=st_valid_border(), x=st.floats(1e-6, 1e6))
    def test_pdf_against_scipy(self, borders, x):
        """Compares the custom PDF implementation against scipy's implementation."""
        left_border, right_border = borders
        dist = Uniform(left_border=left_border, right_border=right_border)
        custom_pdf = dist.pdf(x)
        scipy_pdf = uniform.pdf(x, loc=left_border, scale=right_border - left_border)
        np.testing.assert_allclose(custom_pdf, scipy_pdf, atol=1e-9)

    @given(borders=st_valid_border())
    def test_pdf_integral_is_one(self, borders):
        """Tests that the integral of the PDF over its support is equal to 1."""
        left_border, right_border = borders
        dist = Uniform(left_border=left_border, right_border=right_border)
        integral, error = quad(dist.pdf, left_border, right_border)
        np.testing.assert_allclose(1.0, integral)

    @given(borders=st_valid_border(), x=st.floats(max_value=-1e9, allow_infinity=False))
    def test_pdf_outside_support(self, borders, x):
        """Tests that the PDF is zero for values not in range of parameters."""
        left_border, right_border = borders
        x_val = left_border - abs(x)
        dist = Uniform(left_border=left_border, right_border=right_border)
        assert dist.pdf(x_val) == 0.0


class TestUniformPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @given(
        borders=st_valid_border(),
        p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True)),
    )
    def test_ppf_return_type_and_shape(self, borders, p):
        """Tests the return type and shape of the ppf method."""
        left_border, right_border = borders
        dist = Uniform(left_border=left_border, right_border=right_border)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.shape == p.shape

    @given(borders=st_valid_border(), p=st.floats(0, 1))
    def test_ppf_against_scipy(self, borders, p):
        """Compares the custom PPF implementation against scipy's implementation."""
        left_border, right_border = borders
        dist = Uniform(left_border=left_border, right_border=right_border)
        custom_ppf = dist.ppf(p)
        scipy_ppf = uniform.ppf(p, loc=left_border, scale=right_border - left_border)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-9)

    @pytest.mark.parametrize("p_val", [-0.5, 1.1, 1.5])
    def test_ppf_invalid_input(self, p_val):
        """Tests that PPF returns NaN for probabilities outside the [0, 1) range."""

        dist = Uniform(left_border=0.0, right_border=1.0)
        assert np.isnan(dist.ppf(p_val))


@st.composite
def st_valid_grad_input(draw):
    """Generates valid borders to calculate gradient"""
    left_border = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    right_border = draw(
        st.floats(min_value=left_border + 0.1, max_value=left_border + 20.0, allow_nan=False, allow_infinity=False)
    )

    margin = 0.01
    x_values = draw(
        arrays(
            np.float64,
            st.integers(1, 5),
            elements=st.floats(
                min_value=left_border + margin, max_value=right_border - margin, allow_nan=False, allow_infinity=False
            ),
        )
    )

    return (left_border, right_border), x_values


class TestUniformGradients:
    """Tests for gradient calculation methods."""

    h = 1e-6

    @given(input_data=st_valid_grad_input())
    def test_dlog_left_border_numerical(self, input_data):
        """Checks the analytical gradient for 'left_border' against a numerical approximation."""
        borders, x = input_data
        left_border, right_border = borders

        dist = Uniform(left_border=left_border, right_border=right_border)

        lpdf_plus_h = Uniform(left_border=left_border + self.h, right_border=right_border).lpdf(x)
        lpdf_minus_h = Uniform(left_border=left_border - self.h, right_border=right_border).lpdf(x)

        finite_mask = np.isfinite(lpdf_plus_h) & np.isfinite(lpdf_minus_h)

        if np.any(finite_mask):
            numerical_grad = np.zeros_like(x)
            numerical_grad[finite_mask] = (lpdf_plus_h[finite_mask] - lpdf_minus_h[finite_mask]) / (2 * self.h)

            analytical_grad = dist._dlog_left_border(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.shape == x.shape
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(input_data=st_valid_grad_input())
    def test_dlog_right_border_numerical(self, input_data):
        """Checks the analytical gradient for 'right_border' against a numerical approximation."""
        borders, x = input_data
        left_border, right_border = borders

        dist = Uniform(left_border=left_border, right_border=right_border)

        lpdf_plus_h = Uniform(left_border=left_border, right_border=right_border + self.h).lpdf(x)
        lpdf_minus_h = Uniform(left_border=left_border, right_border=right_border - self.h).lpdf(x)

        finite_mask = np.isfinite(lpdf_plus_h) & np.isfinite(lpdf_minus_h)

        if np.any(finite_mask):
            numerical_grad = np.zeros_like(x)
            numerical_grad[finite_mask] = (lpdf_plus_h[finite_mask] - lpdf_minus_h[finite_mask]) / (2 * self.h)

            analytical_grad = dist._dlog_right_border(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.shape == x.shape
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "fixed_params, expected_shape_col, expected_params",
        [
            ([], 2, ["left_border", "right_border"]),
            (["left_border"], 1, ["right_border"]),
            (["right_border"], 1, ["left_border"]),
            (["left_border", "right_border"], 0, []),
        ],
    )
    def test_log_gradients_structure(self, fixed_params, expected_shape_col, expected_params):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Uniform(left_border=1.0, right_border=3.0)
        for param in fixed_params:
            dist.fix_param(param)

        x = np.array([1.5, 2.0, 3.0])
        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == (len(x), expected_shape_col)

        if "left_border" in expected_params:
            idx = sorted(expected_params).index("left_border")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_left_border(x))
        if "right_border" in expected_params:
            idx = sorted(expected_params).index("right_border")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_right_border(x))


class TestUniformGenerate:
    """Tests for the generate method."""

    def test_generate_type_and_shape(self):
        """Tests that generated samples have the correct type and shape."""

        np.random.seed(42)
        random.seed(42)
        dist = Uniform(left_border=0.0, right_border=2.0)
        size = 100
        samples = dist.generate(size=size)
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float64
        assert samples.shape == (size,)

    def test_generate_zero_size(self):
        """Tests if the generating 0 number of samples returns an empty array"""

        dist = Uniform(left_border=0.0, right_border=1.0)
        assert len(dist.generate(size=0)) == 0

    @pytest.mark.parametrize("size", [-1, -10])
    def test_generate_negative_size(self, size):
        """Tests that generating a negative number of samples raises ValueError."""

        dist = Uniform(left_border=0.0, right_border=1.0)

        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        np.random.seed(123)
        random.seed(123)
        left_border, right_border = 5.0, 5.5
        dist = Uniform(left_border=left_border, right_border=right_border)
        size = 20000

        samples = dist.generate(size=size)

        theoretical_mean = (right_border + left_border) / 2
        theoretical_var = (right_border - left_border) ** 2 / 12

        assert np.mean(samples) == pytest.approx(theoretical_mean, rel=0.1)
        assert np.var(samples) == pytest.approx(theoretical_var, rel=0.1)

    def test_generate_kolmogorov_smirnov(self):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        left_border, right_border = 10.0, 12.0
        dist = Uniform(left_border=left_border, right_border=right_border)
        size = 1000

        samples = dist.generate(size=size)

        ks_statistic, p_value = kstest(samples, "uniform", args=(left_border, right_border - left_border))
        lower_bound = 0.05
        assert p_value > lower_bound
