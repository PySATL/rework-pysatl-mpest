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

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]


@st.composite
def st_valid_border(draw):
    """Generates valid borders"""

    left_border = draw(st.floats(min_value=-1e3, max_value=1e3 - 1, allow_nan=False, allow_infinity=False))
    right_border = draw(
        st.floats(min_value=left_border + 1e-6, max_value=left_border + 1e3, allow_nan=False, allow_infinity=False)
    )
    return left_border, right_border


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestUniformInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self, dtype):
        """Tests that the instance is initialized correctly with valid parameters."""

        l_border, r_border = 0.5, 2.0
        dist = Uniform(left_border=l_border, right_border=r_border, dtype=dtype)
        assert dist.left_border.dtype == dtype
        assert dist.right_border.dtype == dtype
        assert dist.left_border == dtype(l_border)
        assert dist.right_border == dtype(r_border)

    def test_name_property(self, dtype):
        """Tests that the name property returns the correct string."""

        dist = Uniform(left_border=0.0, right_border=1.0, dtype=dtype)
        assert dist.name == "Uniform"

    def test_params_property(self, dtype):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Uniform(left_border=0.0, right_border=1.0, dtype=dtype)
        assert dist.params == {"left_border", "right_border"}

    def test_invariant_violation(self, dtype):
        """Tests that initializing with a infinite borders or left border bigger right border  raises a ValueError."""

        with pytest.raises(ValueError, match="right_border parameter must be strictly greater than left_border"):
            Uniform(0.0, -1.0, dtype=dtype)
        with pytest.raises(ValueError, match="right_border parameter must be strictly greater than left_border"):
            Uniform(0.0, -2.0, dtype=dtype)
        with pytest.raises(ValueError, match="right_border parameter must be strictly greater than left_border"):
            Uniform(0.0, 0.0, dtype=dtype)
        with pytest.raises(ValueError, match="Both borders should be finite values"):
            Uniform(-np.inf, 0.0, dtype=dtype)
        with pytest.raises(ValueError, match="Both borders should be finite values"):
            Uniform(0.0, np.inf, dtype=dtype)

    def test_repr_method(self, dtype):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Uniform(left_border=1.23, right_border=4.56, dtype=dtype)
        repr_str = repr(dist)
        assert (
            repr_str == f"Uniform(left_border={dist.left_border}, right_border={dist.right_border}, "
            f"dtype=np.{dtype.__name__})"
        )

        recreated_dist = eval(repr_str)
        assert dist == recreated_dist


class TestUniformPDF:
    """Tests for the pdf method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(
        borders=st_valid_border(),
        x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)),
    )
    def test_pdf_properties_for_array_input(self, borders, x, dtype):
        """Tests that for an array input, the PDF returns a non-negative array with the correct type and shape."""

        left_border, right_border = borders

        dist = Uniform(left_border=left_border, right_border=right_border, dtype=dtype)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.dtype == dtype
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(x=st.floats(-1e6, 1e6))
    def test_pdf_properties_for_scalar_input(self, x, dtype):
        """Tests that for a scalar input, the PDF returns a non-negative scalar with the correct type."""

        left_border, right_border = -1.0, 12.0
        dist = Uniform(left_border, right_border, dtype=dtype)
        pdf_value = dist.pdf(x)
        assert np.isscalar(pdf_value)
        assert isinstance(pdf_value, dtype)
        assert pdf_value >= 0

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
        integral, error = quad(lambda x: dist.pdf(x).item(), left_border, right_border)
        np.testing.assert_allclose(1.0, integral)

    @given(borders=st_valid_border(), x=st.floats(max_value=-1e9, allow_infinity=False))
    def test_pdf_outside_support(self, borders, x):
        """Tests that the PDF is zero for values not in range of parameters."""

        left_border, right_border = borders
        x_val = left_border - abs(x)
        dist = Uniform(left_border=left_border, right_border=right_border)
        assert dist.pdf(x_val) == 0.0


class TestUniformLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(
        borders=st_valid_border(),
        x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)),
    )
    def test_lpdf_return_type_and_shape_for_array_input(self, borders, x, dtype):
        """Tests the return type and shape of the lpdf method for array input."""

        left_border, right_border = borders
        dist = Uniform(left_border=left_border, right_border=right_border, dtype=dtype)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.dtype == dtype
        assert lpdf_values.shape == x.shape

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(borders=st_valid_border(), x=st.floats(-1e6, 1e6))
    def test_lpdf_return_type_and_shape_for_scalar_input(self, borders, x, dtype):
        """Tests the return type and shape of the lpdf method for scalar input."""

        left_border, right_border = borders
        dist = Uniform(left_border=left_border, right_border=right_border, dtype=dtype)
        lpdf_value = dist.lpdf(x)
        assert np.isscalar(lpdf_value)
        assert isinstance(lpdf_value, dtype)

    @given(borders=st_valid_border(), x=st.floats(1e-6, 1e6))
    def test_lpdf_against_scipy(self, borders, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        left_border, right_border = borders
        dist = Uniform(left_border=left_border, right_border=right_border)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = uniform.logpdf(x, loc=left_border, scale=right_border - left_border)
        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-9)

    @given(borders=st_valid_border(), x=st.floats(min_value=1e-6))
    def test_lpdf_outside_support(self, borders, x):
        """Tests that the LPDF is -inf for values outside the support."""

        left_border, right_border = borders
        dist = Uniform(left_border=left_border, right_border=right_border)
        assert dist.lpdf(left_border - x) == -np.inf
        assert dist.lpdf(right_border + x) == -np.inf


class TestUniformPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(
        borders=st_valid_border(),
        p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True)),
    )
    def test_ppf_return_type_and_shape_for_array_input(self, borders, p, dtype):
        """Tests the return type and shape of the ppf method for array input."""

        left_border, right_border = borders
        dist = Uniform(left_border=left_border, right_border=right_border, dtype=dtype)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.dtype == dtype
        assert ppf_values.shape == p.shape

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(borders=st_valid_border(), p=st.floats(0, 1, exclude_max=True))
    def test_ppf_return_type_and_shape_for_scalar_input(self, borders, p, dtype):
        """Tests the return type and shape of the ppf method for scalar input."""

        left_border, right_border = borders
        dist = Uniform(left_border=left_border, right_border=right_border, dtype=dtype)
        ppf_value = dist.ppf(p)
        assert np.isscalar(ppf_value)
        assert isinstance(ppf_value, dtype)

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
def st_valid_grad_input_array(draw):
    """Generates valid borders to calculate gradient for an array of x."""

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


@st.composite
def st_valid_grad_input_scalar(draw):
    """Generates valid borders to calculate gradient for a scalar x."""

    left_border = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    right_border = draw(
        st.floats(min_value=left_border + 0.1, max_value=left_border + 20.0, allow_nan=False, allow_infinity=False)
    )

    margin = 0.01
    x_value = draw(
        st.floats(
            min_value=left_border + margin, max_value=right_border - margin, allow_nan=False, allow_infinity=False
        )
    )

    return (left_border, right_border), x_value


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestUniformGradients:
    """Tests for gradient calculation methods."""

    h = 1e-6

    @given(input_data=st_valid_grad_input_array())
    def test_dlog_left_border_numerical_for_array_input(self, input_data, dtype):
        """Checks the analytical gradient for 'left_border' against a numerical approximation for array input."""

        borders, x = input_data
        left_border, right_border = borders

        dist = Uniform(left_border=left_border, right_border=right_border, dtype=dtype)
        analytical_grad = dist._dlog_left_border(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Uniform(left_border=left_border + self.h, right_border=right_border).lpdf(x)
            lpdf_minus_h = Uniform(left_border=left_border - self.h, right_border=right_border).lpdf(x)
            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(input_data=st_valid_grad_input_scalar())
    def test_dlog_left_border_for_scalar_input(self, input_data, dtype):
        """Checks that the gradient for 'left_border' for a scalar input returns a scalar."""

        borders, x = input_data
        left_border, right_border = borders

        dist = Uniform(left_border=left_border, right_border=right_border, dtype=dtype)
        analytical_grad = dist._dlog_left_border(x)
        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @given(input_data=st_valid_grad_input_array())
    def test_dlog_right_border_numerical_for_array_input(self, input_data, dtype):
        """Checks the analytical gradient for 'right_border' against a numerical approximation for array input."""

        borders, x = input_data
        left_border, right_border = borders

        dist = Uniform(left_border=left_border, right_border=right_border, dtype=dtype)
        analytical_grad = dist._dlog_right_border(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Uniform(left_border=left_border, right_border=right_border + self.h).lpdf(x)
            lpdf_minus_h = Uniform(left_border=left_border, right_border=right_border - self.h).lpdf(x)
            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3)

    @given(input_data=st_valid_grad_input_scalar())
    def test_dlog_right_border_for_scalar_input(self, input_data, dtype):
        """Checks that the gradient for 'right_border' for a scalar input returns a scalar."""

        borders, x = input_data
        left_border, right_border = borders

        dist = Uniform(left_border=left_border, right_border=right_border, dtype=dtype)
        analytical_grad = dist._dlog_right_border(x)
        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @pytest.mark.parametrize(
        "fixed_params, expected_shape_col, expected_params",
        [
            ([], 2, ["left_border", "right_border"]),
            (["left_border"], 1, ["right_border"]),
            (["right_border"], 1, ["left_border"]),
            (["left_border", "right_border"], 0, []),
        ],
    )
    def test_log_gradients_structure(self, fixed_params, expected_shape_col, expected_params, dtype):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Uniform(left_border=1.0, right_border=3.0, dtype=dtype)
        for param in fixed_params:
            dist.fix_param(param)

        x = np.array([1.5, 2.0, 3.0])
        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.dtype == dtype
        assert gradients.shape == (len(x), expected_shape_col)

        if "left_border" in expected_params:
            idx = sorted(expected_params).index("left_border")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_left_border(x))
        if "right_border" in expected_params:
            idx = sorted(expected_params).index("right_border")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_right_border(x))

    @given(input_data=st_valid_grad_input_scalar())
    def test_log_gradients_for_scalar_input(self, input_data, dtype):
        """Checks that the log_gradients for a scalar input returns a 1D-array."""

        borders, x = input_data
        left_border, right_border = borders
        dist = Uniform(left_border, right_border, dtype=dtype)
        gradients = dist.log_gradients(x)
        assert isinstance(gradients, np.ndarray)
        assert gradients.dtype == dtype
        assert gradients.ndim == 1


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
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
    def test_generate_type_and_shape(self, dtype, size, expected_shape, is_scalar):
        """Tests that generated samples have the correct type and shape."""

        np.random.seed(42)
        random.seed(42)
        dist = Uniform(left_border=0.0, right_border=2.0, dtype=dtype)
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

        dist = Uniform(left_border=0.0, right_border=1.0, dtype=dtype)

        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self, dtype):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        np.random.seed(123)
        random.seed(123)
        left_border, right_border = 5.0, 5.5
        dist = Uniform(left_border=left_border, right_border=right_border, dtype=dtype)
        size = 20000

        samples = dist.generate(size=size)

        theoretical_mean = (right_border + left_border) / 2
        theoretical_var = (right_border - left_border) ** 2 / 12

        assert np.mean(samples, dtype=np.float64) == pytest.approx(theoretical_mean, rel=0.1)
        assert np.var(samples, dtype=np.float64) == pytest.approx(theoretical_var, rel=0.1)

    def test_generate_kolmogorov_smirnov(self, dtype):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        left_border, right_border = 10.0, 12.0
        dist = Uniform(left_border=left_border, right_border=right_border, dtype=dtype)
        size = 1000

        samples = dist.generate(size=size)

        ks_statistic, p_value = kstest(samples, "uniform", args=(left_border, right_border - left_border))
        lower_bound = 0.05
        assert p_value > lower_bound
