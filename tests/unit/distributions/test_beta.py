"""Tests for Beta class"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pysatl_mpest.distributions import Beta
from scipy.integrate import quad
from scipy.stats import beta, kstest

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]


@st.composite
def st_valid_params(draw):
    """Generates well-behaved beta parameters for integration tests."""
    shape1 = draw(st.floats(min_value=1.0, max_value=50.0))
    shape2 = draw(st.floats(min_value=1.0, max_value=50.0))

    left_border = draw(st.floats(min_value=-10.0, max_value=10.0))
    right_border = draw(st.floats(min_value=left_border + 0.5, max_value=left_border + 10.0))

    return shape1, shape2, left_border, right_border


def load_r_test_cases():
    """Loads constants calculated in R with function extraDistr::dnsbeta"""
    csv_path = Path(__file__).parent / "constraints" / "test_beta4params.csv"
    data = pd.read_csv(csv_path)

    id_col = data.columns[0]
    param_cols = data.columns[1:]

    cases = []
    for _, row in data.iterrows():
        params = [np.float64(row[col]) for col in param_cols]
        cases.append(pytest.param(*params, id=str(row[id_col])))

    return cases


@st.composite
def st_params_and_array_x_for_grad(draw):
    """Generator of valid params with an array x for gradient test"""
    shape1 = draw(st.floats(0.1, 100))
    shape2 = draw(st.floats(0.1, 100))
    left = draw(st.floats(-100, 100))
    right = draw(st.floats(left + 0.1, left + 100))

    margin = 1e-2
    x = draw(
        arrays(
            np.float64,
            shape=draw(st.integers(2, 5)),
            elements=st.floats(
                min_value=left + margin, max_value=right - margin, allow_nan=False, allow_infinity=False
            ),
        )
    )
    return (shape1, shape2, left, right, x)


@st.composite
def st_params_and_scalar_x_for_grad(draw):
    """Generator of valid params with a scalar x for gradient test"""
    shape1 = draw(st.floats(0.1, 100))
    shape2 = draw(st.floats(0.1, 100))
    lower = draw(st.floats(-100, 100))
    upper = draw(st.floats(lower + 0.1, lower + 100))

    margin = 1e-2
    x = draw(st.floats(min_value=lower + margin, max_value=upper - margin, allow_nan=False, allow_infinity=False))
    return (shape1, shape2, lower, upper, x)


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestBetaInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self, dtype):
        """Tests that the instance is initialized correctly with valid parameters."""

        shape1, shape2, left_border, right_border = 0.5, 2.0, -1.0, 1.0
        dist = Beta(alpha=shape1, beta=shape2, left_border=left_border, right_border=right_border, dtype=dtype)
        assert dist.alpha.dtype == dtype
        assert dist.beta.dtype == dtype
        assert dist.left_border.dtype == dtype
        assert dist.right_border.dtype == dtype
        assert dist.alpha == dtype(shape1)
        assert dist.beta == dtype(shape2)
        assert dist.left_border == dtype(left_border)
        assert dist.right_border == dtype(right_border)

    def test_name_property(self, dtype):
        """Tests that the name property returns the correct string."""

        dist = Beta(alpha=1.0, beta=2.0, left_border=-1.0, right_border=1.0, dtype=dtype)
        assert dist.name == "Beta"

    def test_params_property(self, dtype):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Beta(alpha=1.0, beta=2.0, left_border=-1.0, right_border=1.0, dtype=dtype)
        assert dist.params == {"alpha", "beta", "left_border", "right_border"}

    def test_alpha_invariant_violation(self, dtype):
        """Tests that initializing with a non-positive alpha raises a ValueError."""

        with pytest.raises(ValueError, match="Alpha parameter should be positive or zero"):
            Beta(alpha=-1.0, beta=2.0, left_border=10.0, right_border=20.0, dtype=dtype)
        with pytest.raises(ValueError, match="Alpha parameter should be positive or zero"):
            Beta(alpha=-20.0, beta=2.0, left_border=10.0, right_border=20.0, dtype=dtype)

    def test_alpha_assignment_violation(self, dtype):
        """Tests that assigning a non-positive rate after initialization raises a ValueError."""

        dist = Beta(alpha=1.0, beta=2.0, left_border=10.0, right_border=20.0, dtype=dtype)
        with pytest.raises(ValueError, match="Alpha parameter should be positive or zero"):
            dist.alpha = -1.0
        with pytest.raises(ValueError, match="Alpha parameter should be positive or zero"):
            dist.alpha = -10.0

    def test_beta_invariant_violation(self, dtype):
        """Tests that initializing with a non-positive beta raises a ValueError."""

        with pytest.raises(ValueError, match="Beta parameter should be positive or zero"):
            Beta(alpha=1.0, beta=-2.0, left_border=10.0, right_border=20.0, dtype=dtype)
        with pytest.raises(ValueError, match="Beta parameter should be positive or zero"):
            Beta(alpha=1.0, beta=-20.0, left_border=10.0, right_border=20.0, dtype=dtype)

    def test_beta_assignment_violation(self, dtype):
        """Tests that assigning a non-positive beta after initialization raises a ValueError."""

        dist = Beta(alpha=1.0, beta=2.0, left_border=10.0, right_border=20.0, dtype=dtype)
        with pytest.raises(ValueError, match="Beta parameter should be positive or zero"):
            dist.beta = -1.0
        with pytest.raises(ValueError, match="Beta parameter should be positive or zero"):
            dist.beta = -10.0

    def test_invariant_bounds_violation(self, dtype):
        """Tests that initializing with a left border bigger right border raises a ValueError."""

        with pytest.raises(ValueError, match="Left border must be less than right border"):
            Beta(alpha=1.0, beta=2.0, left_border=10.0, right_border=5.0, dtype=dtype)
        with pytest.raises(ValueError, match="Left border must be less than right border"):
            Beta(alpha=1.0, beta=2.0, left_border=10.0, right_border=10.0, dtype=dtype)

    def test_repr_method(self, dtype):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Beta(alpha=1.1, beta=2.1, left_border=10.1, right_border=20.1, dtype=dtype)
        repr_str = repr(dist)
        assert (
            repr_str == f"Beta(alpha={dist.alpha}, beta={dist.beta}, left_border={dist.left_border}, "
            f"right_border={dist.right_border}, dtype=np.{dtype.__name__})"
        )

        recreated_dist = eval(repr_str)
        assert dist == recreated_dist


class TestBetaPDF:
    """Tests for the pdf method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_pdf_properties_for_array_input(self, x, dtype):
        """Tests that for an array input, the PDF returns a non-negative array with the correct type and shape."""

        alpha, beta, left_border, right_border = 1.0, 1.0, 2.9, 10.0
        dist = Beta(alpha, beta, left_border, right_border, dtype=dtype)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.dtype == dtype
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(x=st.floats(-1e6, 1e6))
    def test_pdf_properties_for_scalar_input(self, x, dtype):
        """Tests that for a scalar input, the PDF returns a non-negative scalar with the correct type and shape."""

        alpha, beta, left_border, right_border = 1.0, 1.0, 2.9, 10.0
        dist = Beta(alpha, beta, left_border, right_border, dtype=dtype)
        pdf_value = dist.pdf(x)
        assert np.isscalar(pdf_value)
        assert isinstance(pdf_value, dtype)
        assert pdf_value >= 0

    @pytest.mark.parametrize("x,shape1,shape2,left_border,right_border,expected_pdf", load_r_test_cases())
    def test_pdf_against_R(self, x, shape1, shape2, left_border, right_border, expected_pdf):
        """Compares the custom PDF implementation against scipy's implementation."""

        dist = Beta(shape1, shape2, left_border, right_border)
        custom_pdf = dist.pdf(x)
        np.testing.assert_allclose(custom_pdf, expected_pdf, atol=1e-9)

    @given(params=st_valid_params())
    def test_pdf_integral_is_one(self, params):
        """Tests that the integral of the PDF over its support is equal to 1."""

        shape1, shape2, left_border, right_border = params
        dist = Beta(shape1, shape2, left_border, right_border)
        integral, error = quad(lambda x: dist.pdf(x).item(), left_border, right_border)
        np.testing.assert_allclose(1.0, integral, rtol=1e-6, atol=1e-8)

    @given(x=st.floats(min_value=1e-4, max_value=1e2, allow_infinity=False))
    def test_pdf_outside_support(self, x):
        """Tests that the PDF is zero for values less than the location parameter."""

        x_val = 2.0 - abs(x)
        dist = Beta(1.0, 1.0, 2.0, 10.0)
        assert dist.pdf(x_val) == 0.0


class TestBetaLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(params=st_valid_params(), x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_return_type_and_shape_for_array_input(self, params, x, dtype):
        """Tests that for an array input, the LPDF returns an array with the correct type and shape."""

        shape1, shape2, left_border, right_border = params
        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.dtype == dtype
        assert lpdf_values.shape == x.shape

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(params=st_valid_params(), x=st.floats(-1e6, 1e6))
    def test_lpdf_return_type_and_shape_for_scalar_input(self, params, x, dtype):
        """Tests that for a scalar input, the LPDF returns a scalar with the correct type and shape."""

        alpha, beta, left_border, right_border = params
        dist = Beta(alpha, beta, left_border, right_border, dtype=dtype)
        lpdf_value = dist.lpdf(x)
        assert np.isscalar(lpdf_value)
        assert isinstance(lpdf_value, dtype)

    @given(params=st_valid_params(), x=st.floats(1e-6, 1e6))
    def test_lpdf_against_scipy(self, params, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        shape1, shape2, left_border, right_border = params

        assume(left_border < x < right_border)

        dist = Beta(shape1, shape2, left_border, right_border)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = beta.logpdf(x, shape1, shape2, loc=left_border, scale=right_border - left_border)

        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-5)

    @given(params=st_valid_params(), x=st.floats(max_value=-1e6, allow_infinity=False))
    def test_lpdf_outside_support(self, params, x):
        """Tests that the LPDF is -inf for values less than the location parameter."""

        shape1, shape2, left_border, right_border = params
        x_val = left_border - abs(x)
        dist = Beta(shape1, shape2, left_border, right_border)
        assert dist.lpdf(x_val) == -np.inf


class TestBetaPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(
        params=st_valid_params(), p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True))
    )
    def test_ppf_return_type_and_shape_for_array_input(self, params, p, dtype):
        """Tests that for an array input, the PPDF returns an array with the correct type and shape."""

        shape1, shape2, left_border, right_border = params
        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.dtype == dtype
        assert ppf_values.shape == p.shape

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(params=st_valid_params(), p=st.floats(0, 1, exclude_max=True))
    def test_ppf_return_type_and_shape_for_scalar_input(self, params, p, dtype):
        """Tests that for a scalar input, the PPDF returns a scalar with the correct type and shape."""

        alpha, beta, left_border, right_border = params
        dist = Beta(alpha, beta, left_border, right_border, dtype=dtype)
        ppf_value = dist.ppf(p)
        assert np.isscalar(ppf_value)
        assert isinstance(ppf_value, dtype)

    @given(params=st_valid_params(), p=st.floats(0, 1))
    def test_ppf_against_scipy(self, params, p):
        """Compares the custom PPF implementation against scipy's implementation."""

        shape1, shape2, left_border, right_border = params
        dist = Beta(shape1, shape2, left_border, right_border)
        custom_ppf = dist.ppf(p)
        scipy_ppf = beta.ppf(p, shape1, shape2, loc=left_border, scale=right_border - left_border)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-9)

    @pytest.mark.parametrize("p_val", [-0.5, 1.1, 1.5])
    def test_ppf_invalid_input(self, p_val):
        """Tests that PPF returns NaN for probabilities outside the [0, 1) range."""

        dist = Beta(1.0, 1.0, 1.0, 10.0)
        assert np.isnan(dist.ppf(p_val))


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestBetaGradients:
    """Tests for gradient calculation methods."""

    h = 1e-6

    @given(params_x=st_params_and_array_x_for_grad())
    def test_dlog_shape1_numerical_for_array_input(self, params_x, dtype):
        """Checks the analytical gradient for 'shape1' against a numerical approximation."""

        shape1, shape2, left_border, right_border, x = params_x

        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        analytical_grad = dist._dlog_alpha(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Beta(shape1 + self.h, shape2, left_border, right_border).lpdf(x)
            lpdf_minus_h = Beta(shape1 - self.h, shape2, left_border, right_border).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(params_x=st_params_and_scalar_x_for_grad())
    def test_dlog_shape1_for_scalar_input(self, params_x, dtype):
        """Checks that the gradient for 'shape1' for a scalar input returns scalar."""

        shape1, shape2, left_border, right_border, x = params_x

        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        analytical_grad = dist._dlog_alpha(x)

        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @given(params_x=st_params_and_array_x_for_grad())
    def test_dlog_shape2_numerical_for_array_input(self, params_x, dtype):
        """Checks the analytical gradient for 'shape2' against a numerical approximation."""

        shape1, shape2, left_border, right_border, x = params_x

        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        analytical_grad = dist._dlog_beta(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Beta(shape1, shape2 + self.h, left_border, right_border).lpdf(x)
            lpdf_minus_h = Beta(shape1, shape2 - self.h, left_border, right_border).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(params_x=st_params_and_scalar_x_for_grad())
    def test_dlog_shape2_for_scalar_input(self, params_x, dtype):
        """Checks that the gradient for 'shape2' for a scalar input returns scalar."""

        shape1, shape2, left_border, right_border, x = params_x

        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        analytical_grad = dist._dlog_beta(x)

        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @given(params_x=st_params_and_array_x_for_grad())
    def test_dlog_left_border_numerical_for_array_input(self, params_x, dtype):
        """Checks the analytical gradient for 'left_border' against a numerical approximation."""

        shape1, shape2, left_border, right_border, x = params_x

        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        analytical_grad = dist._dlog_left_border(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Beta(shape1, shape2, left_border + self.h, right_border).lpdf(x)
            lpdf_minus_h = Beta(shape1, shape2, left_border - self.h, right_border).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(params_x=st_params_and_scalar_x_for_grad())
    def test_dlog_left_border_for_scalar_input(self, params_x, dtype):
        """Checks that the gradient for 'left_border' for a scalar input returns scalar."""

        shape1, shape2, left_border, right_border, x = params_x

        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        analytical_grad = dist._dlog_left_border(x)

        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @given(params_x=st_params_and_array_x_for_grad())
    def test_dlog_right_border_numerical_for_array_input(self, params_x, dtype):
        """Checks the analytical gradient for 'right_border' against a numerical approximation."""
        shape1, shape2, left_border, right_border, x = params_x

        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        analytical_grad = dist._dlog_right_border(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Beta(shape1, shape2, left_border, right_border + self.h).lpdf(x)
            lpdf_minus_h = Beta(shape1, shape2, left_border, right_border - self.h).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(params_x=st_params_and_scalar_x_for_grad())
    def test_dlog_right_border_for_scalar_input(self, params_x, dtype):
        """Checks that the gradient for 'right_border' for a scalar input returns scalar."""

        shape1, shape2, left_border, right_border, x = params_x

        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        analytical_grad = dist._dlog_right_border(x)

        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @pytest.mark.parametrize(
        "fixed_params, expected_shape_col, expected_params",
        [
            ([], 4, ["alpha", "beta", "left_border", "right_border"]),
            (["alpha"], 3, ["beta", "left_border", "right_border"]),
            (["alpha", "beta"], 2, ["left_border", "right_border"]),
            (["alpha", "beta", "left_border"], 1, ["right_border"]),
            (["alpha", "beta", "left_border", "right_border"], 0, []),
        ],
    )
    def test_log_gradients_structure(self, fixed_params, expected_shape_col, expected_params, dtype):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Beta(1.0, 1.0, 1.0, 10.0, dtype=dtype)
        for param in fixed_params:
            dist.fix_param(param)

        x = np.array([1.5, 2.0, 3.0])
        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.dtype == dtype
        assert gradients.shape == (len(x), expected_shape_col)

        if "alpha" in expected_params:
            idx = sorted(expected_params).index("alpha")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_alpha(x))
        if "beta" in expected_params:
            idx = sorted(expected_params).index("beta")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_beta(x))
        if "left_border" in expected_params:
            idx = sorted(expected_params).index("left_border")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_left_border(x))
        if "right_border" in expected_params:
            idx = sorted(expected_params).index("right_border")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_right_border(x))

    @given(params_x=st_params_and_scalar_x_for_grad())
    def test_log_gradients_for_scalar_input(self, params_x, dtype):
        """Checks that the log_gradients for a scalar input returns 1D-array."""

        shape1, shape2, left_border, right_border, x = params_x
        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)

        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.dtype == dtype
        assert gradients.ndim == 1


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestBetaGenerate:
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
        dist = Beta(1.0, 1.0, 1.0, 10.0, dtype=dtype)
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
        dist = Beta(1.0, 1.0, 1.0, 10.0, dtype=dtype)
        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self, dtype):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        np.random.seed(123)
        random.seed(123)
        shape1, shape2, left_border, right_border = 1.0, 1.0, 1.0, 10.0
        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        size = 20000

        samples = dist.generate(size=size)

        theoretical_mean = left_border + (right_border - left_border) * shape1 / (shape1 + shape2)

        theoretical_var = (
            (right_border - left_border) ** 2 * shape1 * shape2 / ((shape1 + shape2) ** 2 * (shape1 + shape2 + 1))
        )

        assert np.mean(samples, dtype=np.float64) == pytest.approx(theoretical_mean, rel=0.1)
        assert np.var(samples, dtype=np.float64) == pytest.approx(theoretical_var, rel=0.1)

    def test_generate_kolmogorov_smirnov(self, dtype):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        shape1, shape2, left_border, right_border = 1.0, 1.0, 1.0, 10.0
        dist = Beta(shape1, shape2, left_border, right_border, dtype=dtype)
        size = 1000

        samples = dist.generate(size=size)

        ks_statistic, p_value = kstest(samples, "beta", args=(shape1, shape2, left_border, right_border - left_border))
        lower_bound = 0.05
        assert p_value > lower_bound
