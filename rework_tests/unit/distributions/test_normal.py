"""Tests for Normal class"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import random
from typing import ClassVar

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from rework_pysatl_mpest.distributions import Normal
from rework_tests.unit.distributions.test_continuous_distribution import DTypeHandlingMixin
from scipy.integrate import quad
from scipy.stats import kstest, norm

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]

# Strategies for hypothesis
st_loc = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
st_scale = st.floats(min_value=0.01, max_value=1e3, allow_nan=False, allow_infinity=False)


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestNormalInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self, dtype):
        """Tests that the instance is initialized correctly with valid parameters."""

        loc, scale = 10.0, 2.5
        dist = Normal(loc=loc, scale=scale, dtype=dtype)
        assert dist.loc.dtype == dtype
        assert dist.scale.dtype == dtype
        assert dist.loc == dtype(loc)
        assert dist.scale == dtype(scale)

    def test_name_property(self, dtype):
        """Tests that the name property returns the correct string."""

        dist = Normal(loc=0.0, scale=1.0, dtype=dtype)
        assert dist.name == "Normal"

    def test_params_property(self, dtype):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Normal(loc=0.0, scale=1.0, dtype=dtype)
        assert dist.params == {"loc", "scale"}

    def test_scale_invariant_violation(self, dtype):
        """Tests that initializing with a non-positive scale raises a ValueError."""

        with pytest.raises(ValueError, match="Scale parameter must be positive"):
            Normal(loc=0.0, scale=0.0, dtype=dtype)
        with pytest.raises(ValueError, match="Scale parameter must be positive"):
            Normal(loc=0.0, scale=-1.0, dtype=dtype)

    def test_scale_assignment_violation(self, dtype):
        """Tests that assigning a non-positive rate after initialization raises a ValueError."""

        dist = Normal(loc=0.0, scale=1.0, dtype=dtype)
        with pytest.raises(ValueError, match="Scale parameter must be positive"):
            dist.scale = 0.0
        with pytest.raises(ValueError, match="Scale parameter must be positive"):
            dist.scale = -10.0

    def test_repr_method(self, dtype):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Normal(loc=1.23, scale=4.56, dtype=dtype)
        repr_str = repr(dist)
        assert repr_str == f"Normal(loc={dist.loc}, scale={dist.scale}, dtype=np.{dtype.__name__})"

        recreated_dist = eval(repr_str)
        assert dist == recreated_dist


class TestNormalPDF:
    """Tests for the pdf method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_pdf_properties(self, loc, scale, x, dtype):
        """Tests that the PDF is non-negative and has the correct return type and shape."""

        dist = Normal(loc=loc, scale=scale, dtype=dtype)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.dtype == dtype
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

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
        integral, error = quad(lambda x: dist.pdf(x).item(), loc - scale * 6, loc + scale * 6)
        np.testing.assert_allclose(1.0, integral, atol=1e-7)


class TestNormalLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_return_type_and_shape(self, loc, scale, x, dtype):
        """Tests the return type and shape of the lpdf method."""

        dist = Normal(loc=loc, scale=scale, dtype=dtype)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.dtype == dtype
        assert lpdf_values.shape == x.shape

    @given(loc=st_loc, scale=st_scale, x=st.floats(-1e6, 1e6))
    def test_lpdf_against_scipy(self, loc, scale, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        dist = Normal(loc=loc, scale=scale)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = norm.logpdf(x, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-9)


class TestNormalPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(loc=st_loc, scale=st_scale, p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1)))
    def test_ppf_return_type_and_shape(self, loc, scale, p, dtype):
        """Tests the return type and shape of the ppf method."""

        dist = Normal(loc=loc, scale=scale, dtype=dtype)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.dtype == dtype
        assert ppf_values.shape == p.shape

    @given(loc=st_loc, scale=st_scale, p=st.floats(1e-6, 1.0 - 1e-6))
    def test_ppf_against_scipy(self, loc, scale, p):
        """Compares the custom PPF implementation against scipy's implementation."""

        dist = Normal(loc=loc, scale=scale)
        custom_ppf = dist.ppf(p)
        scipy_ppf = norm.ppf(p, loc=loc, scale=scale)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-9)


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestNormalGradients:
    """Tests for gradient calculation methods."""

    h = 1e-6

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(-1e3, 1e3)))
    def test_dlog_loc_numerical(self, loc, scale, x, dtype):
        """Checks the analytical gradient for 'loc' against a numerical approximation."""

        dist = Normal(loc, scale, dtype=dtype)
        analytical_grad = dist._dlog_loc(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Normal(loc + self.h, scale).lpdf(x)
            lpdf_minus_h = Normal(loc - self.h, scale).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(loc=st_loc, scale=st_scale, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(-1e3, 1e3)))
    def test_dlog_scale_numerical(self, loc, scale, x, dtype):
        """Checks the analytical gradient for 'scale' against a numerical approximation."""

        dist = Normal(loc, scale, dtype=dtype)
        analytical_grad = dist._dlog_scale(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Normal(loc, scale + self.h).lpdf(x)
            lpdf_minus_h = Normal(loc, scale - self.h).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "fixed_params, expected_cols, expected_params",
        [([], 2, ["loc", "scale"]), (["loc"], 1, ["scale"]), (["scale"], 1, ["loc"]), (["loc", "scale"], 0, [])],
    )
    def test_log_gradients_structure(self, fixed_params, expected_cols, expected_params, dtype):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Normal(loc=1.0, scale=2.0, dtype=dtype)
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


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestNormalGenerate:
    """Tests for the generate method."""

    def test_generate_type_and_shape(self, dtype):
        """Tests that generated samples have the correct type and shape."""

        dist = Normal(loc=0.0, scale=1.0, dtype=dtype)
        samples = dist.generate(size=100)
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == dtype
        assert samples.shape == (100,)

    def test_generate_zero_size(self, dtype):
        """Tests if the generating 0 number of samples returns an empty array"""

        dist = Normal(loc=0.0, scale=1.0, dtype=dtype)
        assert len(dist.generate(size=0)) == 0

    @pytest.mark.parametrize("size", [-1, -10])
    def test_generate_negative_size(self, size, dtype):
        """Tests that generating a negative number of samples raises ValueError."""

        dist = Normal(loc=0.0, scale=1.0, dtype=dtype)

        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self, dtype):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        np.random.seed(123)
        random.seed(123)
        loc, scale = 15.0, 3.0
        dist = Normal(loc=loc, scale=scale, dtype=dtype)
        size = 50000

        samples = dist.generate(size=size)
        assert np.mean(samples, dtype=np.float64) == pytest.approx(loc, rel=0.05)
        assert np.var(samples, dtype=np.float64) == pytest.approx(scale**2, rel=0.05)

    def test_generate_kolmogorov_smirnov(self, dtype):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        loc, scale = -10.0, 5.0
        dist = Normal(loc=loc, scale=scale, dtype=dtype)
        size = 1000

        samples = dist.generate(size=size)
        ks_statistic, p_value = kstest(samples, "norm", args=(loc, scale))
        expected_p_value = 0.05
        assert p_value > expected_p_value


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestNormalDType(DTypeHandlingMixin):
    distribution_class = Normal
    default_params: ClassVar[dict] = {"loc": 0.0, "scale": 1.0}

    def test_init_with_dtype_sets_correct_types(self, dtype):
        self.check_init_with_dtype_sets_correct_types(dtype)

    @pytest.mark.parametrize("size", [0, 10])
    def test_generate_returns_correct_dtype(self, size, dtype):
        self.check_generate_returns_correct_dtype(size, dtype)

    @pytest.mark.parametrize("method_name", ["pdf", "lpdf", "log_gradients"])
    @given(x_data=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e3, 1e3)))
    def check_methods_taking_x_return_correct_dtype(self, method_name, x_data, dtype):
        self.check_methods_taking_x_return_correct_dtype(method_name, x_data, dtype)

    @given(p_data=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True)))
    def check_ppf_returns_correct_dtype(self, p_data, dtype):
        self.check_ppf_returns_correct_dtype(p_data, dtype)

    @pytest.mark.parametrize("method_name", ["_dlog_loc", "_dlog_scale"])
    @given(x_data=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_dlog_methods_returns_correct_dtype(self, x_data, method_name, dtype):
        self.check_dlog_methods_returns_correct_dtype(x_data, method_name, dtype)
