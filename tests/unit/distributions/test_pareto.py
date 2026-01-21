"""Tests for Pareto class"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pysatl_mpest.distributions.pareto import Pareto
from scipy.integrate import quad
from scipy.stats import kstest, pareto

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]

st_shape = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
st_scale = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)


def load_r_test_cases():
    """Loads constants calculated in R with VGAM::dpareto"""
    csv_path = Path(__file__).parent / "constraints" / "pareto_type_1_test.csv"
    data = pd.read_csv(csv_path)

    id_col = data.columns[0]
    param_cols = data.columns[1:]

    cases = []
    for _, row in data.iterrows():
        params = [np.float64(row[col]) for col in param_cols]
        cases.append(pytest.param(*params, id=str(row[id_col])))

    return cases


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestParetoInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self, dtype):
        """Tests that the instance is initialized correctly with valid parameters."""

        shape, scale = 0.5, 2.0
        dist = Pareto(shape=shape, scale=scale, dtype=dtype)
        assert dist.shape.dtype == dtype
        assert dist.scale.dtype == dtype
        assert dist.shape == dtype(shape)
        assert dist.scale == dtype(scale)

    def test_name_property(self, dtype):
        """Tests that the name property returns the correct string."""

        dist = Pareto(shape=1.0, scale=2.0, dtype=dtype)
        assert dist.name == "Pareto"

    def test_params_property(self, dtype):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Pareto(shape=1.0, scale=1.0, dtype=dtype)
        assert dist.params == {"shape", "scale"}

    def test_shape_invariant_violation(self, dtype):
        """Tests that initializing with a non-positive shape raises a ValueError."""

        with pytest.raises(ValueError, match="Shape parameter must be a positive"):
            Pareto(shape=0.0, scale=1.0, dtype=dtype)
        with pytest.raises(ValueError, match="Shape parameter must be a positive"):
            Pareto(shape=-1.0, scale=1.0, dtype=dtype)

    def test_scale_invariant_violation(self, dtype):
        """Tests that initializing with a non-positive scale raises a ValueError."""

        with pytest.raises(ValueError, match="Scale parameter must be a positive"):
            Pareto(shape=1.0, scale=0.0, dtype=dtype)
        with pytest.raises(ValueError, match="Scale parameter must be a positive"):
            Pareto(shape=1.0, scale=-1.0, dtype=dtype)

    def test_shape_assignment_violation(self, dtype):
        """Tests that assigning a non-positive shape after initialization raises a ValueError."""

        dist = Pareto(shape=1.0, scale=1.0, dtype=dtype)
        with pytest.raises(ValueError, match="Shape parameter must be a positive"):
            dist.shape = 0.0
        with pytest.raises(ValueError, match="Shape parameter must be a positive"):
            dist.shape = -10.0

    def test_scale_assignment_violation(self, dtype):
        """Tests that assigning a non-positive scale after initialization raises a ValueError."""

        dist = Pareto(shape=1.0, scale=1.0, dtype=dtype)
        with pytest.raises(ValueError, match="Scale parameter must be a positive"):
            dist.scale = 0.0
        with pytest.raises(ValueError, match="Scale parameter must be a positive"):
            dist.scale = -10.0

    def test_repr_method(self, dtype):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Pareto(shape=1.23, scale=4.56, dtype=dtype)
        repr_str = repr(dist)
        assert repr_str == f"Pareto(shape={dist.shape}, scale={dist.scale}, dtype=np.{dtype.__name__})"

        recreated_dist = eval(repr_str)
        assert dist == recreated_dist


class TestParetoPDF:
    """Tests for the pdf method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e2, 1e2)))
    def test_pdf_properties_for_array_input(self, x, dtype):
        """Tests that for an array input, the PDF returns a non-negative array with the correct type and shape."""

        dist = Pareto(shape=1.0, scale=2.0, dtype=dtype)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.dtype == dtype
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(x=st.floats(-1e2, 1e2))
    def test_pdf_properties_for_scalar_input(self, x, dtype):
        """Tests that for a scalar input, the PDF returns a non-negative scalar with the correct type."""

        shape, scale = 1.0, 2.0
        dist = Pareto(shape, scale, dtype=dtype)
        pdf_value = dist.pdf(x)
        assert np.isscalar(pdf_value)
        assert isinstance(pdf_value, dtype)
        assert pdf_value >= 0

    @pytest.mark.parametrize("x,shape,scale,expected_pdf", load_r_test_cases())
    def test_pdf_against_R(self, shape, scale, x, expected_pdf):
        """Compares the custom PDF implementation against scipy's implementation."""

        dist = Pareto(shape=shape, scale=scale)
        custom_pdf = dist.pdf(x)

        np.testing.assert_allclose(expected_pdf, custom_pdf, atol=1e-8)

    @pytest.mark.parametrize(
        "shape,scale",
        [(2.0, 1.0), (1.0, 1.0), (0.25, 1.0), (100.0, 3.0), (1.7, 0.8), (4.05854031201452, 17.7524529929971)],
    )
    def test_pdf_integral_is_one(self, shape, scale):
        """Tests that the integral of the PDF over its support is equal to 1."""

        dist = Pareto(shape=shape, scale=scale)
        integral, error = quad(lambda x: dist.pdf(x).item(), scale, np.inf, epsabs=1e-10, epsrel=1e-10, limit=100)
        assert np.isfinite(integral), f"Integral diverged: {integral}"
        np.testing.assert_allclose(1.0, integral, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize(
        "shape,scale, x",
        [
            (2.0, 1.0, 100.0),
            (1.0, 1.0, 1000.0),
            (0.25, 1.0, 1.0),
            (100.0, 3.0, -1.0),
            (1.7, 0.8, -8.0),
            (4.05854031201452, 17.7524529929971, 10.0),
        ],
    )
    def test_pdf_outside_support(self, shape, scale, x):
        """Tests that the PDF is zero for values less than the location parameter."""
        x = np.asarray(x, dtype=np.float64)
        x_val = scale - abs(x)
        dist = Pareto(shape=shape, scale=scale)
        assert dist.pdf(x_val) == 0.0


class TestParetoLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(shape=st_shape, scale=st_scale, x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_return_type_and_shape_for_array_input(self, shape, scale, x, dtype):
        """Tests the return type and shape of the lpdf method for array input."""

        dist = Pareto(shape=shape, scale=scale, dtype=dtype)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.dtype == dtype
        assert lpdf_values.shape == x.shape

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(shape=st_shape, scale=st_scale, x=st.floats(-1e6, 1e6))
    def test_lpdf_return_type_and_shape_for_scalar_input(self, shape, scale, x, dtype):
        """Tests the return type and shape of the lpdf method for scalar input."""

        dist = Pareto(shape=shape, scale=scale, dtype=dtype)
        lpdf_value = dist.lpdf(x)
        assert np.isscalar(lpdf_value)
        assert isinstance(lpdf_value, dtype)

    @given(shape=st_shape, scale=st_scale, x=st.floats(1e-3, 1e3, allow_infinity=False, allow_nan=False))
    def test_lpdf_against_scipy(self, shape, scale, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        assume(np.isfinite(pareto.logpdf(x, scale=scale, b=shape, loc=0.0)))
        dist = Pareto(shape=shape, scale=scale)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = pareto.logpdf(x, scale=scale, b=shape, loc=0.0)
        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-12, rtol=1e-3)

    @given(shape=st_shape, scale=st_scale, x=st.floats(min_value=1e2, max_value=1e4, allow_infinity=False))
    def test_lpdf_outside_support(self, shape, scale, x):
        """Tests that the LPDF is -inf for values less than the location parameter."""

        x_val = scale - x
        dist = Pareto(shape=shape, scale=scale)
        assert dist.lpdf(x_val) == -np.inf


class TestParetoPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(
        shape=st_shape,
        scale=st_scale,
        p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True)),
    )
    def test_ppf_return_type_and_shape_for_array_input(self, shape, scale, p, dtype):
        """Tests the return type and shape of the ppf method for array input."""

        dist = Pareto(shape=shape, scale=scale, dtype=dtype)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.dtype == dtype
        assert ppf_values.shape == p.shape

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    @given(shape=st_shape, scale=st_scale, p=st.floats(0, 1, exclude_max=True))
    def test_ppf_return_type_and_shape_for_scalar_input(self, shape, scale, p, dtype):
        """Tests the return type and shape of the ppf method for scalar input."""

        dist = Pareto(shape=shape, scale=scale, dtype=dtype)
        ppf_value = dist.ppf(p)
        assert np.isscalar(ppf_value)
        assert isinstance(ppf_value, dtype)

    @given(shape=st_shape, scale=st_scale, p=st.floats(0, 1))
    def test_ppf_against_scipy(self, shape, scale, p):
        """Compares the custom PPF implementation against scipy's implementation."""

        dist = Pareto(shape=shape, scale=scale)
        custom_ppf = dist.ppf(p)
        scipy_ppf = pareto.ppf(p, scale=scale, b=shape)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-9)

    @pytest.mark.parametrize("p_val", [-0.5, 1.1, 1.5])
    def test_ppf_invalid_input(self, p_val):
        """Tests that PPF returns NaN for probabilities outside the [0, 1) range."""

        dist = Pareto(shape=1.0, scale=5.0)
        assert np.isnan(dist.ppf(p_val))


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestParetoGradients:
    """Tests for gradient calculation methods."""

    h = 1e-6

    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(shape=st_shape, scale=st_scale, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)))
    def test_dlog_shape_numerical_for_array_input(self, shape, scale, x, dtype):
        """Checks the analytical gradient for 'shape' against a numerical approximation for array input."""

        assume(np.all(x > scale))

        dist = Pareto(shape=shape, scale=scale, dtype=dtype)
        analytical_grad = dist._dlog_shape(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Pareto(shape=shape + self.h, scale=scale).lpdf(x)
            lpdf_minus_h = Pareto(shape=shape - self.h, scale=scale).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(shape=st_shape, scale=st_scale, x=st.floats(1e-3, 1e3))
    def test_dlog_shape_for_scalar_input(self, shape, scale, x, dtype):
        """Checks that the gradient for 'shape' for a scalar input returns a scalar."""

        assume(x > scale)
        dist = Pareto(shape, scale, dtype=dtype)
        analytical_grad = dist._dlog_shape(x)
        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(shape=st_shape, scale=st_scale, x=arrays(np.float64, st.integers(1, 10), elements=st.floats(1e-3, 1e3)))
    def test_dlog_scale_numerical_for_array_input(self, shape, scale, x, dtype):
        """Checks the analytical gradient for 'scale' against a numerical approximation for array input."""

        assume(np.all(x > scale + self.h))

        dist = Pareto(shape=shape, scale=scale, dtype=dtype)
        analytical_grad = dist._dlog_scale(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.dtype == dtype
        assert analytical_grad.shape == x.shape

        if dtype == np.float64:
            lpdf_plus_h = Pareto(shape=shape, scale=scale + self.h).lpdf(x)
            lpdf_minus_h = Pareto(shape=shape, scale=scale - self.h).lpdf(x)

            numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
            np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-3, rtol=1e-3)

    @given(shape=st_shape, scale=st_scale, x=st.floats(1e-3, 1e3))
    def test_dlog_scale_for_scalar_input(self, shape, scale, x, dtype):
        """Checks that the gradient for 'scale' for a scalar input returns a scalar."""

        assume(x > scale + self.h)
        dist = Pareto(shape, scale, dtype=dtype)
        analytical_grad = dist._dlog_scale(x)
        assert np.isscalar(analytical_grad)
        assert isinstance(analytical_grad, dtype)

    @pytest.mark.parametrize(
        "fixed_params, expected_shape_col, expected_params",
        [
            ([], 2, ["shape", "scale"]),
            (["shape"], 1, ["scale"]),
            (["scale"], 1, ["shape"]),
            (["shape", "scale"], 0, []),
        ],
    )
    def test_log_gradients_structure(self, fixed_params, expected_shape_col, expected_params, dtype):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Pareto(shape=1.0, scale=2.0, dtype=dtype)
        for param in fixed_params:
            dist.fix_param(param)

        x = np.array([1.5, 2.0, 3.0])
        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.dtype == dtype
        assert gradients.shape == (len(x), expected_shape_col)

        if "shape" in expected_params:
            idx = sorted(expected_params).index("shape")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_shape(x))
        if "scale" in expected_params:
            idx = sorted(expected_params).index("scale")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_scale(x))

    @given(shape=st_shape, scale=st_scale, x=st.floats(1e-3, 1e3))
    def test_log_gradients_for_scalar_input(self, shape, scale, x, dtype):
        """Checks that the log_gradients for a scalar input returns a 1D-array."""

        assume(x > scale + self.h)
        dist = Pareto(shape, scale, dtype=dtype)
        gradients = dist.log_gradients(x)
        assert isinstance(gradients, np.ndarray)
        assert gradients.dtype == dtype
        assert gradients.ndim == 1


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestParetoGenerate:
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
        dist = Pareto(shape=1.0, scale=2.0, dtype=dtype)
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

        dist = Pareto(shape=1.0, scale=2.0, dtype=dtype)

        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self, dtype):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        np.random.seed(123)
        random.seed(123)
        shape, scale = 5.0, 0.5
        dist = Pareto(shape=shape, scale=scale, dtype=dtype)
        size = 20000

        samples = dist.generate(size=size)

        theoretical_mean = (shape * scale) / (shape - 1)
        theoretical_var = ((scale**2) * shape) / ((shape - 1) ** 2 * (shape - 2))

        assert np.mean(samples, dtype=np.float64) == pytest.approx(theoretical_mean, rel=0.1)
        assert np.var(samples, dtype=np.float64) == pytest.approx(theoretical_var, rel=0.1)

    def test_generate_kolmogorov_smirnov(self, dtype):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        shape, scale = 10.0, 2.0
        dist = Pareto(shape=shape, scale=scale, dtype=dtype)
        size = 1000

        samples = dist.generate(size=size)

        ks_statistic, p_value = kstest(samples, "pareto", args=(shape, 0, scale))
        lower_bound = 0.05
        assert p_value > lower_bound
