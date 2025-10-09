"""Tests for Beta class"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from rework_pysatl_mpest.distributions.beta import Beta
from scipy.integrate import quad
from scipy.stats import beta, kstest


@st.composite
def st_valid_params(draw):
    """Generates well-behaved beta parameters for integration tests."""
    shape1 = draw(st.floats(min_value=1.0, max_value=50.0))
    shape2 = draw(st.floats(min_value=1.0, max_value=50.0))

    lower_bound = draw(st.floats(min_value=-10.0, max_value=10.0))
    upper_bound = draw(st.floats(min_value=lower_bound + 0.5, max_value=lower_bound + 10.0))

    return shape1, shape2, lower_bound, upper_bound


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
def st_params_and_x_for_grad(draw):
    """Generator of valid params with x for gradient test"""
    shape1 = draw(st.floats(0.1, 100))
    shape2 = draw(st.floats(0.1, 100))
    lower = draw(st.floats(-100, 100))
    upper = draw(st.floats(lower + 0.1, lower + 100))

    margin = 1e-2
    x = draw(
        arrays(
            np.float64,
            shape=draw(st.integers(1, 5)),
            elements=st.floats(
                min_value=lower + margin, max_value=upper - margin, allow_nan=False, allow_infinity=False
            ),
        )
    )
    return (shape1, shape2, lower, upper, x)


class TestBetaInitialization:
    """Tests for the __init__ method and basic properties."""

    def test_initialization_successful(self):
        """Tests that the instance is initialized correctly with valid parameters."""

        shape1, shape2, lower_bound, upper_bound = 0.5, 2.0, -1.0, 1.0
        dist = Beta(shape1=shape1, shape2=shape2, lower_bound=lower_bound, upper_bound=upper_bound)
        assert isinstance(dist.shape1, float)
        assert isinstance(dist.shape2, float)
        assert isinstance(dist.lower_bound, float)
        assert isinstance(dist.upper_bound, float)
        assert dist.shape1 == shape1
        assert dist.shape2 == shape2
        assert dist.lower_bound == lower_bound
        assert dist.upper_bound == upper_bound

    def test_name_property(self):
        """Tests that the name property returns the correct string."""

        dist = Beta(shape1=1.0, shape2=2.0, lower_bound=-1.0, upper_bound=1.0)
        assert dist.name == "Beta"

    def test_params_property(self):
        """Tests that the params property returns the correct set of parameter names."""

        dist = Beta(shape1=1.0, shape2=2.0, lower_bound=-1.0, upper_bound=1.0)
        assert dist.params == {"shape1", "shape2", "lower_bound", "upper_bound"}

    def test_shape1_invariant_violation(self):
        """Tests that initializing with a non-positive shape1 raises a ValueError."""

        with pytest.raises(ValueError, match="Shape1 parameter should be positive or zero"):
            Beta(shape1=-1.0, shape2=2.0, lower_bound=10.0, upper_bound=20.0)
        with pytest.raises(ValueError, match="Shape1 parameter should be positive or zero"):
            Beta(shape1=-20.0, shape2=2.0, lower_bound=10.0, upper_bound=20.0)

    def test_shape1_assignment_violation(self):
        """Tests that assigning a non-positive rate after initialization raises a ValueError."""

        dist = Beta(shape1=1.0, shape2=2.0, lower_bound=10.0, upper_bound=20.0)
        with pytest.raises(ValueError, match="Shape1 parameter should be positive or zero"):
            dist.shape1 = -1.0
        with pytest.raises(ValueError, match="Shape1 parameter should be positive or zero"):
            dist.shape1 = -10.0

    def test_shape2_invariant_violation(self):
        """Tests that initializing with a non-positive shape1 raises a ValueError."""

        with pytest.raises(ValueError, match="Shape2 parameter should be positive or zero"):
            Beta(shape1=1.0, shape2=-2.0, lower_bound=10.0, upper_bound=20.0)
        with pytest.raises(ValueError, match="Shape2 parameter should be positive"):
            Beta(shape1=1.0, shape2=-20.0, lower_bound=10.0, upper_bound=20.0)

    def test_shape2_assignment_violation(self):
        """Tests that assigning a non-positive shape2 after initialization raises a ValueError."""

        dist = Beta(shape1=1.0, shape2=2.0, lower_bound=10.0, upper_bound=20.0)
        with pytest.raises(ValueError, match="Shape2 parameter should be positive or zero"):
            dist.shape2 = -1.0
        with pytest.raises(ValueError, match="Shape2 parameter should be positive or zero"):
            dist.shape2 = -10.0

    def test_invariant_bounds_violation(self):
        """Tests that initializing with a lower bound bigger upper bound  raises a ValueError."""
        with pytest.raises(ValueError, match="Lower bound must be smaller Upper bound"):
            Beta(shape1=1.0, shape2=2.0, lower_bound=10.0, upper_bound=5.0)
        with pytest.raises(ValueError, match="Lower bound must be smaller Upper bound"):
            Beta(shape1=1.0, shape2=2.0, lower_bound=10.0, upper_bound=10.0)

    def test_repr_method(self):
        """Tests that the __repr__ method provides a reproducible string."""

        dist = Beta(shape1=1.0, shape2=2.0, lower_bound=10.0, upper_bound=20.0)
        repr_str = repr(dist)
        assert repr_str == "Beta(shape1=1.0, shape2=2.0, lower_bound=10.0, upper_bound=20.0)"

        recreated_dist = eval(repr_str)
        assert isinstance(recreated_dist, Beta)
        assert recreated_dist.shape1 == dist.shape1
        assert recreated_dist.shape2 == dist.shape2
        assert recreated_dist.lower_bound == dist.lower_bound
        assert recreated_dist.upper_bound == dist.upper_bound


class TestBetaPDF:
    """Tests for the pdf method using hypothesis."""

    @given(x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_pdf_properties(self, x):
        """Tests that the PDF is non-negative and has the correct return type and shape."""
        shape1, shape2, lower_bound, upper_bound = 1.0, 1.0, 2.9, 10.0
        dist = Beta(shape1, shape2, lower_bound, upper_bound)
        pdf_values = dist.pdf(x)
        assert isinstance(pdf_values, np.ndarray)
        assert pdf_values.shape == x.shape
        assert np.all(pdf_values >= 0)

    @pytest.mark.parametrize("x,shape1,shape2,lower_bound,upper_bound,expected_pdf", load_r_test_cases())
    def test_pdf_against_R(self, x, shape1, shape2, lower_bound, upper_bound, expected_pdf):
        """Compares the custom PDF implementation against scipy's implementation."""
        dist = Beta(shape1, shape2, lower_bound, upper_bound)
        custom_pdf = dist.pdf(x)
        np.testing.assert_allclose(custom_pdf, expected_pdf, atol=1e-9)

    @given(params=st_valid_params())
    def test_pdf_integral_is_one(self, params):
        """Tests that the integral of the PDF over its support is equal to 1."""
        shape1, shape2, lower_bound, upper_bound = params
        dist = Beta(shape1, shape2, lower_bound, upper_bound)
        integral, error = quad(dist.pdf, lower_bound, upper_bound)
        np.testing.assert_allclose(1.0, integral, rtol=1e-6, atol=1e-8)

    @given(x=st.floats(min_value=1e-4, max_value=1e2, allow_infinity=False))
    def test_pdf_outside_support(self, x):
        """Tests that the PDF is zero for values less than the location parameter."""
        x_val = 2.0 - abs(x)
        dist = Beta(1.0, 1.0, 2.0, 10.0)
        assert dist.pdf(x_val) == 0.0


class TestBetaLPDF:
    """Tests for the lpdf (log-PDF) method using hypothesis."""

    @given(params=st_valid_params(), x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_return_type_and_shape(self, params, x):
        """Tests the return type and shape of the lpdf method."""
        shape1, shape2, lower_bound, upper_bound = params
        dist = Beta(shape1, shape2, lower_bound, upper_bound)
        lpdf_values = dist.lpdf(x)
        assert isinstance(lpdf_values, np.ndarray)
        assert lpdf_values.shape == x.shape

    @given(params=st_valid_params(), x=st.floats(1e-6, 1e6))
    def test_lpdf_against_scipy(self, params, x):
        """Compares the custom LPDF implementation against scipy's implementation."""

        shape1, shape2, lower_bound, upper_bound = params
        dist = Beta(shape1, shape2, lower_bound, upper_bound)
        custom_lpdf = dist.lpdf(x)
        scipy_lpdf = beta.logpdf(x, shape1, shape2, loc=lower_bound, scale=upper_bound - lower_bound)

        np.testing.assert_allclose(custom_lpdf, scipy_lpdf, atol=1e-5)

    @given(params=st_valid_params(), x=st.floats(max_value=-1e6, allow_infinity=False))
    def test_lpdf_outside_support(self, params, x):
        """Tests that the LPDF is -inf for values less than the location parameter."""
        shape1, shape2, lower_bound, upper_bound = params
        x_val = lower_bound - abs(x)
        dist = Beta(shape1, shape2, lower_bound, upper_bound)
        assert dist.lpdf(x_val) == -np.inf


class TestBetaPPF:
    """Tests for the ppf (Percent Point Function) method using hypothesis."""

    @given(
        params=st_valid_params(), p=arrays(np.float64, st.integers(0, 10), elements=st.floats(0, 1, exclude_max=True))
    )
    def test_ppf_return_type_and_shape(self, params, p):
        """Tests the return type and shape of the ppf method."""
        shape1, shape2, lower_bound, upper_bound = params
        dist = Beta(shape1, shape2, lower_bound, upper_bound)
        ppf_values = dist.ppf(p)
        assert isinstance(ppf_values, np.ndarray)
        assert ppf_values.shape == p.shape

    @given(params=st_valid_params(), p=st.floats(0, 1))
    def test_ppf_against_scipy(self, params, p):
        """Compares the custom PPF implementation against scipy's implementation."""
        shape1, shape2, lower_bound, upper_bound = params
        dist = Beta(shape1, shape2, lower_bound, upper_bound)
        custom_ppf = dist.ppf(p)
        scipy_ppf = beta.ppf(p, shape1, shape2, loc=lower_bound, scale=upper_bound - lower_bound)
        np.testing.assert_allclose(custom_ppf, scipy_ppf, atol=1e-9)

    @pytest.mark.parametrize("p_val", [-0.5, 1.1, 1.5])
    def test_ppf_invalid_input(self, p_val):
        """Tests that PPF returns NaN for probabilities outside the [0, 1) range."""

        dist = Beta(1.0, 1.0, 1.0, 10.0)
        assert np.isnan(dist.ppf(p_val))


class TestBetaGradients:
    """Tests for gradient calculation methods."""

    h = 1e-6

    @given(params_x=st_params_and_x_for_grad())
    def test_dlog_shape1_numerical(self, params_x):
        """Checks the analytical gradient for 'shape1' against a numerical approximation."""
        shape1, shape2, lower_bound, upper_bound, x = params_x

        dist = Beta(shape1, shape2, lower_bound, upper_bound)

        lpdf_plus_h = Beta(shape1 + self.h, shape2, lower_bound, upper_bound).lpdf(x)
        lpdf_minus_h = Beta(shape1 - self.h, shape2, lower_bound, upper_bound).lpdf(x)

        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        analytical_grad = dist._dlog_shape1(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.shape == x.shape
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(params_x=st_params_and_x_for_grad())
    def test_dlog_shape2_numerical(self, params_x):
        """Checks the analytical gradient for 'shape2' against a numerical approximation."""
        shape1, shape2, lower_bound, upper_bound, x = params_x

        dist = Beta(shape1, shape2, lower_bound, upper_bound)

        lpdf_plus_h = Beta(shape1, shape2 + self.h, lower_bound, upper_bound).lpdf(x)
        lpdf_minus_h = Beta(shape1, shape2 - self.h, lower_bound, upper_bound).lpdf(x)

        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        analytical_grad = dist._dlog_shape2(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.shape == x.shape
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(params_x=st_params_and_x_for_grad())
    def test_dlog_lower_bound_numerical(self, params_x):
        """Checks the analytical gradient for 'lower_bound' against a numerical approximation."""
        shape1, shape2, lower_bound, upper_bound, x = params_x

        dist = Beta(shape1, shape2, lower_bound, upper_bound)

        lpdf_plus_h = Beta(shape1, shape2, lower_bound + self.h, upper_bound).lpdf(x)
        lpdf_minus_h = Beta(shape1, shape2, lower_bound - self.h, upper_bound).lpdf(x)

        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        analytical_grad = dist._dlog_lower_bound(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.shape == x.shape
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @given(params_x=st_params_and_x_for_grad())
    def test_dlog_upper_bound_numerical(self, params_x):
        """Checks the analytical gradient for 'upper_bound' against a numerical approximation."""
        shape1, shape2, lower_bound, upper_bound, x = params_x

        dist = Beta(shape1, shape2, lower_bound, upper_bound)

        lpdf_plus_h = Beta(shape1, shape2, lower_bound, upper_bound + self.h).lpdf(x)
        lpdf_minus_h = Beta(shape1, shape2, lower_bound, upper_bound - self.h).lpdf(x)

        numerical_grad = (lpdf_plus_h - lpdf_minus_h) / (2 * self.h)
        analytical_grad = dist._dlog_upper_bound(x)

        assert isinstance(analytical_grad, np.ndarray)
        assert analytical_grad.shape == x.shape
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize(
        "fixed_params, expected_shape_col, expected_params",
        [
            ([], 4, ["shape1", "shape2", "lower_bound", "upper_bound"]),
            (["shape1"], 3, ["shape2", "lower_bound", "upper_bound"]),
            (["shape1", "shape2"], 2, ["lower_bound", "upper_bound"]),
            (["shape1", "shape2", "lower_bound"], 1, ["upper_bound"]),
            (["shape1", "shape2", "lower_bound", "upper_bound"], 0, []),
        ],
    )
    def test_log_gradients_structure(self, fixed_params, expected_shape_col, expected_params):
        """Tests the structure and content of log_gradients with various fixed parameters."""

        dist = Beta(1.0, 1.0, 1.0, 10.0)
        for param in fixed_params:
            dist.fix_param(param)

        x = np.array([1.5, 2.0, 3.0])
        gradients = dist.log_gradients(x)

        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == (len(x), expected_shape_col)

        if "shape1" in expected_params:
            idx = sorted(expected_params).index("shape1")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_shape1(x))
        if "shape2" in expected_params:
            idx = sorted(expected_params).index("shape2")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_shape2(x))
        if "lower_bound" in expected_params:
            idx = sorted(expected_params).index("lower_bound")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_lower_bound(x))
        if "upper_bound" in expected_params:
            idx = sorted(expected_params).index("upper_bound")
            np.testing.assert_allclose(gradients[:, idx], dist._dlog_upper_bound(x))


class TestBetaGenerate:
    """Tests for the generate method."""

    def test_generate_type_and_shape(self):
        """Tests that generated samples have the correct type and shape."""

        np.random.seed(42)
        random.seed(42)
        dist = Beta(1.0, 1.0, 1.0, 10.0)
        size = 100
        samples = dist.generate(size=size)
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float64
        assert samples.shape == (size,)

    def test_generate_zero_size(self):
        """Tests if the generating 0 number of samples returns an empty array"""

        dist = Beta(1.0, 1.0, 1.0, 10.0)
        assert len(dist.generate(size=0)) == 0

    @pytest.mark.parametrize("size", [-1, -10])
    def test_generate_negative_size(self, size):
        """Tests that generating a negative number of samples raises ValueError."""

        dist = Beta(1.0, 1.0, 1.0, 10.0)

        with pytest.raises(ValueError):
            dist.generate(size=size)

    def test_generate_statistical_properties(self):
        """Tests if the generated samples have correct statistical properties (mean, variance)."""

        np.random.seed(123)
        random.seed(123)
        shape1, shape2, lower_bound, upper_bound = 1.0, 1.0, 1.0, 10.0
        dist = Beta(shape1, shape2, lower_bound, upper_bound)
        size = 20000

        samples = dist.generate(size=size)

        theoretical_mean = lower_bound + (upper_bound - lower_bound) * shape1 / (shape1 + shape2)

        theoretical_var = (
            (upper_bound - lower_bound) ** 2 * shape1 * shape2 / ((shape1 + shape2) ** 2 * (shape1 + shape2 + 1))
        )

        assert np.mean(samples) == pytest.approx(theoretical_mean, rel=0.1)
        assert np.var(samples) == pytest.approx(theoretical_var, rel=0.1)

    def test_generate_kolmogorov_smirnov(self):
        """Performs a Kolmogorov-Smirnov test to check if samples fit the distribution."""

        np.random.seed(456)
        random.seed(456)
        shape1, shape2, lower_bound, upper_bound = 1.0, 1.0, 1.0, 10.0
        dist = Beta(shape1, shape2, lower_bound, upper_bound)
        size = 1000

        samples = dist.generate(size=size)

        ks_statistic, p_value = kstest(samples, "beta", args=(shape1, shape2, lower_bound, upper_bound - lower_bound))
        lower_bound = 0.05
        assert p_value > lower_bound
