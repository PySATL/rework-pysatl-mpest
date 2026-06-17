"""Tests for L-Moments optimization strategy for Exponential distribution"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pysatl_mpest.distributions import Exponential
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from pysatl_mpest.estimators.iterative._strategies import lmoments_strategy
from pysatl_mpest.exceptions import NumericalStabilityError

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]

# Test Fixtures
# -------------


@pytest.fixture(params=DTYPES_TO_TEST)
def parametrized_exponential_setup(request) -> tuple[Exponential, PipelineState, np.floating]:
    """
    Creates a parametrized fixture providing an Exponential component and a
    corresponding PipelineState for various dtypes.
    """
    dtype = request.param

    component = Exponential(loc=0.0, rate=1.0, dtype=dtype)

    state = PipelineState(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype),
        H=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=dtype),
        prev_mixture=None,
        curr_mixture=None,
        error=None,
    )
    return component, state, dtype


# Tests
# -----


def test_lmoments_exponential_raises_value_error_if_h_is_none(parametrized_exponential_setup):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed.
    """
    exponential_component, state, _ = parametrized_exponential_setup
    state.H = None

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "rate"}, maximization_strategy=MaximizationStrategy.LMOMENTS
    )

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        lmoments_strategy(exponential_component, state, block, optimizer=None)


def test_lmoments_exponential_returns_correct_types(parametrized_exponential_setup):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, DType]).
    """
    exponential_component, pipeline_state, dtype = parametrized_exponential_setup

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "rate"}, maximization_strategy=MaximizationStrategy.LMOMENTS
    )

    result = lmoments_strategy(exponential_component, pipeline_state, block, optimizer=None)

    assert isinstance(result, tuple)
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, dtype)


@pytest.mark.parametrize(
    "params_to_optimize_in_block, fixed_params_on_component, expected_keys",
    [
        ({"loc", "rate"}, set(), {"loc", "rate"}),  # Optimize both, none are fixed
        ({"loc"}, set(), {"loc"}),  # Optimize only loc
        ({"rate"}, set(), {"rate"}),  # Optimize only rate
        ({"loc", "rate"}, {"loc"}, {"rate"}),  # Optimize both, but loc is fixed
        ({"loc", "rate"}, {"rate"}, {"loc"}),  # Optimize both, but rate is fixed
        ({"loc", "rate"}, {"loc", "rate"}, set()),  # Optimize both, but both are fixed
        ({"non_existent_param", "loc"}, set(), {"loc"}),  # Ignore non-existent params
        (set(), set(), set()),  # Optimize nothing
    ],
)
def test_lmoments_exponential_respects_fixed_and_optimizable_params(
    parametrized_exponential_setup, params_to_optimize_in_block, fixed_params_on_component, expected_keys
):
    """
    Verifies that the strategy correctly identifies which parameters
    to update based on the optimization block and the component's fixed parameters.
    """
    exponential_component, pipeline_state, _ = parametrized_exponential_setup

    for param in fixed_params_on_component:
        exponential_component.fix_param(param)

    block = OptimizationBlock(
        component_id=0,
        params_to_optimize=params_to_optimize_in_block,
        maximization_strategy=MaximizationStrategy.LMOMENTS,
    )

    _, new_params = lmoments_strategy(exponential_component, pipeline_state, block, optimizer=None)

    assert set(new_params.keys()) == expected_keys


def test_lmoments_exponential_handles_negligible_responsibility(parametrized_exponential_setup):
    """
    Verifies that if a component's total responsibility (N_j) is near zero,
    its parameters are not updated.
    """

    exponential_component, pipeline_state, dtype = parametrized_exponential_setup

    pipeline_state.H.fill(dtype(1e-10))  # Make all responsibilities negligible
    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={"loc", "rate"},
        maximization_strategy=MaximizationStrategy.LMOMENTS,
    )

    _, new_params = lmoments_strategy(exponential_component, pipeline_state, block, optimizer=None)

    assert new_params == {}


def test_lmoments_exponential_rate_fallback_when_mean_equals_loc(parametrized_exponential_setup):
    """
    Tests the edge case where the weighted mean of the data equals the component's location.

    In the formula `rate = 1 / (mean - loc)`, if `mean == loc`, a division by zero would occur.
    The code should handle this by keeping the component's original rate.
    """
    component, state, dtype = parametrized_exponential_setup

    loc_val = component.loc
    state.X = np.array([loc_val, loc_val], dtype=dtype)
    state.H = np.array([[1.0], [1.0]], dtype=dtype)

    component.fix_param("loc")
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"rate"}, maximization_strategy=MaximizationStrategy.LMOMENTS
    )

    _, new_params = lmoments_strategy(component, state, block, optimizer=None)

    assert Exponential.PARAM_RATE in new_params
    assert new_params[Exponential.PARAM_RATE] == component.rate


@pytest.mark.parametrize(
    "x_val, n_samples",
    [
        # Case 1: Overflow during first lmoment calculation (l1)
        pytest.param(65504.0, 2, id="overflow_first_lmoment"),
        # Case 2: Overflow ONLY during second lmoment calculation (l2)
        pytest.param(300.0, 220, id="overflow_second_lmoment"),
    ],
    ids=["overflow_first_lmoment", "overflow_second_lmoment"],
)
def test_lmoments_exponential_overflow_handling(parametrized_exponential_setup, x_val, n_samples, request):
    """
    Verifies that NumericalStabilityError is correctly set in state
    upon overflow in either the first moment or the second moment calculation.
    """
    # --- Arrange ---
    dtype = np.float16

    component = Exponential(loc=0.0, rate=1.0, dtype=dtype)

    if request.node.callspec.id == "overflow_first_lmoment":
        X_data = np.full(n_samples, x_val, dtype=dtype)
    else:
        X_data = np.array([x_val * (i + 1) for i in range(n_samples)], dtype=dtype)

    H = np.ones((n_samples, 1), dtype=dtype)

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "rate"}, maximization_strategy=MaximizationStrategy.LMOMENTS
    )

    state = PipelineState(X=X_data, H=H, curr_mixture=None, prev_mixture=None, error=None)

    # --- Act ---
    _, new_params = lmoments_strategy(component, state, block, optimizer=None)

    # --- Assert ---
    assert state.error is not None, f"Expected error for case with n_samples={n_samples}, x_val={x_val}"
    assert isinstance(state.error, NumericalStabilityError)
    assert "Overflow detected during Lmoments optimization" in str(state.error)
    assert new_params == {}


# Property-Based Test with Hypothesis
# -----------------------------------


@st.composite
def exponential_data_and_true_params(draw, dtype_strategy=st.sampled_from([np.float64])):
    """
    Generates a true Exponential component and a data sample from it,
    all configured with a specific dtype.
    """
    dtype = draw(dtype_strategy)

    true_loc = draw(st.floats(min_value=-100, max_value=100))
    true_rate = draw(st.floats(min_value=0.1, max_value=100))
    true_component = Exponential(loc=true_loc, rate=true_rate, dtype=dtype)

    # 2. Generate a data sample from this distribution
    sample_size = draw(st.integers(min_value=10000, max_value=10000))
    X = true_component.generate(size=sample_size).astype(dtype)

    return X, dtype(true_loc), dtype(true_rate), dtype


@settings(max_examples=50)
@given(data=exponential_data_and_true_params())
def test_lmoments_exponential_recovers_true_params_on_ideal_data(data):
    """
    Scenario 3 (Statistical Sanity Check): Verifies that the analytical formulas
    can recover the original parameters from a sample when responsibilities are 1.0.
    This confirms the statistical validity of the implemented formulas in an ideal case.
    """
    # --- Arrange ---
    X, true_loc, true_rate, dtype = data

    # This is the key assumption for this test: perfect knowledge that all
    # data points belong to our component of interest.
    H_j = np.ones_like(X, dtype=dtype)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T  # Simulate a 2-component mixture

    # Use a starting component with completely different parameters
    start_component = Exponential(loc=-999.0, rate=0.001, dtype=dtype)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "rate"}, maximization_strategy=MaximizationStrategy.LMOMENTS
    )

    _, new_params = lmoments_strategy(start_component, state, block, optimizer=None)

    # The estimated parameters won't be exactly the same due to sampling noise,
    # so we use pytest.approx with a relative tolerance.
    # For a large enough sample, the estimates should be reasonably close.
    assert new_params[Exponential.PARAM_LOC] == pytest.approx(true_loc, abs=0.2)
    assert new_params[Exponential.PARAM_RATE] == pytest.approx(true_rate, rel=0.1)

    # Verify that the returned parameters have the correct dtype.
    assert isinstance(new_params[Exponential.PARAM_LOC], dtype)
    assert isinstance(new_params[Exponential.PARAM_RATE], dtype)
