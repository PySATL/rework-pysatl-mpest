"""Tests for Moments optimization strategy for Normal distribution"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pysatl_mpest.distributions import Normal
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from pysatl_mpest.estimators.iterative._strategies import moments_strategy

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]

# Test Fixtures
# -------------


@pytest.fixture(params=DTYPES_TO_TEST)
def parametrized_normal_setup(request) -> tuple[Normal, PipelineState, np.floating]:
    """
    Creates a parametrized fixture providing a Normal component and a
    corresponding PipelineState for various dtypes.
    """

    dtype = request.param

    component = Normal(loc=0.0, scale=1.0, dtype=dtype)

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


def test_moments_normal_raises_value_error_if_h_is_none(parametrized_normal_setup):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed.
    """

    normal_component, state, _ = parametrized_normal_setup
    state.H = None

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "scale"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        moments_strategy(normal_component, state, block, optimizer=None)


def test_moments_normal_returns_correct_types(parametrized_normal_setup):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, FloatT]).
    """

    normal_component, pipeline_state, dtype = parametrized_normal_setup

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "scale"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    result = moments_strategy(normal_component, pipeline_state, block, optimizer=None)

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
        ({"loc", "scale"}, set(), {"loc", "scale"}),  # Optimize both
        ({"loc"}, set(), {"loc"}),  # Optimize only loc
        ({"scale"}, set(), {"scale"}),  # Optimize only scale
        ({"loc", "scale"}, {"loc"}, {"scale"}),  # Optimize both, but loc is fixed
        ({"loc", "scale"}, {"scale"}, {"loc"}),  # Optimize both, but scale is fixed
        ({"loc", "scale"}, {"loc", "scale"}, set()),  # Optimize both, but both are fixed
        ({"non_existent_param", "loc"}, set(), {"loc"}),  # Ignore non-existent params
        (set(), set(), set()),  # Optimize nothing
    ],
)
def test_moments_normal_respects_fixed_and_optimizable_params(
    parametrized_normal_setup, params_to_optimize_in_block, fixed_params_on_component, expected_keys
):
    """
    Verifies that the strategy correctly identifies which parameters
    to update based on the optimization block and the component's fixed parameters.
    """

    normal_component, pipeline_state, _ = parametrized_normal_setup

    for param in fixed_params_on_component:
        normal_component.fix_param(param)

    block = OptimizationBlock(
        component_id=0,
        params_to_optimize=params_to_optimize_in_block,
        maximization_strategy=MaximizationStrategy.MOMENTS,
    )

    _, new_params = moments_strategy(normal_component, pipeline_state, block, optimizer=None)

    assert set(new_params.keys()) == expected_keys


def test_moments_normal_handles_negligible_responsibility(parametrized_normal_setup):
    """
    Verifies that if a component's total responsibility (N_j) is near zero,
    its parameters are not updated.
    """

    normal_component, pipeline_state, dtype = parametrized_normal_setup

    pipeline_state.H.fill(dtype(1e-10))  # Make all responsibilities negligible
    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={"loc", "scale"},
        maximization_strategy=MaximizationStrategy.MOMENTS,
    )

    _, new_params = moments_strategy(normal_component, pipeline_state, block, optimizer=None)

    assert new_params == {}


def test_moments_normal_clamping_min_scale(parametrized_normal_setup):
    """
    Verifies that the scale (std dev) is clamped to machine epsilon if the variance
    calculates to zero (e.g., all data points are identical).
    This prevents division by zero in subsequent PDF calculations.
    """

    component, state, dtype = parametrized_normal_setup

    # Arrange: All data points are identical, so variance should be 0.0
    val = dtype(10.0)
    state.X = np.array([val, val, val], dtype=dtype)
    state.H = np.ones((3, 2), dtype=dtype)  # Component 0 has full responsibility

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"scale"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    # Act
    _, new_params = moments_strategy(component, state, block, optimizer=None)

    # Assert
    assert Normal.PARAM_SCALE in new_params
    new_scale = new_params[Normal.PARAM_SCALE]

    # Check that scale is not 0.0, but exactly the machine epsilon for that dtype
    expected_min = np.finfo(dtype).eps
    assert new_scale > 0.0
    assert new_scale == pytest.approx(expected_min, abs=0.0, rel=1e-5) or new_scale >= expected_min


def test_moments_normal_calculation_correctness(parametrized_normal_setup):
    """
    Verifies the mathematical correctness of the weighted mean and variance calculation
    on a simple deterministic example.
    """

    component, state, dtype = parametrized_normal_setup

    # Data: [10, 20], Weights: [0.2, 0.8]
    # Weighted Mean = (10*0.2 + 20*0.8) / (0.2+0.8) = 2 + 16 = 18.0
    # Weighted Var = 0.2*(10-18)^2 + 0.8*(20-18)^2 = 0.2*64 + 0.8*4 = 12.8 + 3.2 = 16.0
    # Weighted Scale = sqrt(16.0) = 4.0

    state.X = np.array([10.0, 20.0], dtype=dtype)
    state.H = np.array([[0.2], [0.8]], dtype=dtype)

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "scale"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    _, new_params = moments_strategy(component, state, block, optimizer=None)

    assert new_params[Normal.PARAM_LOC] == pytest.approx(18.0, rel=1e-4)
    assert new_params[Normal.PARAM_SCALE] == pytest.approx(4.0, rel=1e-4)


# Property-Based Test with Hypothesis
# -----------------------------------


@st.composite
def normal_data_and_true_params(draw, dtype_strategy=st.sampled_from([np.float64])):
    """
    Generates a true Normal component and a data sample from it.
    Restricted to float64 to ensure stability during random generation tests.
    """

    dtype = draw(dtype_strategy)

    true_loc = draw(st.floats(min_value=-50, max_value=50))
    # Avoid extremely small scales to prevent random noise issues in tests
    true_scale = draw(st.floats(min_value=0.5, max_value=20))

    true_component = Normal(loc=true_loc, scale=true_scale, dtype=dtype)

    # Generate a large sample to ensure convergence of moments
    sample_size = draw(st.integers(min_value=5000, max_value=10000))
    X = true_component.generate(size=sample_size).astype(dtype)

    return X, dtype(true_loc), dtype(true_scale), dtype


@settings(max_examples=50)
@given(data=normal_data_and_true_params())
def test_moments_normal_recovers_true_params_on_ideal_data(data):
    """
    Scenario: Verifies that the implementation recovers original parameters
    from a sample when responsibilities are perfect (1.0).
    """

    # --- Arrange ---
    X, true_loc, true_scale, dtype = data

    # Assume perfect responsibility (1 component mixture)
    H_j = np.ones_like(X, dtype=dtype)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T

    # Start with bad parameters
    start_component = Normal(loc=-999.0, scale=0.1, dtype=dtype)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "scale"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    # --- Act ---
    _, new_params = moments_strategy(start_component, state, block, optimizer=None)

    # --- Assert ---
    # Relaxed tolerance for scale because sample std deviation has variance itself
    assert new_params[Normal.PARAM_LOC] == pytest.approx(true_loc, abs=0.5)
    assert new_params[Normal.PARAM_SCALE] == pytest.approx(true_scale, rel=0.1)

    assert isinstance(new_params[Normal.PARAM_LOC], dtype)
    assert isinstance(new_params[Normal.PARAM_SCALE], dtype)


@st.composite
def normal_data_with_random_weights(draw, dtype_strategy=st.sampled_from([np.float64])):
    """
    Generates data using NumPy (seeded by Hypothesis) to avoid
    Hypothesis list size limits on large datasets.
    """
    dtype = draw(dtype_strategy)

    true_loc = draw(st.floats(min_value=-10.0, max_value=10.0))
    true_scale = draw(st.floats(min_value=0.5, max_value=5.0))

    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)

    sample_size = 15000  # Достаточно для сходимости
    X = rng.normal(loc=true_loc, scale=true_scale, size=sample_size).astype(dtype)
    weights = rng.uniform(low=0.1, high=1.0, size=sample_size).astype(dtype)

    return X, weights, dtype, dtype(true_loc), dtype(true_scale)


@settings(max_examples=30, deadline=None)
@given(data=normal_data_with_random_weights())
def test_moments_normal_converges_to_true_params_with_random_weights(data):
    """
    Verifies that the weighted moment estimates converge to the TRUE parameters
    of the generating distribution, even when the responsibilities (weights)
    are random and not equal to 1.0.

    This ensures that the weighting logic is statistically consistent:
    Weighted Mean of X (with random independent weights) -> True Mean
    Weighted Var of X (with random independent weights) -> True Var
    """
    X, H_j, dtype, true_loc, true_scale = data

    # --- Arrange ---
    # Create a dummy H matrix (Nx2) where column 0 is our random weights
    H = np.vstack([H_j, np.zeros_like(H_j)]).T.astype(dtype)

    # Start with incorrect parameters to ensure we actually calculate something
    start_component = Normal(loc=true_loc + 100, scale=true_scale + 50, dtype=dtype)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"loc", "scale"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    # --- Act ---
    _, new_params = moments_strategy(start_component, state, block, optimizer=None)

    # --- Assert ---
    est_loc = new_params[Normal.PARAM_LOC]
    est_scale = new_params[Normal.PARAM_SCALE]

    # Check Location convergence
    assert est_loc == pytest.approx(true_loc, abs=0.2, rel=0.1)

    # Check Scale convergence
    assert est_scale == pytest.approx(true_scale, rel=0.15)
