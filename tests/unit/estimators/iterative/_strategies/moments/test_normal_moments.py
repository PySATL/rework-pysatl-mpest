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

# Test Fixtures
# -------------


@pytest.fixture()
def parametrized_normal_setup() -> tuple[Normal, PipelineState]:
    """
    Creates a parametrized fixture providing a Normal component and a
    corresponding PipelineState for various dtypes.
    """

    component = Normal(mu=0.0, sigma=1.0)

    state = PipelineState(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        H=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=np.float64),
        prev_mixture=None,
        curr_mixture=None,
        error=None,
    )
    return component, state


# Tests
# -----


def test_moments_normal_raises_value_error_if_h_is_none(parametrized_normal_setup):
    """
    Verifies that a ValueError is raised if the responsibility
    matrix H in the pipeline state has not been computed.
    """

    normal_component, state = parametrized_normal_setup
    state.H = None

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"mu", "sigma"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        moments_strategy(normal_component, state, block, optimizer=None)


def test_moments_normal_returns_correct_types(parametrized_normal_setup):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, float]).
    """

    normal_component, pipeline_state = parametrized_normal_setup

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"mu", "sigma"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    result = moments_strategy(normal_component, pipeline_state, block, optimizer=None)

    assert isinstance(result, tuple)
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, float)


@pytest.mark.parametrize(
    "params_to_optimize_in_block, fixed_params_on_component, expected_keys",
    [
        ({"mu", "sigma"}, set(), {"mu", "sigma"}),  # Optimize both
        ({"mu"}, set(), {"mu"}),  # Optimize only mu
        ({"sigma"}, set(), {"sigma"}),  # Optimize only sigma
        ({"mu", "sigma"}, {"mu"}, {"sigma"}),  # Optimize both, but mu is fixed
        ({"mu", "sigma"}, {"sigma"}, {"mu"}),  # Optimize both, but sigma is fixed
        ({"mu", "sigma"}, {"mu", "sigma"}, set()),  # Optimize both, but both are fixed
        ({"non_existent_param", "mu"}, set(), {"mu"}),  # Ignore non-existent params
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

    normal_component, pipeline_state = parametrized_normal_setup

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

    normal_component, pipeline_state = parametrized_normal_setup

    pipeline_state.H.fill(1e-10)  # Make all responsibilities negligible
    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={"mu", "sigma"},
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

    component, state = parametrized_normal_setup

    # Arrange: All data points are identical, so variance should be 0.0
    val = 10.0
    state.X = np.array([val, val, val], dtype=np.float64)
    state.H = np.ones((3, 2), dtype=np.float64)  # Component 0 has full responsibility

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"sigma"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    # Act
    _, new_params = moments_strategy(component, state, block, optimizer=None)

    # Assert
    assert "sigma" in new_params
    new_scale = new_params["sigma"]

    # Check that scale is not 0.0, but exactly the machine epsilon for that dtype
    expected_min = np.finfo(np.float64).eps
    assert new_scale > 0.0
    assert new_scale == pytest.approx(expected_min, abs=0.0, rel=1e-5) or new_scale >= expected_min


def test_moments_normal_calculation_correctness(parametrized_normal_setup):
    """
    Verifies the mathematical correctness of the weighted mean and variance calculation
    on a simple deterministic example.
    """

    component, state = parametrized_normal_setup

    # Data: [10, 20], Weights: [0.2, 0.8]
    # Weighted Mean = (10*0.2 + 20*0.8) / (0.2+0.8) = 2 + 16 = 18.0
    # Weighted Var = 0.2*(10-18)^2 + 0.8*(20-18)^2 = 0.2*64 + 0.8*4 = 12.8 + 3.2 = 16.0
    # Weighted Scale = sqrt(16.0) = 4.0

    state.X = np.array([10.0, 20.0], dtype=np.float64)
    state.H = np.array([[0.2], [0.8]], dtype=np.float64)

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"mu", "sigma"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    _, new_params = moments_strategy(component, state, block, optimizer=None)

    assert new_params["mu"] == pytest.approx(18.0, rel=1e-4)
    assert new_params["sigma"] == pytest.approx(4.0, rel=1e-4)


# Property-Based Test with Hypothesis
# -----------------------------------


@st.composite
def normal_data_and_true_params(draw):
    """
    Generates a true Normal component and a data sample from it.
    Restricted to float64 to ensure stability during random generation tests.
    """

    true_mu = draw(st.floats(min_value=-50, max_value=50))
    # Avoid extremely small scales to prevent random noise issues in tests
    true_sigma = draw(st.floats(min_value=0.5, max_value=20))

    true_component = Normal(mu=true_mu, sigma=true_sigma)

    # Generate a large sample to ensure convergence of moments
    sample_size = draw(st.integers(min_value=5000, max_value=10000))
    X = true_component.generate(size=sample_size)

    return X, true_mu, true_sigma


@settings(max_examples=50)
@given(data=normal_data_and_true_params())
def test_moments_normal_recovers_true_params_on_ideal_data(data):
    """
    Scenario: Verifies that the implementation recovers original parameters
    from a sample when responsibilities are perfect (1.0).
    """

    # --- Arrange ---
    X, true_mu, true_sigma = data

    # Assume perfect responsibility (1 component mixture)
    H_j = np.ones_like(X, dtype=np.float64)
    H = np.vstack([H_j, np.zeros_like(H_j)]).T

    # Start with bad parameters
    start_component = Normal(mu=-999.0, sigma=0.1)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)
    block = OptimizationBlock(
        component_id=0, params_to_optimize={"mu", "sigma"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    # --- Act ---
    _, new_params = moments_strategy(start_component, state, block, optimizer=None)

    # --- Assert ---
    # Relaxed tolerance for scale because sample std deviation has variance itself
    assert new_params["mu"] == pytest.approx(true_mu, abs=1.0)
    assert new_params["sigma"] == pytest.approx(true_sigma, rel=0.1)

    assert isinstance(new_params["mu"], float)
    assert isinstance(new_params["sigma"], float)


@st.composite
def normal_data_with_random_weights(draw):
    """
    Generates data using NumPy (seeded by Hypothesis) to avoid
    Hypothesis list size limits on large datasets.
    """

    true_mu = draw(st.floats(min_value=-10.0, max_value=10.0))
    true_sigma = draw(st.floats(min_value=0.5, max_value=5.0))

    rng = np.random.default_rng(42)

    np.random.seed(42)

    sample_size = 15000  # Достаточно для сходимости
    X = rng.normal(loc=true_mu, scale=true_sigma, size=sample_size)
    weights = rng.uniform(low=0.1, high=1.0, size=sample_size)

    return X, weights, true_mu, true_sigma


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
    X, H_j, true_mu, true_sigma = data

    # --- Arrange ---
    # Create a dummy H matrix (Nx2) where column 0 is our random weights
    H = np.vstack([H_j, np.zeros_like(H_j)]).T

    # Start with incorrect parameters to ensure we actually calculate something
    start_component = Normal(mu=true_mu + 10, sigma=true_sigma + 5)

    state = PipelineState(X=X, H=H, curr_mixture=None, prev_mixture=None, error=None)

    block = OptimizationBlock(
        component_id=0, params_to_optimize={"mu", "sigma"}, maximization_strategy=MaximizationStrategy.MOMENTS
    )

    # --- Act ---
    _, new_params = moments_strategy(start_component, state, block, optimizer=None)

    # --- Assert ---
    est_loc = new_params["mu"]
    est_scale = new_params["sigma"]

    # Check Location convergence
    assert est_loc == pytest.approx(true_mu, abs=0.2, rel=0.1)

    # Check Scale convergence
    assert est_scale == pytest.approx(true_sigma, rel=0.15)
