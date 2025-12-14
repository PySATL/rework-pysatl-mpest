"""Tests for Observed Data Likelihood optimization strategy."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from rework_pysatl_mpest.core import MixtureModel, Parameter
from rework_pysatl_mpest.distributions import ContinuousDistribution
from rework_pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from rework_pysatl_mpest.estimators.iterative._strategies import observed_data_likelihood_strategy
from rework_pysatl_mpest.optimizers import Optimizer

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]

# Helper classes for test isolation
# ---------------------------------


class DummyDistribution(ContinuousDistribution):
    """
    A simple dummy implementation of ContinuousDistribution for testing purposes.
    Allows controlling the PDF output for math verification.
    """

    param1 = Parameter()
    param2 = Parameter()

    def __init__(self, param1: float, param2: float, pdf_value: float = 1.0, dtype: np.floating = np.float64):
        super().__init__(dtype=dtype)
        self.param1 = param1
        self.param2 = param2
        self._pdf_value = pdf_value

    @property
    def name(self) -> str:
        return "Dummy"

    @property
    def params(self) -> set[str]:
        return {"param1", "param2"}

    def pdf(self, X):
        # Return a constant PDF value for all X to make manual calculation easy
        return np.full_like(X, self._pdf_value, dtype=self.dtype)

    def ppf(self, P):
        return np.array([])

    def lpdf(self, X):
        return np.log(self.pdf(X))

    def log_gradients(self, X):
        return np.array([])

    def generate(self, size: int):
        return np.array([])


# --- Test Fixtures ---


@pytest.fixture(params=DTYPES_TO_TEST)
def parametrized_setup(
    mocker, request
) -> tuple[ContinuousDistribution, PipelineState, OptimizationBlock, Optimizer, np.floating]:
    """
    Fixture that creates a PipelineState with a MixtureModel containing
    multiple components to test the 'background term' logic.
    """

    dtype = request.param

    target_comp = DummyDistribution(param1=1.0, param2=2.0, pdf_value=2.0, dtype=dtype)
    bg_comp = DummyDistribution(param1=10.0, param2=20.0, pdf_value=0.5, dtype=dtype)

    mock_mixture = mocker.create_autospec(MixtureModel, instance=True)
    mock_mixture.components = [target_comp, bg_comp]
    mock_mixture.weights = np.array([0.4, 0.6], dtype=dtype)

    state = PipelineState(
        X=np.array([1.0, 2.0, 3.0], dtype=dtype),
        H=None,  # H is not used in this strategy
        prev_mixture=None,
        curr_mixture=mock_mixture,
        error=None,
    )

    block = OptimizationBlock(
        component_id=0,
        params_to_optimize={"param1", "param2"},
        maximization_strategy=MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD,
    )

    optimizer = mocker.create_autospec(Optimizer, instance=True)
    # Mock return value for minimize
    optimizer.minimize.return_value = [dtype(5.0), dtype(5.0)]

    return target_comp, state, block, optimizer, dtype


# Tests
# -----


def test_odl_strategy_does_not_modify_original_component(parametrized_setup):
    """
    Verifies that the original component object in the mixture is not modified
    in place before the optimization result is applied (strategy works on copies).
    """

    target_comp, state, block, optimizer, _ = parametrized_setup
    original_param1 = target_comp.param1

    observed_data_likelihood_strategy(target_comp, state, block, optimizer)

    assert target_comp.param1 == original_param1


def test_odl_strategy_calls_optimizer_minimize_once(parametrized_setup):
    """
    Verifies that the optimizer's minimize method is called exactly once.
    """

    target_comp, state, block, optimizer, _ = parametrized_setup

    observed_data_likelihood_strategy(target_comp, state, block, optimizer)
    optimizer.minimize.assert_called_once()


def test_odl_strategy_returns_correct_types(parametrized_setup):
    """
    Verifies that the function returns a tuple with the correct
    data types (int, dict[str, DType]).
    """

    target_comp, state, block, optimizer, dtype = parametrized_setup
    result = observed_data_likelihood_strategy(target_comp, state, block, optimizer)

    assert isinstance(result, tuple)
    assert isinstance(result[0], int)
    assert isinstance(result[1], dict)

    if result[1]:
        key, value = next(iter(result[1].items()))
        assert isinstance(key, str)
        assert isinstance(value, dtype)


def test_odl_strategy_math_correctness(parametrized_setup):
    """
    Verifies the mathematical logic of the target function passed to the optimizer.
    The target function must calculate: - sum( log( Background + w_target * PDF_target ) )
    """

    target_comp, state, block, optimizer, dtype = parametrized_setup

    # Extract setup data for manual verification
    X = state.X
    weights = state.curr_mixture.weights  # [0.4, 0.6]
    target_w = weights[0]  # 0.4
    bg_w = weights[1]  # 0.6
    bg_pdf_val = 0.5  # From fixture setup for bg_comp
    target_pdf_val = 2.0  # From fixture setup for target_comp

    # 1. Run strategy to capture the target closure
    observed_data_likelihood_strategy(target_comp, state, block, optimizer)

    args, _ = optimizer.minimize.call_args
    captured_target_func, initial_vector = args

    # 2. Simulate optimizer passing new parameters
    # Let's say the optimizer tries a vector that doesn't change PDF (for simplicity of check),
    # or we just rely on the DummyDistribution returning fixed PDF=2.0 regardless of params.
    test_params_vector = [dtype(1.0), dtype(2.0)]

    # 3. Calculate expected Negative Log Likelihood manually
    # Background term per sample = w_bg * pdf_bg = 0.6 * 0.5 = 0.3
    background_term = bg_w * bg_pdf_val

    # Target term per sample = w_target * pdf_target = 0.4 * 2.0 = 0.8
    comp_term = target_w * target_pdf_val

    # Mixture PDF per sample = 0.3 + 0.8 = 1.1
    mixture_pdf_val = background_term + comp_term

    # Log Likelihood per sample = ln(1.1)
    # Total Log Likelihood = N_samples * ln(1.1)
    n_samples = len(X)
    expected_nll = -1 * n_samples * np.log(mixture_pdf_val)

    # 4. Execute captured function
    actual_nll = captured_target_func(test_params_vector)

    # 5. Assert (using approx for float precision)
    assert np.isclose(actual_nll, expected_nll, rtol=1e-5)


def test_odl_strategy_respects_fixed_params(parametrized_setup):
    """
    Verifies that parameters marked as 'fixed' are not optimized.
    """

    target_comp, state, block, optimizer, dtype = parametrized_setup

    # Fix 'param1'
    target_comp.fix_param("param1")

    # Optimizer should only return one value now
    optimizer.minimize.return_value = [dtype(99.0)]

    _, new_params = observed_data_likelihood_strategy(target_comp, state, block, optimizer)

    assert "param1" not in new_params
    assert "param2" in new_params
    assert new_params["param2"] == dtype(99.0)

    # Verify optimizer was called with smaller vector
    args, _ = optimizer.minimize.call_args
    _, initial_vector = args
    assert len(initial_vector) == 1


def test_odl_strategy_handles_single_component_mixture(mocker):
    """
    Verifies that the strategy works even if there is only 1 component
    (background term should be 0).
    """

    dtype = np.float64
    target_comp = DummyDistribution(1.0, 2.0, pdf_value=2.0, dtype=dtype)

    mock_mixture = mocker.create_autospec(MixtureModel, instance=True)
    mock_mixture.components = [target_comp]
    mock_mixture.weights = np.array([1.0], dtype=dtype)

    state = PipelineState(
        X=np.array([10.0], dtype=dtype), H=None, curr_mixture=mock_mixture, prev_mixture=None, error=None
    )

    block = OptimizationBlock(0, {"param1", "param2"}, MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD)
    optimizer = mocker.create_autospec(Optimizer, instance=True)
    optimizer.minimize.return_value = [1.0, 2.0]

    observed_data_likelihood_strategy(target_comp, state, block, optimizer)

    # Capture target function to verify background is 0
    args, _ = optimizer.minimize.call_args
    target_func, _ = args

    # Expected: -sum(log(0 + 1.0 * 2.0)) = -1 * log(2) = -0.693...
    res = target_func([1.0, 2.0])
    expected = -np.log(2.0)

    assert np.isclose(res, expected)
