"""Tests for ExpectationStep"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import ExpectationStep, MaximizationStep, PipelineState

# --- Test Fixtures ---

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]


@pytest.fixture(params=DTYPES_TO_TEST)
def initial_pipeline_state(request, mocker) -> PipelineState:
    """
    Creates an initial PipelineState for use in tests.

    This fixture is executed for each data type in DTYPES_TO_TEST. It constructs
    a mock mixture model and a pipeline state where all relevant components
    (lpdf return values, weights, input data X) share the same parametrized dtype.
    """
    dtype = request.param

    mock_component_1 = mocker.MagicMock()
    mock_component_2 = mocker.MagicMock()

    mock_component_1.lpdf.return_value = np.log([0.6, 0.8]).astype(dtype)
    mock_component_2.lpdf.return_value = np.log([0.1, 0.3]).astype(dtype)

    mixture = mocker.create_autospec(MixtureModel, instance=True)
    mixture.components = (mock_component_1, mock_component_2)
    mixture.log_weights = np.log([0.7, 0.3]).astype(dtype)
    mixture.dtype = dtype

    return PipelineState(
        X=np.array([[1], [2]], dtype=dtype),
        H=None,
        prev_mixture=None,
        curr_mixture=mixture,
        error=None,
    )


# --- Tests ---


def test_expectation_step_initialization():
    """
    Verifies the correct initialization of the `is_soft` flag.
    """

    # By default, is_soft should be True
    default_step = ExpectationStep()
    assert default_step.is_soft is True

    # Explicitly set to True
    soft_step = ExpectationStep(is_soft=True)
    assert soft_step.is_soft is True

    # Explicitly set to False
    hard_step = ExpectationStep(is_soft=False)
    assert hard_step.is_soft is False


def test_available_next_steps_property():
    """
    Verifies that the `available_next_steps` property returns
    the correct list of permissible subsequent steps.
    """

    step = ExpectationStep()
    assert step.available_next_steps == [MaximizationStep], "ExpectationStep should be followed by MaximizationStep"


def test_run_soft_assignment_calculates_h_correctly(initial_pipeline_state):
    """
    Verifies the correct calculation of the responsibility
    matrix H in soft assignment mode.
    """

    # --- Arrange ---
    state = initial_pipeline_state
    step = ExpectationStep(is_soft=True)

    expected_h = np.array([[0.42 / 0.45, 0.03 / 0.45], [0.56 / 0.65, 0.09 / 0.65]], dtype=state.curr_mixture.dtype)

    result_state = step.run(state)

    assert result_state.H is not None
    atol = 1e-3 if state.curr_mixture.dtype == np.float16 else 1e-6
    assert_allclose(result_state.H, expected_h, rtol=1e-6, atol=atol, err_msg="Soft H matrix calculation is incorrect")
    # dtype correct
    assert result_state.H.dtype == state.curr_mixture.dtype


def test_run_hard_assignment_calculates_h_correctly(initial_pipeline_state):
    """
    Verifies the correct calculation of the responsibility
    matrix H in hard assignment mode (is_soft=False).
    """

    state = initial_pipeline_state
    step = ExpectationStep(is_soft=False)

    expected_h_hard = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=state.curr_mixture.dtype)

    result_state = step.run(state)

    assert result_state.H is not None
    assert_array_equal(
        result_state.H,
        expected_h_hard,
        err_msg="Hard H matrix calculation is incorrect",
    )
    # dtype correct
    assert result_state.H.dtype == state.curr_mixture.dtype


def test_run_returns_state_and_modifies_only_h(initial_pipeline_state):
    """
    Verifies that the `run` method returns the same state object,
    modifying only its H attribute.
    """

    state = initial_pipeline_state
    step = ExpectationStep(is_soft=False)

    original_x_ref = state.X
    original_mixture_ref = state.curr_mixture
    original_prev_mixture_ref = state.prev_mixture
    original_error_ref = state.error

    result_state = step.run(state)

    # Check the type and identity of the returned object
    assert isinstance(result_state, PipelineState), "The return type should be PipelineState"
    assert result_state is state, "The method should return the same state object"

    # Check that the H attribute was modified (from None to np.ndarray)
    assert state.H is not None
    assert isinstance(result_state.H, np.ndarray)

    # Check that other attributes remain the exact same objects
    assert result_state.X is original_x_ref, "Data X object reference should not be changed"
    assert result_state.curr_mixture is original_mixture_ref, "`curr_mixture` object reference should not be changed"
    assert result_state.prev_mixture is original_prev_mixture_ref, (
        "`prev_mixture` object reference should not be changed"
    )
    assert result_state.error is original_error_ref, "`error` object reference should not be changed"
