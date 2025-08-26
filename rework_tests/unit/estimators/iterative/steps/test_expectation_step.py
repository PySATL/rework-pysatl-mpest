"""Tests for ExpectationStep"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from rework_pysatl_mpest.core.mixture import MixtureModel
from rework_pysatl_mpest.estimators.iterative.pipeline_state import PipelineState
from rework_pysatl_mpest.estimators.iterative.steps.expectation_step import (
    ExpectationStep,
)
from rework_pysatl_mpest.estimators.iterative.steps.maximization_step import (
    MaximizationStep,
)

# --- Test Fixtures ---


@pytest.fixture
def mock_mixture(mocker):
    """
    Creates a mock MixtureModel object with two components using pytest-mock.
    This allows for complete control over the inputs for the step being tested.
    """

    mock_component_1 = mocker.MagicMock()
    mock_component_2 = mocker.MagicMock()

    mock_component_1.lpdf.return_value = np.log([0.6, 0.8])
    mock_component_2.lpdf.return_value = np.log([0.1, 0.3])

    mixture = mocker.create_autospec(MixtureModel, instance=True)
    mixture.components = (mock_component_1, mock_component_2)
    mixture.log_weights = np.log([0.7, 0.3])

    return mixture


@pytest.fixture
def initial_pipeline_state(mock_mixture) -> PipelineState:
    """
    Creates an initial PipelineState for use in tests.
    """
    return PipelineState(
        X=np.array([[1], [2]]),
        H=None,
        prev_mixture=None,
        curr_mixture=mock_mixture,
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

    expected_h = np.array([[0.42 / 0.45, 0.03 / 0.45], [0.56 / 0.65, 0.09 / 0.65]])

    result_state = step.run(state)

    assert result_state.H is not None
    assert_allclose(result_state.H, expected_h, rtol=1e-6, err_msg="Soft H matrix calculation is incorrect")


def test_run_hard_assignment_calculates_h_correctly(initial_pipeline_state):
    """
    Verifies the correct calculation of the responsibility
    matrix H in hard assignment mode (is_soft=False).
    """

    state = initial_pipeline_state
    step = ExpectationStep(is_soft=False)

    expected_h_hard = np.array([[1.0, 0.0], [1.0, 0.0]])

    result_state = step.run(state)

    assert result_state.H is not None
    assert_array_equal(
        result_state.H,
        expected_h_hard,
        err_msg="Hard H matrix calculation is incorrect",
    )


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
