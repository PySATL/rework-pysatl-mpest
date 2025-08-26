"""Tests for StepBreakpointer"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from rework_pysatl_mpest.core.mixture import MixtureModel
from rework_pysatl_mpest.distributions.exponential import Exponential
from rework_pysatl_mpest.estimators.iterative.breakpointers.step_breakpointer import (
    StepBreakpointer,
)
from rework_pysatl_mpest.estimators.iterative.pipeline_state import PipelineState

# --- Helper Fixtures ---


@pytest.fixture
def dummy_state() -> PipelineState:
    """Provides a dummy PipelineState instance for tests."""

    dummy_mixture = MixtureModel([Exponential(0, 1)])
    return PipelineState(np.array([]), None, None, dummy_mixture, None)


# --- Initialization Tests ---


class TestInitialization:
    """Tests for the StepBreakpointer constructor (__init__)."""

    @pytest.mark.parametrize(
        "max_steps",
        [
            1,  # Boundary case: minimum allowed value
            5,  # Typical value
            100,  # Larger value
        ],
    )
    def test_initialization_with_valid_max_steps(self, max_steps: int):
        """
        Basic Test: Verifies that the class initializes correctly with valid
        positive integer values for max_steps.
        """

        breakpointer = StepBreakpointer(max_steps)
        assert breakpointer.max_steps == max_steps
        assert breakpointer._current_step == 0

    def test_initialization_value_error_message(self):
        """
        Negative Test: Checks the specific error message for a ValueError
        to ensure it is informative.
        """

        expected_message = "The maximum number of steps must be greater than or equal to 1"
        with pytest.raises(ValueError, match=expected_message):
            StepBreakpointer(0)


# --- Core Logic Tests for the check() Method ---


class TestCheckLogic:
    """Tests for the core logic of the check() method."""

    @pytest.mark.parametrize(
        "max_steps",
        [
            1,  # Boundary case: stops on the first call
            3,  # Typical case
            10,  # A longer scenario
        ],
    )
    def test_check_stops_at_correct_step(self, max_steps: int, dummy_state):
        """
        Basic Test: Verifies that check() returns False for n-1 calls and
        True on the nth call, where n is max_steps.
        """

        breakpointer = StepBreakpointer(max_steps)

        # The first `max_steps - 1` calls should return False
        for i in range(1, max_steps):
            should_stop = breakpointer.check(dummy_state)
            assert not should_stop, f"Should not stop at step {i} for max_steps={max_steps}"
            assert breakpointer._current_step == i, "Internal counter is incorrect"

        # The final, `max_steps`-th call should return True
        should_stop_final = breakpointer.check(dummy_state)
        assert should_stop_final, f"Should stop at the final step {max_steps}"

    def test_counter_resets_after_stopping(self, dummy_state):
        """
        Counter Reset Test: Verifies that the internal step counter is reset
        to 1 after the check() method returns True.
        """

        max_steps = 5
        breakpointer = StepBreakpointer(max_steps)

        # Call check() enough times to trigger the stop condition
        for _ in range(max_steps):
            breakpointer.check(dummy_state)

        # According to the implementation, the counter should be reset to 1
        assert breakpointer._current_step == 0

    def test_breakpointer_is_reusable_after_reset(self, dummy_state):
        """
        Reusability Test: Ensures that after the counter is reset, the instance
        can be used again for another full cycle of iterations.
        """

        max_steps = 3
        breakpointer = StepBreakpointer(max_steps)

        # First cycle
        assert not breakpointer.check(dummy_state)  # step 1
        assert not breakpointer.check(dummy_state)  # step 2
        assert breakpointer.check(dummy_state)  # step 3, stops and resets to 1

        # Verify the counter has been reset
        assert breakpointer._current_step == 0

        # Second cycle
        assert not breakpointer.check(dummy_state)  # step 1 of the new cycle
        assert not breakpointer.check(dummy_state)  # step 2
        assert breakpointer.check(dummy_state)  # step 3, stops and resets again

        assert breakpointer._current_step == 0
