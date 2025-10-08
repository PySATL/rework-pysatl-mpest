"""Tests for PriorThresholdPruner"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy._core.numerictypes import float64
from numpy.typing import ArrayLike, NDArray
from rework_pysatl_mpest.core import MixtureModel
from rework_pysatl_mpest.distributions import ContinuousDistribution
from rework_pysatl_mpest.estimators.iterative import PipelineState, PriorThresholdPruner


class DummyDistribution(ContinuousDistribution):
    """A simple mock implementation of ContinuousDistribution for testing purposes."""

    def __init__(self, name: str):
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> set[str]:
        return set()

    def pdf(self, X: ArrayLike) -> NDArray[float64]:
        pass

    def ppf(self, P: ArrayLike) -> NDArray[float64]:
        pass

    def lpdf(self, X: ArrayLike) -> NDArray[float64]:
        pass

    def log_gradients(self, X: ArrayLike) -> NDArray[float64]:
        pass

    def generate(self, size: int) -> NDArray[float64]:
        pass

    def __repr__(self):
        return f"DummyDistribution(name='{self.name}')"


# --- Fixtures ---


@pytest.fixture
def dummy_components() -> list[DummyDistribution]:
    """Fixture that provides a list of unique dummy components."""

    return [DummyDistribution(f"comp_{i}") for i in range(3)]


# --- Tests for __init__ ---


def test_init_successful_with_valid_threshold():
    """
    Basic test: verifies that initialization is successful with a valid
    threshold value.
    """

    threshold = 0.1
    pruner = PriorThresholdPruner(threshold)
    assert pruner.threshold == threshold


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        0.0,  # Edge value (not allowed)
        1.0,  # Edge value (not allowed)
        -0.1,  # Invalid value (less than 0)
        1.1,  # Invalid value (greater than 1)
        -100,
        100,
    ],
    ids=["zero", "one", "negative", "greater_than_one", "large_negative", "large_positive"],
)
def test_init_raises_value_error_for_invalid_threshold(invalid_threshold):
    """
    Negative test: verifies that a ValueError is raised on initialization
    with invalid edge and out-of-range values.
    """

    with pytest.raises(ValueError, match="Threshold must be between 0 and 1."):
        PriorThresholdPruner(invalid_threshold)


# --- Tests for the prune method ---


@pytest.mark.parametrize(
    "threshold, initial_weights, expected_n_components, expected_remaining_indices",
    [
        # Case 1: No component is removed (all weights > threshold)
        (0.05, [0.4, 0.3, 0.3], 3, [0, 1, 2]),
        # Case 2: One component is removed
        (0.15, [0.8, 0.1, 0.1], 1, [0]),
        # Case 3: Two components are removed
        (0.2, [0.7, 0.15, 0.15], 1, [0]),
        # Case 4: The component with the smallest weight is removed
        (0.1, [0.8, 0.15, 0.05], 2, [0, 1]),
        # Case 5: Nothing is removed as weight equals the threshold
        (0.1, [0.8, 0.1, 0.1], 3, [0, 1, 2]),
    ],
    ids=[
        "no_pruning",
        "prune_one_component",
        "prune_two_components",
        "prune_smallest_weight",
        "weight_equals_threshold",
    ],
)
def test_prune_removes_correct_components(
    dummy_components, threshold, initial_weights, expected_n_components, expected_remaining_indices
):
    """
    Basic and edge case tests: checks the core logic of component removal.
    Ensures that only components with weights < threshold are removed
    and that the weights of the remaining components are correctly renormalized.
    """

    pruner = PriorThresholdPruner(threshold)
    initial_mixture = MixtureModel(dummy_components, weights=initial_weights)
    state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)

    new_state = pruner.prune(state)

    # Check that the number of components matches the expected value
    assert new_state.curr_mixture.n_components == expected_n_components

    # Check that the correct components remain
    expected_components = tuple(dummy_components[i] for i in expected_remaining_indices)
    assert new_state.curr_mixture.components == expected_components

    # Check that the sum of weights after pruning is 1.0
    assert np.isclose(np.sum(new_state.curr_mixture.weights), 1.0)


def test_prune_does_not_remove_last_component(dummy_components):
    """
    Edge case test: verifies that the pruner does not remove the last component,
    even if its weight is below the threshold.
    """

    pruner = PriorThresholdPruner(threshold=0.9)
    # Mixture with two components, both below the threshold
    initial_mixture = MixtureModel(dummy_components[:2], weights=[0.5, 0.5])
    state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)

    # After the first prune, one component will remain, which should not be removed
    new_state = pruner.prune(state)
    assert new_state.curr_mixture.n_components == 1


def test_prune_preserves_other_pipeline_state_attributes(dummy_components):
    """
    Function purity test: verifies that the prune method only modifies the
    `curr_mixture` attribute of the state object, leaving other attributes untouched.
    """

    pruner = PriorThresholdPruner(threshold=0.5)

    # Create a fully populated state object
    X_data = np.array([1, 2, 3])
    H_data = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    prev_mix = MixtureModel([DummyDistribution("prev_comp")], weights=[1.0])
    curr_mix = MixtureModel(dummy_components[:2], weights=[0.1, 0.9])
    error_obj = ValueError("test error")

    initial_state = PipelineState(X=X_data, H=H_data, prev_mixture=prev_mix, curr_mixture=curr_mix, error=error_obj)

    # Store the object IDs for comparison
    initial_X_id = id(initial_state.X)
    initial_H_id = id(initial_state.H)
    initial_prev_mix_id = id(initial_state.prev_mixture)
    initial_error_id = id(initial_state.error)
    initial_curr_mix_id = id(initial_state.curr_mixture)

    new_state = pruner.prune(initial_state)

    # Verify that other state attributes have not changed (are the same objects)
    assert id(new_state.X) == initial_X_id
    assert id(new_state.H) == initial_H_id
    assert id(new_state.prev_mixture) == initial_prev_mix_id
    assert id(new_state.error) == initial_error_id

    # Verify that the curr_mixture object has been replaced (since `copy` creates a new object).
    # This is important to confirm that the mutation does not happen on the original object
    # passed into PipelineState.
    assert id(new_state.curr_mixture) != initial_curr_mix_id
    assert new_state.curr_mixture.n_components == 1  # The component with weight 0.1 should have been removed


# This code commented, because need change copy to deepcopy in pruner and
# for testing this need __eq__ and __ne__ methods for ContinuousDistribution
# def test_prune_does_not_modify_original_mixture_object(dummy_components):
#     """
#     Verifies that the original MixtureModel object passed into PipelineState
#     is not mutated, as `prune` is expected to work on a copy.
#     """
#
#     pruner = PriorThresholdPruner(threshold=0.5)
#     initial_mixture = MixtureModel(dummy_components[:2], weights=[0.1, 0.9])
#
#     # Create a deep copy for later comparison
#     original_mixture_snapshot = deepcopy(initial_mixture)
#
#     state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)
#
#     pruner.prune(state)
#
#     # Verify that the original `initial_mixture` object has not changed
#     assert initial_mixture.n_components == original_mixture_snapshot.n_components
#     assert np.allclose(initial_mixture.weights, original_mixture_snapshot.weights)
#     assert initial_mixture.components == original_mixture_snapshot.components


# --- Tests using Hypothesis ---


@given(
    # Generate a list of positive floats to serve as weights
    weights_list=st.lists(st.floats(min_value=1e-6, max_value=1.0, allow_nan=False), min_size=1, max_size=10),
    # Generate a threshold
    threshold=st.floats(min_value=1e-6, max_value=1.0 - 1e-6, allow_nan=False),
)
def test_prune_with_hypothesis_generated_data(weights_list, threshold):
    """
    Property-based test: uses Hypothesis to verify the logic on a wide range
    of randomly generated data.
    """

    # Normalize the generated weights so that their sum is 1.0
    weights_sum = sum(weights_list)
    if weights_sum == 0:  # Avoid division by zero
        return
    initial_weights = np.array(weights_list) / weights_sum

    num_components = len(initial_weights)
    components = [DummyDistribution(f"comp_{i}") for i in range(num_components)]

    pruner = PriorThresholdPruner(threshold)
    initial_mixture = MixtureModel(components, weights=initial_weights)
    state = PipelineState(X=np.array([]), H=None, prev_mixture=None, curr_mixture=initial_mixture, error=None)

    new_state = pruner.prune(state)
    new_mixture = new_state.curr_mixture

    # Property 1: The mixture must always have at least one component remaining
    assert new_mixture.n_components >= 1

    # Property 2: The number of components should not increase
    assert new_mixture.n_components <= initial_mixture.n_components

    # Property 3: If more than one component remains, all of the original
    # components that were kept must have had a weight >= threshold.
    # It is hard to check this directly as weights are recalculated.
    # Instead, we'll verify that all removed components had an initial weight < threshold.

    initial_components_set = set(initial_mixture.components)
    final_components_set = set(new_mixture.components)
    removed_components = initial_components_set - final_components_set

    # If not all components were removed down to one
    if len(removed_components) < initial_mixture.n_components - 1:
        for i, comp in enumerate(initial_mixture.components):
            if comp in removed_components:
                assert initial_mixture.weights[i] < threshold

    # Property 4: The sum of weights must always be 1.0
    assert np.isclose(np.sum(new_mixture.weights), 1.0)
