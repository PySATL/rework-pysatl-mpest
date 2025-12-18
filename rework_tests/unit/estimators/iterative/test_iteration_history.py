"""Tests for IterationHistory class"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from rework_pysatl_mpest.core import MixtureModel
from rework_pysatl_mpest.distributions import Exponential
from rework_pysatl_mpest.estimators.iterative._iteration_history import IterationRecord, IterationsHistory

# objects - 2-3  init(once_in =_iteration) (1,2,3)
# record = IterationRecord(history._counter, state.curr_mixture, state.X, state.H, self.pruners, state.error)
# save_record(record) - 3 cases +
# clear() - 1 case +
# clear_records() - 1 case +
# history[index] - 4-6 cases +
# len() - 2-3 cases +


# functions to set up objects for tests
@pytest.fixture
def exponential_components() -> list[Exponential]:
    return [Exponential(loc=0, rate=1), Exponential(loc=5, rate=2)]


@pytest.fixture
def responsibility_matrix():
    return np.array([[0.5, 0.5], [0.5, 0.5]])


@pytest.fixture
def sample_record_0(exponential_components, responsibility_matrix) -> IterationRecord:
    return IterationRecord(
        0, MixtureModel(exponential_components, weights=[0.5, 0.5]), np.array([1, 2]), responsibility_matrix, None, None
    )


@pytest.fixture
def sample_record_1(exponential_components, responsibility_matrix) -> IterationRecord:
    return IterationRecord(
        1, MixtureModel(exponential_components, weights=[0.4, 0.6]), np.array([5, 4]), responsibility_matrix, None, None
    )


@pytest.fixture
def setup_history_for_index(sample_record_0, sample_record_1) -> IterationsHistory:
    history = IterationsHistory()
    history.save_record(sample_record_0)
    history.save_record(sample_record_1)
    return history


@pytest.fixture(params=[(1, 5), (2, 8), (3, 10)])
def setup_for_records(request, exponential_components, responsibility_matrix):
    once_in_iter, n = request.param
    history = IterationsHistory(once_in_iter)

    for i in range(n):
        record = IterationRecord(
            history._counter,
            MixtureModel(exponential_components, np.array([0.5, 0.5])),
            np.array([1, i]),
            responsibility_matrix,
            None,
            None,
        )
        history.save_record(record)
    return history, once_in_iter, n


# tests
class TestIndex:
    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (-2, "sample_record_0"),
            (-1, "sample_record_1"),
            (0, "sample_record_0"),
            (1, "sample_record_1"),
        ],
    )
    def test_valid_index(self, setup_history_for_index, index: int, expected, request):
        expected_record = request.getfixturevalue(expected)
        actual = setup_history_for_index[index]

        assert actual == expected_record

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (4, "Index 4 out of range for container containing 2 elements"),
            (-4, "Index -4 out of range for container containing 2 elements"),
        ],
    )
    def test_invalid_index(self, setup_history_for_index, index, expected):
        with pytest.raises(IndexError, match=expected):
            _ = setup_history_for_index[index]

    @pytest.mark.parametrize(
        ("once_in_iterations", "expected_msg"),
        [
            (0, "once_in_iterations must be a positive integer"),
        ],
    )
    def test_getitem_negative_index_out_of_bounds(self, once_in_iterations, expected_msg):
        """Tests the specific branch `once_in_iterations < 1` in __init__."""
        with pytest.raises(ValueError, match=expected_msg):
            IterationsHistory(once_in_iterations)


class TestRecordingAndLen:
    @pytest.mark.parametrize(("once_in_iter", "n", "expected"), [(1, 5, 5), (2, 8, 4), (3, 10, 4)])
    def test_len(self, once_in_iter: int, n: int, expected: int, exponential_components, responsibility_matrix):
        history = IterationsHistory(once_in_iter)

        for i in range(n):
            record = IterationRecord(
                i,
                MixtureModel(exponential_components, weights=[0.5, 0.5]),
                np.array([1, n]),
                responsibility_matrix,
                None,
                None,
            )
            history.save_record(record)

        assert len(history) == expected

    def test_recording_content(self, setup_for_records, exponential_components, responsibility_matrix):
        history, once_in_iter, n = setup_for_records
        iteration_numbers = [i for i in range(0, n, once_in_iter)]

        for i in range(0, len(history)):
            expected_records = IterationRecord(
                iteration_numbers[i],
                MixtureModel(exponential_components, np.array([0.5, 0.5])),
                np.array([1, iteration_numbers[i]]),
                responsibility_matrix,
                None,
                None,
            )
            assert history[i].iteration == expected_records.iteration
            assert np.array_equal(history[i].X, expected_records.X)


class TestClear:
    @pytest.fixture
    def populated_history(self, sample_record_0, sample_record_1):
        history = IterationsHistory()
        history.save_record(sample_record_0)
        history.save_record(sample_record_1)
        history.save_record(sample_record_0)
        return history

    def test_reset(self, populated_history):
        history = populated_history
        EXPECTED_LEN_OF_HISTORY = 3

        assert len(history) == EXPECTED_LEN_OF_HISTORY
        assert history._counter == EXPECTED_LEN_OF_HISTORY

        history.reset()
        assert len(history) == 0
        assert history._counter == 0
        assert history.once_in_iterations == 1
