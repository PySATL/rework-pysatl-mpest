"""Tests for the ECM class, which acts as a facade for the Pipeline."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.distributions import Exponential, Normal
from pysatl_mpest.estimators import ECM
from pysatl_mpest.estimators.iterative import (
    Breakpointer,
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    PipelineState,
    Pruner,
)
from pysatl_mpest.optimizers import Optimizer
from pytest_mock import MockerFixture

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]

# --- Mocks and Fixtures for testing ECM ---


class StopAfterOneIteration(Breakpointer):
    """A simple breakpointer that stops the pipeline after the first iteration."""

    def __init__(self):
        self.called = False

    def check(self, state: PipelineState) -> bool:
        if self.called:
            return True
        self.called = True
        return False


@pytest.fixture
def mock_optimizer(mocker: MockerFixture) -> Optimizer:
    """Provides a mock optimizer object using pytest-mock."""

    return mocker.Mock(spec=Optimizer)


@pytest.fixture
def mock_breakpointers() -> list[Breakpointer]:
    """Provides a list with a single mock breakpointer."""

    return [StopAfterOneIteration()]


@pytest.fixture
def mock_pruners() -> list[Pruner]:
    """Provides an empty list of pruners for simplicity."""

    return []


@pytest.fixture(params=DTYPES_TO_TEST)
def sample_mixture(request) -> MixtureModel:
    """Provides a mixture model with two components for testing with parametrized dtype."""

    dtype = request.param
    components = [Normal(loc=0, scale=1, dtype=dtype), Exponential(loc=10, rate=1, dtype=dtype)]
    # Fix a parameter in one component to test if `params_to_optimize` is correctly used.
    components[1].fix_param("rate")
    return MixtureModel(components, weights=[0.4, 0.6], dtype=dtype)


@pytest.fixture(params=DTYPES_TO_TEST)
def sample_data(request) -> np.ndarray:
    """Provides a simple NumPy array of data with parametrized dtype."""
    dtype = request.param
    return np.array([1, 2, 3, 11, 12, 13], dtype=dtype)


# --- Test Cases ---


class TestECMInitialization:
    """Tests for the constructor of the ECM class."""

    def test_init_stores_dependencies_correctly(
        self, mock_optimizer: Optimizer, mock_breakpointers: list[Breakpointer], mock_pruners: list[Pruner]
    ):
        """
        Tests that the constructor correctly assigns the provided breakpointers,
        pruners, and optimizer to the instance attributes.
        """

        ecm = ECM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        assert ecm.breakpointers is not mock_breakpointers
        assert ecm.breakpointers == mock_breakpointers
        assert ecm.pruners is not mock_pruners
        assert ecm.pruners == mock_pruners
        assert ecm.optimizer is mock_optimizer
        assert ecm._history is None


class TestECMFit:
    """Tests for the `fit` method of the ECM class."""

    def test_fit_creates_and_runs_pipeline(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        """
        Tests that `fit` correctly instantiates and calls the `fit` method of a Pipeline
        with the right high-level arguments.
        """

        # Arrange
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecm.Pipeline")
        mock_pipeline_instance = MockPipeline.return_value
        mock_pipeline_instance.fit.return_value = sample_mixture

        ecm = ECM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        # Act
        result = ecm.fit(sample_data, sample_mixture)

        # Assert
        MockPipeline.assert_called_once()
        mock_pipeline_instance.fit.assert_called_once_with(sample_data, sample_mixture)
        assert result is sample_mixture

        # Check types
        assert result.dtype == sample_mixture.dtype
        assert result.weights.dtype == sample_mixture.dtype
        for component in result.components:
            assert component.dtype == sample_mixture.dtype

    def test_fit_configures_pipeline_steps_correctly(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        """
        Verifies that the Pipeline created by ECM.fit has the correct structure:
        an ExpectationStep followed by a correctly configured MaximizationStep.
        """

        # Arrange
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecm.Pipeline")
        ecm = ECM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)
        record_frequency = 5
        num_of_steps = 2

        # Act
        ecm.fit(sample_data, sample_mixture, once_in_iterations=record_frequency)

        # Assert
        args, _ = MockPipeline.call_args
        pipeline_steps, pipeline_breakpointers, pipeline_pruners, pipeline_record_freq = (
            args[0],
            args[1],
            args[2],
            args[3],
        )

        assert pipeline_breakpointers == mock_breakpointers
        assert pipeline_pruners == mock_pruners
        assert pipeline_record_freq == record_frequency

        assert len(pipeline_steps) == num_of_steps
        assert isinstance(pipeline_steps[0], ExpectationStep)
        assert isinstance(pipeline_steps[1], MaximizationStep)

        m_step = pipeline_steps[1]
        assert m_step.optimizer is mock_optimizer
        assert len(m_step.blocks) == sample_mixture.n_components

        for i, (block, component) in enumerate(zip(m_step.blocks, sample_mixture.components)):
            assert block.component_id == i
            assert block.params_to_optimize == component.params_to_optimize
            assert block.maximization_strategy == MaximizationStrategy.QFUNCTION

        assert "rate" not in m_step.blocks[1].params_to_optimize
        assert "loc" in m_step.blocks[1].params_to_optimize

    def test_history_access_lifecycle(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        """
        Tests the full lifecycle of the history property:
        1. Raises AttributeError before `fit` is called.
        2. Becomes accessible after the first `fit` call.
        3. Is overwritten with a new object on a subsequent `fit` call.
        """

        # Arrange
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecm.Pipeline")

        breakpointers = [StopAfterOneIteration()]
        ecm = ECM(breakpointers=breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        # 1. Assert initial state: history access raises a specific error
        with pytest.raises(AttributeError, match="History is not available. Call the 'fit' method first."):
            _ = ecm.history

        # 2. Act & Assert after first call
        mock_pipeline_instance_1 = MockPipeline.return_value
        mock_pipeline_instance_1.history = "first_history_object"

        ecm.fit(sample_data, sample_mixture)

        assert ecm.history == "first_history_object"
        first_history_id = id(ecm.history)

        # 3. Act & Assert after second call
        mock_pipeline_instance_2 = MockPipeline.return_value
        mock_pipeline_instance_2.history = "second_history_object"

        ecm.fit(sample_data, sample_mixture)

        assert ecm.history == "second_history_object"
        second_history_id = id(ecm.history)

        assert first_history_id != second_history_id
