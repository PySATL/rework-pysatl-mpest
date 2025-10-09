"""Tests for the ECM class, which acts as a facade for the Pipeline."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from pytest_mock import MockerFixture
from rework_pysatl_mpest.core import MixtureModel
from rework_pysatl_mpest.distributions import Exponential, Normal
from rework_pysatl_mpest.estimators import ECM
from rework_pysatl_mpest.estimators.iterative import (
    Breakpointer,
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    PipelineState,
    Pruner,
)
from rework_pysatl_mpest.optimizers import Optimizer

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


@pytest.fixture
def sample_mixture() -> MixtureModel:
    """Provides a mixture model with two components for testing."""

    components = [Normal(loc=0, scale=1), Exponential(loc=10, rate=1)]
    # Fix a parameter in one component to test if `params_to_optimize` is correctly used.
    components[1].fix_param("rate")
    return MixtureModel(components, weights=[0.4, 0.6])


@pytest.fixture
def sample_data() -> np.ndarray:
    """Provides a simple NumPy array of data."""

    return np.array([1, 2, 3, 11, 12, 13])


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

        assert ecm.breakpointers is not mock_breakpointers  # Should be a new list
        assert ecm.breakpointers == mock_breakpointers
        assert ecm.pruners is not mock_pruners
        assert ecm.pruners == mock_pruners
        assert ecm.optimizer is mock_optimizer


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
        MockPipeline = mocker.patch("rework_pysatl_mpest.estimators.ecm.Pipeline")
        mock_pipeline_instance = MockPipeline.return_value
        mock_pipeline_instance.fit.return_value = sample_mixture  # Return something to be the result

        ecm = ECM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        # Act
        result = ecm.fit(sample_data, sample_mixture)

        # Assert
        MockPipeline.assert_called_once()
        mock_pipeline_instance.fit.assert_called_once_with(sample_data, sample_mixture)
        assert result is sample_mixture

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
        MockPipeline = mocker.patch("rework_pysatl_mpest.estimators.ecm.Pipeline")
        ecm = ECM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)
        log_frequency = 5
        num_of_steps = 2

        # Act
        ecm.fit(sample_data, sample_mixture, once_in_iterations=log_frequency)

        # Assert
        args, _ = MockPipeline.call_args
        pipeline_steps, pipeline_breakpointers, pipeline_pruners, pipeline_log_freq = (
            args[0],
            args[1],
            args[2],
            args[3],
        )

        # 1. Check general pipeline configuration
        assert pipeline_breakpointers == mock_breakpointers
        assert pipeline_pruners == mock_pruners
        assert pipeline_log_freq == log_frequency

        # 2. Check the sequence and types of steps
        assert len(pipeline_steps) == num_of_steps
        assert isinstance(pipeline_steps[0], ExpectationStep)
        assert isinstance(pipeline_steps[1], MaximizationStep)

        # 3. Deeply inspect the MaximizationStep configuration
        m_step = pipeline_steps[1]
        assert m_step.optimizer is mock_optimizer
        assert len(m_step.blocks) == sample_mixture.n_components

        # 4. Verify each OptimizationBlock corresponds to a component and uses QFUNCTION
        for i, (block, component) in enumerate(zip(m_step.blocks, sample_mixture.components)):
            assert block.component_id == i
            assert block.maximization_strategy == MaximizationStrategy.QFUNCTION
            assert block.params_to_optimize == component.params_to_optimize

        # Check that the fixed parameter was correctly excluded
        assert "rate" not in m_step.blocks[1].params_to_optimize
        assert "loc" in m_step.blocks[1].params_to_optimize

    def test_fit_sets_and_overwrites_logger(
        self,
        mock_optimizer: Optimizer,
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        """
        Tests that the `logger` attribute is created after `fit` runs, and that it is
        replaced with a new logger instance on a subsequent `fit` call.
        """

        # Arrange
        breakpointers = [StopAfterOneIteration()]
        ecm = ECM(breakpointers=breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        # Assert initial state
        assert not hasattr(ecm, "logger")

        # Act 1: First call to fit()
        ecm.fit(sample_data, sample_mixture)

        # Assert 1: Logger now exists
        assert hasattr(ecm, "logger")
        assert len(ecm.logger) > 0
        first_logger_id = id(ecm.logger)

        # Act 2: Second call to fit()
        ecm.fit(sample_data, sample_mixture)

        # Assert 2: Logger still exists but is a new object
        assert hasattr(ecm, "logger")
        second_logger_id = id(ecm.logger)

        assert first_logger_id != second_logger_id
        assert len(ecm.logger) > 0
