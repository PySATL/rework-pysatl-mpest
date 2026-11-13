"""Tests for the ECME class, which acts as a facade for the Pipeline."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators import ECME
from pysatl_mpest.estimators.iterative import (
    Breakpointer,
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    PipelineState,
    WeightsStep,
)
from pysatl_mpest.optimizers import Optimizer
from pytest_mock import MockerFixture

from tests.helpers.families import family

Q = MaximizationStrategy.QFUNCTION
ODL = MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD


class StopOnSecondCheck(Breakpointer):
    """Return true on the second check; the mocked pipeline never calls it."""

    def __init__(self):
        self.called = False

    def check(self, state: PipelineState) -> bool:
        if self.called:
            return True
        self.called = True
        return False


@pytest.fixture
def mock_optimizer(mocker: MockerFixture) -> Optimizer:
    return mocker.Mock(spec=Optimizer)


@pytest.fixture
def mock_breakpointers() -> list[Breakpointer]:
    return [StopOnSecondCheck()]


@pytest.fixture
def ecme(mock_optimizer: Optimizer, mock_breakpointers: list[Breakpointer]) -> ECME:
    return ECME(breakpointers=mock_breakpointers, pruners=[], optimizer=mock_optimizer, default_strategy="odl")


@pytest.fixture
def three_normal_mixture() -> MixtureModel:
    # All components have ``mu`` and ``sigma`` free, so parameter groups can split them.
    components = [
        family(FamilyName.NORMAL)(mu=0.0, sigma=1.0),
        family(FamilyName.NORMAL)(mu=5.0, sigma=2.0),
        family(FamilyName.NORMAL)(mu=-5.0, sigma=1.5),
    ]
    return MixtureModel(components)


@pytest.fixture
def sample_data() -> np.ndarray:
    return np.array([1, 2, 3, 11, 12, 13], dtype=np.float64)


def fitted_m_step_blocks(mocker: MockerFixture, fit_call) -> list:
    """Run ``fit_call`` with a mocked Pipeline and return the blocks of its MaximizationStep."""
    MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
    fit_call()
    m_step = MockPipeline.call_args[0][0][2]
    return m_step.blocks


class TestInitialization:
    def test_copies_sequences_and_keeps_optimizer(
        self, mock_optimizer: Optimizer, mock_breakpointers: list[Breakpointer]
    ):
        pruners = []
        ecm_e = ECME(
            breakpointers=mock_breakpointers, pruners=pruners, optimizer=mock_optimizer, default_strategy="q-func"
        )

        assert ecm_e.breakpointers is not mock_breakpointers
        assert ecm_e.breakpointers == mock_breakpointers
        assert ecm_e.pruners is not pruners
        assert ecm_e.optimizer is mock_optimizer
        assert ecm_e.default_strategy_name == "q-func"
        assert ecm_e._history is None

    def test_rejects_an_unknown_default_strategy(self, mock_optimizer: Optimizer):
        with pytest.raises(ValueError, match="Unknown default_strategy"):
            ECME(breakpointers=[StopOnSecondCheck()], pruners=[], optimizer=mock_optimizer, default_strategy="bad")


class TestStrategyIndices:
    """Index arguments of ``fit`` select strategies and reject invalid indices."""

    @pytest.mark.parametrize("q_indices", [-1, [-1], [-2, 0]], ids=["int", "list", "list-with-negative"])
    def test_negative_q_indices_raise_value_error(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
        q_indices,
    ):
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")

        with pytest.raises(ValueError, match="negative"):
            ecme.fit(sample_data, three_normal_mixture, q_indices_raw=q_indices)

        MockPipeline.assert_not_called()

    @pytest.mark.parametrize("odl_indices", [-1, [-1]], ids=["int", "list"])
    def test_negative_odl_indices_raise_value_error(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
        odl_indices,
    ):
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")

        with pytest.raises(ValueError, match="negative"):
            ecme.fit(sample_data, three_normal_mixture, odl_indices_raw=odl_indices)

        MockPipeline.assert_not_called()

    @pytest.mark.parametrize("indices", [3, [0, 5]], ids=["at-bound", "list"])
    def test_out_of_bounds_indices_raise_value_error(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
        indices,
    ):
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")

        with pytest.raises(ValueError, match="out of bounds"):
            ecme.fit(sample_data, three_normal_mixture, q_indices_raw=indices)

        with pytest.raises(ValueError, match="out of bounds"):
            ecme.fit(sample_data, three_normal_mixture, odl_indices_raw=indices)

        MockPipeline.assert_not_called()

    def test_overlapping_strategy_indices_raise_value_error(
        self, mocker: MockerFixture, ecme: ECME, three_normal_mixture: MixtureModel, sample_data: np.ndarray
    ):
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")

        with pytest.raises(ValueError, match="specified for both"):
            ecme.fit(sample_data, three_normal_mixture, q_indices_raw=[0, 1], odl_indices_raw=1)

        MockPipeline.assert_not_called()


class TestPipeline:
    def test_returns_pipeline_result(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        mock_pipeline_instance = MockPipeline.return_value
        mock_pipeline_instance.fit.return_value = three_normal_mixture

        result = ecme.fit(sample_data, three_normal_mixture)

        MockPipeline.assert_called_once()
        mock_pipeline_instance.fit.assert_called_once_with(sample_data, three_normal_mixture)
        assert result is three_normal_mixture

    def test_passes_steps_and_configuration_to_pipeline(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        record_frequency = 5
        num_of_steps = 3

        ecme.fit(sample_data, three_normal_mixture, once_in_iterations=record_frequency)

        args = MockPipeline.call_args[0]
        pipeline_steps, pipeline_breakpointers, pipeline_pruners, pipeline_record_freq = (
            args[0],
            args[1],
            args[2],
            args[3],
        )

        assert pipeline_breakpointers == mock_breakpointers
        assert pipeline_pruners == []
        assert pipeline_record_freq == record_frequency

        assert len(pipeline_steps) == num_of_steps
        assert isinstance(pipeline_steps[0], ExpectationStep)
        assert isinstance(pipeline_steps[1], WeightsStep)
        assert isinstance(pipeline_steps[2], MaximizationStep)
        assert pipeline_steps[2].optimizer is mock_optimizer

    def test_builds_q_chains_before_odl_chains(
        self, mocker: MockerFixture, ecme: ECME, three_normal_mixture: MixtureModel, sample_data: np.ndarray
    ):
        # Component 1 is assigned the Q-function strategy; the order of blocks differs
        # from the component order: all Q chains come first, then the ODL chains.
        blocks = fitted_m_step_blocks(mocker, lambda: ecme.fit(sample_data, three_normal_mixture, q_indices_raw=1))

        assert [(block.component_id, block.maximization_strategy) for block in blocks] == [
            (1, Q),
            (0, ODL),
            (2, ODL),
        ]
        assert all(block.params_to_optimize is None for block in blocks)

    def test_explicit_odl_indices_leave_the_rest_to_the_default_strategy(
        self, mocker: MockerFixture, ecme: ECME, three_normal_mixture: MixtureModel, sample_data: np.ndarray
    ):
        blocks = fitted_m_step_blocks(
            mocker, lambda: ecme.fit(sample_data, three_normal_mixture, odl_indices_raw=[0, 2])
        )

        assert [(block.component_id, block.maximization_strategy) for block in blocks] == [
            (0, ODL),
            (1, ODL),
            (2, ODL),
        ]

    def test_uses_q_function_as_default_strategy(
        self, mocker: MockerFixture, mock_optimizer: Optimizer, mock_breakpointers: list[Breakpointer]
    ):
        ecm_e = ECME(
            breakpointers=mock_breakpointers,
            pruners=[],
            optimizer=mock_optimizer,
            default_strategy="q-func",
        )
        mock_mixture = mocker.MagicMock(spec=MixtureModel)
        mock_mixture.n_components = 2

        blocks = fitted_m_step_blocks(mocker, lambda: ecm_e.fit(X=np.zeros(5), mixture=mock_mixture))

        assert [(block.component_id, block.maximization_strategy) for block in blocks] == [(0, Q), (1, Q)]


class TestParameterGroups:
    """Blocks built from ``parameter_groups`` and validation of the argument."""

    def test_groups_split_chains_for_q_and_odl_components(
        self, mocker: MockerFixture, ecme: ECME, three_normal_mixture: MixtureModel, sample_data: np.ndarray
    ):
        blocks = fitted_m_step_blocks(
            mocker,
            lambda: ecme.fit(
                sample_data,
                three_normal_mixture,
                q_indices_raw=1,
                parameter_groups={0: [{"mu"}, {"sigma"}], 1: [{"mu"}, {"sigma"}]},
            ),
        )

        assert [(block.component_id, block.params_to_optimize, block.maximization_strategy) for block in blocks] == [
            (1, {"mu"}, Q),
            (1, {"sigma"}, Q),
            (0, {"mu"}, ODL),
            (0, {"sigma"}, ODL),
            (2, None, ODL),
        ]

    def test_groups_apply_to_the_default_strategy_components(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        ecm_e = ECME(
            breakpointers=mock_breakpointers,
            pruners=[],
            optimizer=mock_optimizer,
            default_strategy="q-func",
        )

        blocks = fitted_m_step_blocks(
            mocker,
            lambda: ecm_e.fit(
                sample_data, three_normal_mixture, odl_indices_raw=2, parameter_groups={0: [{"mu"}, {"sigma"}]}
            ),
        )

        assert [(block.component_id, block.params_to_optimize, block.maximization_strategy) for block in blocks] == [
            (0, {"mu"}, Q),
            (0, {"sigma"}, Q),
            (1, None, Q),
            (2, None, ODL),
        ]

    @pytest.mark.parametrize(
        "parameter_groups, match",
        [
            ({-1: [{"mu"}]}, "negative"),
            ({3: [{"mu"}]}, "out of bounds"),
            ({0: []}, "must not be empty"),
            ({0: [{"mu"}, set()]}, "empty group"),
            ({0: [{"mu", "sigma"}, {"sigma"}]}, "overlap"),
            ({0: [{"mu"}]}, "not covered"),
            ({0: [{"mu", "sigma", "x"}]}, "are not parameters"),
        ],
        ids=["negative-key", "out-of-bounds", "empty-list", "empty-group", "overlap", "uncovered", "unknown"],
    )
    def test_invalid_parameter_groups_raise_before_the_pipeline_is_created(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
        parameter_groups: dict,
        match: str,
    ):
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")

        with pytest.raises(ValueError, match=match):
            ecme.fit(sample_data, three_normal_mixture, parameter_groups=parameter_groups)

        MockPipeline.assert_not_called()

    def test_parameter_groups_is_keyword_only(
        self, ecme: ECME, three_normal_mixture: MixtureModel, sample_data: np.ndarray
    ):
        with pytest.raises(TypeError):
            ecme.fit(sample_data, three_normal_mixture, None, None, 1, {0: [{"mu"}, {"sigma"}]})


class TestHistory:
    def test_raises_attribute_error_before_fit(self, ecme: ECME):
        with pytest.raises(AttributeError, match="Call the 'fit' method first"):
            _ = ecme.history

    def test_returns_pipeline_history_after_fit(
        self, mocker: MockerFixture, ecme: ECME, three_normal_mixture: MixtureModel, sample_data: np.ndarray
    ):
        mock_pipeline_cls = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        mock_pipeline_instance = mock_pipeline_cls.return_value

        expected_history = mocker.Mock()
        mock_pipeline_instance.history = expected_history

        ecme.fit(sample_data, three_normal_mixture)

        assert ecme.history is expected_history
