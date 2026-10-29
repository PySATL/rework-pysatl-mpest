"""
Tests for the ECME class, which acts as a facade for the Pipeline.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_mpest.core.mixture import MixtureModel
from pysatl_mpest.estimators.ecme import ECME
from pysatl_mpest.estimators.iterative.steps.block import MaximizationStrategy


class TestECME:
    @pytest.fixture
    def mock_deps(self, mocker):
        return {
            "breakpointers": [mocker.Mock()],
            "pruners": [mocker.Mock()],
            "optimizer": mocker.Mock(),
        }

    @pytest.fixture
    def ecme_instance(self, mock_deps):
        return ECME(
            breakpointers=mock_deps["breakpointers"],
            pruners=mock_deps["pruners"],
            optimizer=mock_deps["optimizer"],
            default_strategy="odl",
        )

    def test_history_raises_attribute_error_before_fit(self, ecme_instance):
        with pytest.raises(AttributeError, match="Call the 'fit' method first"):
            _ = ecme_instance.history

    def test_history_returns_pipeline_history_after_fit(self, ecme_instance, mocker):
        mock_pipeline_cls = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        mock_pipeline_instance = mock_pipeline_cls.return_value

        expected_history = mocker.Mock()
        mock_pipeline_instance.history = expected_history

        mock_mixture = mocker.MagicMock(spec=MixtureModel)
        mock_mixture.n_components = 1

        ecme_instance.fit(X=np.array([]), mixture=mock_mixture)

        assert ecme_instance.history is expected_history

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            (None, set()),
            (1, {1}),
            ([1, 2, 3], {1, 2, 3}),
            ((0, 5), {0, 5}),
            (np.array([1]), {1}),
        ],
    )
    def test_normalize_indices(self, ecme_instance, input_val, expected):
        result = ecme_instance._normalize_indices(input_val)
        assert result == expected
        assert isinstance(result, set)

    @pytest.mark.parametrize(
        "idx, n_components, expected",
        [
            (0, 5, 0),
            (4, 5, 4),
            (-1, 5, 4),
            (-5, 5, 0),
        ],
    )
    def test_resolve_index_valid(self, ecme_instance, idx, n_components, expected):
        assert ecme_instance._resolve_index(idx, n_components) == expected

    @pytest.mark.parametrize(
        "idx, n_components",
        [
            (5, 5),
            (10, 5),
            (-6, 5),
        ],
    )
    def test_resolve_index_out_of_bounds(self, ecme_instance, idx, n_components):
        with pytest.raises(ValueError, match="out of bounds"):
            ecme_instance._resolve_index(idx, n_components)

    def test_fit_indices_intersection_error(self, ecme_instance, mocker):
        mock_mixture = mocker.MagicMock(spec=MixtureModel)
        mock_mixture.n_components = 3

        with pytest.raises(ValueError, match="specified for both"):
            ecme_instance.fit(X=[], mixture=mock_mixture, q_indices_raw=[1], odl_indices_raw=[-2])

    def test_fit_strategy_assignment_and_execution(self, ecme_instance, mocker):
        mock_pipeline_cls = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        mock_max_step_cls = mocker.patch("pysatl_mpest.estimators.ecme.MaximizationStep")
        mock_exp_step_cls = mocker.patch("pysatl_mpest.estimators.ecme.ExpectationStep")

        expected_result = mocker.Mock(spec=MixtureModel)
        mock_pipeline_inst = mock_pipeline_cls.return_value
        mock_pipeline_inst.fit.return_value = expected_result

        mock_mixture = mocker.MagicMock(spec=MixtureModel)
        mock_mixture.n_components = 3

        X_dummy = np.zeros((10, 2))
        expected_once_in_iterations = 5
        expected_number_of_steps = 2
        expected_number_of_blocks = 3

        result = ecme_instance.fit(
            X=X_dummy, mixture=mock_mixture, q_indices_raw=0, odl_indices_raw=1, once_in_iterations=5
        )

        assert result == expected_result

        mock_pipeline_cls.assert_called_once()
        pipeline_args = mock_pipeline_cls.call_args

        assert pipeline_args[0][1] == ecme_instance.breakpointers
        assert pipeline_args[0][2] == ecme_instance.pruners
        assert pipeline_args[0][3] == expected_once_in_iterations

        steps_arg = pipeline_args[0][0]
        assert len(steps_arg) == expected_number_of_steps

        assert steps_arg[0] == mock_exp_step_cls.return_value
        assert steps_arg[1] == mock_max_step_cls.return_value

        mock_max_step_cls.assert_called_once()
        max_step_args = mock_max_step_cls.call_args

        blocks = max_step_args[0][0]
        assert len(blocks) == expected_number_of_blocks

        assert blocks[0].component_id == 0
        assert blocks[0].params_to_optimize is None
        assert blocks[0].maximization_strategy == MaximizationStrategy.QFUNCTION

        assert blocks[1].component_id == 1
        assert blocks[1].params_to_optimize is None
        assert blocks[1].maximization_strategy == MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD

        assert blocks[2].component_id == 2  # noqa: PLR2004
        assert blocks[2].params_to_optimize is None
        assert blocks[2].maximization_strategy == MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD

        mock_pipeline_inst.fit.assert_called_once_with(X_dummy, mock_mixture)

    def test_fit_default_strategy_q_func(self, mocker):
        deps = {
            "breakpointers": [mocker.Mock()],
            "pruners": [],
            "optimizer": mocker.Mock(),
        }
        ecme = ECME(**deps, default_strategy="q-func")

        mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        mock_max_step_cls = mocker.patch("pysatl_mpest.estimators.ecme.MaximizationStep")
        mocker.patch("pysatl_mpest.estimators.ecme.ExpectationStep")

        mock_mixture = mocker.MagicMock(spec=MixtureModel)
        mock_mixture.n_components = 1

        ecme.fit(X=[], mixture=mock_mixture)

        blocks = mock_max_step_cls.call_args[0][0]
        assert blocks[0].maximization_strategy == MaximizationStrategy.QFUNCTION
