import numpy as np
import pytest
from pysatl_mpest.core.mixture import MixtureModel
from pysatl_mpest.estimators.ecme import ECME
from pysatl_mpest.estimators.iterative.steps.block import MaximizationStrategy


class TestECME:
    """Tests for the ECME class using pytest-mock."""

    @pytest.fixture
    def mock_deps(self, mocker):
        """Fixture for creating mock dependencies."""

        return {
            "breakpointers": [mocker.Mock()],
            "pruners": [mocker.Mock()],
            "optimizer": mocker.Mock(),
        }

    @pytest.fixture
    def ecme_instance(self, mock_deps):
        """Fixture for creating an ECME instance."""

        return ECME(
            breakpointers=mock_deps["breakpointers"],
            pruners=mock_deps["pruners"],
            optimizer=mock_deps["optimizer"],
            default_strategy="odl",
        )

    # --- Tests for history property ---

    def test_history_raises_attribute_error_before_fit(self, ecme_instance):
        """Verifies that accessing history raises AttributeError before fit is called."""

        with pytest.raises(AttributeError, match="Call the 'fit' method first"):
            _ = ecme_instance.history

    def test_history_returns_pipeline_history_after_fit(self, ecme_instance, mocker):
        """Verifies that the history property returns the pipeline history after fit."""

        # Mock Pipeline
        mock_pipeline_cls = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        mock_pipeline_instance = mock_pipeline_cls.return_value

        # Set expected history in the pipeline mock
        expected_history = mocker.Mock()
        mock_pipeline_instance.history = expected_history

        # Mock mixture
        mock_mixture = mocker.MagicMock(spec=MixtureModel)
        mock_mixture.n_components = 1
        mock_mixture.__iter__.return_value = iter([mocker.Mock()])

        # Call fit
        ecme_instance.fit(X=np.array([]), mixture=mock_mixture)

        # Verify
        assert ecme_instance.history is expected_history

    # --- Tests for _normalize_indices method ---

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
        """Verifies normalization of various index input types into a set."""

        result = ecme_instance._normalize_indices(input_val)
        assert result == expected
        assert isinstance(result, set)

    # --- Tests for _resolve_index method ---

    @pytest.mark.parametrize(
        "idx, n_components, expected",
        [
            (0, 5, 0),  # Positive index
            (4, 5, 4),  # Boundary positive
            (-1, 5, 4),  # Negative (last)
            (-5, 5, 0),  # Negative (first)
        ],
    )
    def test_resolve_index_valid(self, ecme_instance, idx, n_components, expected):
        """Verifies correct resolution of positive and negative indices."""

        assert ecme_instance._resolve_index(idx, n_components) == expected

    @pytest.mark.parametrize(
        "idx, n_components",
        [
            (5, 5),  # Out of bounds (positive)
            (10, 5),
            (-6, 5),  # Out of bounds (negative)
        ],
    )
    def test_resolve_index_out_of_bounds(self, ecme_instance, idx, n_components):
        """Verifies that ValueError is raised when index is out of bounds."""

        with pytest.raises(ValueError, match="out of bounds"):
            ecme_instance._resolve_index(idx, n_components)

    # --- Tests for fit method logic ---

    def test_fit_indices_intersection_error(self, ecme_instance, mocker):
        """Verifies that an error is raised when indices intersect between q_indices and odl_indices."""

        mock_mixture = mocker.MagicMock(spec=MixtureModel)
        mock_mixture.n_components = 3

        # Index 1 intersects with index -2 (which resolves to 1 when n=3)
        with pytest.raises(ValueError, match="specified for both"):
            ecme_instance.fit(X=[], mixture=mock_mixture, q_indices_raw=[1], odl_indices_raw=[-2])

    def test_fit_strategy_assignment_and_execution(self, ecme_instance, mocker):
        """
        Comprehensive test of fit logic:
        1. Strategy assignment (Q-func, ODL, Default).
        2. Creation of Pipeline with correct steps.
        3. Execution of pipeline.fit and return value verification.
        """

        # --- Prepare Mocks ---
        mock_pipeline_cls = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        mock_max_step_cls = mocker.patch("pysatl_mpest.estimators.ecme.MaximizationStep")
        mock_exp_step_cls = mocker.patch("pysatl_mpest.estimators.ecme.ExpectationStep")

        mock_pipeline_inst = mock_pipeline_cls.return_value
        expected_result = mocker.Mock(spec=MixtureModel)
        mock_pipeline_inst.fit.return_value = expected_result

        # --- Prepare Mixture (3 components) ---
        comp0 = mocker.Mock()
        comp0.params_to_optimize = {"p0"}
        comp1 = mocker.Mock()
        comp1.params_to_optimize = {"p1"}
        comp2 = mocker.Mock()
        comp2.params_to_optimize = {"p2"}

        mock_mixture = mocker.MagicMock(spec=MixtureModel)
        mock_mixture.n_components = 3
        # Setup iteration over mixture
        mock_mixture.__iter__.return_value = iter([comp0, comp1, comp2])
        # Setup item access (for completeness, though logic uses iteration)
        mock_mixture.__getitem__.side_effect = [comp0, comp1, comp2]

        # --- Execute fit ---
        # Scenario:
        # Index 0: Q-Func (explicit)
        # Index 1: ODL (explicit via negative index -2)
        # Index 2: Default (ODL, since fixture sets default="odl")
        X_dummy = np.zeros((10, 2))
        expected_once_in_iterations = 5
        expected_number_of_steps = 2
        expected_number_of_blocks = 3

        result = ecme_instance.fit(
            X=X_dummy, mixture=mock_mixture, q_indices_raw=0, odl_indices_raw=1, once_in_iterations=5
        )

        # --- Assertions ---

        # 1. Verify return value
        assert result == expected_result

        # 2. Verify Pipeline initialization
        mock_pipeline_cls.assert_called_once()
        pipeline_args = mock_pipeline_cls.call_args

        # Verify simple Pipeline arguments
        assert pipeline_args[0][1] == ecme_instance.breakpointers
        assert pipeline_args[0][2] == ecme_instance.pruners
        assert pipeline_args[0][3] == expected_once_in_iterations

        # 3. Verify steps passed to Pipeline
        steps_arg = pipeline_args[0][0]  # List of steps
        assert len(steps_arg) == expected_number_of_steps

        # Verify types of steps (or that they are return values of the classes)
        assert steps_arg[0] == mock_exp_step_cls.return_value
        assert steps_arg[1] == mock_max_step_cls.return_value

        # 4. Detailed check of MaximizationStep configuration (strategies)
        mock_max_step_cls.assert_called_once()
        max_step_args = mock_max_step_cls.call_args

        # The first argument is the list of blocks
        blocks = max_step_args[0][0]
        assert len(blocks) == expected_number_of_blocks

        # Block 0: should be QFUNCTION
        assert blocks[0].component_id == 0
        assert blocks[0].params_to_optimize == {"p0"}
        assert blocks[0].maximization_strategy == MaximizationStrategy.QFUNCTION

        # Block 1: should be ODL
        assert blocks[1].component_id == 1
        assert blocks[1].params_to_optimize == {"p1"}
        assert blocks[1].maximization_strategy == MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD

        # Block 2: should be Default -> ODL
        assert blocks[2].component_id == 2  # noqa: PLR2004
        assert blocks[2].params_to_optimize == {"p2"}
        assert blocks[2].maximization_strategy == MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD

        # 5. Verify pipeline.fit call
        mock_pipeline_inst.fit.assert_called_once_with(X_dummy, mock_mixture)

    def test_fit_default_strategy_q_func(self, mocker):
        """
        Verifies that if default_strategy='q-func', unassigned components
        get the QFUNCTION strategy.
        """

        # Manually create dependencies for this test
        deps = {
            "breakpointers": [mocker.Mock()],
            "pruners": [],
            "optimizer": mocker.Mock(),
        }
        ecme = ECME(**deps, default_strategy="q-func")

        # Mock classes
        mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        mock_max_step_cls = mocker.patch("pysatl_mpest.estimators.ecme.MaximizationStep")
        mocker.patch("pysatl_mpest.estimators.ecme.ExpectationStep")

        # Mixture with 1 component
        comp0 = mocker.Mock()
        mock_mixture = mocker.MagicMock(spec=MixtureModel)
        mock_mixture.n_components = 1
        mock_mixture.__iter__.return_value = iter([comp0])

        # Call fit without explicit indices
        ecme.fit(X=[], mixture=mock_mixture)

        # Verify that the block received QFUNCTION strategy
        blocks = mock_max_step_cls.call_args[0][0]
        assert blocks[0].maximization_strategy == MaximizationStrategy.QFUNCTION
