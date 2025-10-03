"""Tests for ContinuousDistribution class"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution

# Dummy distribution classes
# --------------------------


class DummyDistribution(ContinuousDistribution):
    """
    A concrete implementation of ContinuousDistribution for testing purposes.
    This class implements all abstract methods, allowing us to instantiate it
    and test the non-abstract methods of the base class.
    """

    def __init__(self, param1: float = 1.0, param2: float = 2.0):
        """Initializes with two simple parameters."""

        super().__init__()
        self._param1 = param1
        self._param2 = param2

    @property
    def param1(self):
        return self._param1

    @param1.setter
    def param1(self, value):
        self._param1 = value

    @property
    def param2(self):
        return self._param2

    @param2.setter
    def param2(self, value):
        self._param2 = value

    @property
    def name(self) -> str:
        return "Dummy"

    @property
    def params(self) -> set[str]:
        return {"param1", "param2"}

    def pdf(self, X: ArrayLike) -> NDArray[np.float64]:
        return np.ones_like(np.asarray(X), dtype=float)

    def ppf(self, P: ArrayLike) -> NDArray[np.float64]:
        return np.ones_like(np.asarray(P), dtype=float)

    def lpdf(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        A predictable lpdf is needed to test q_function.
        Returns log(X) for simplicity. Handles X=0 to avoid -inf.
        """

        X = np.asarray(X, dtype=float)
        return np.log1p(X)

    def log_gradients(self, X: ArrayLike) -> NDArray[np.float64]:
        X = np.atleast_1d(X)
        num_params = len(self.params_to_optimize)
        return np.zeros((len(X), num_params))

    def generate(self, size: int) -> NDArray[np.float64]:
        return np.arange(size, dtype=float)


class DummyInfLpdfDistribution(DummyDistribution):
    def lpdf(self, X: ArrayLike) -> NDArray[np.float64]:
        """Returns -inf if X < 0, otherwise works like the parent."""

        X = np.asarray(X, dtype=float)
        return np.where(X < 0, -np.inf, np.log1p(X))


# Fixtures
# --------


@pytest.fixture
def dummy_dist() -> DummyDistribution:
    """Provides a fresh instance of DummyDistribution for each test."""

    return DummyDistribution(param1=10.0, param2=20.0)


@pytest.fixture
def dummy_inf_dist() -> DummyInfLpdfDistribution:
    return DummyInfLpdfDistribution()


# Tests
# -----


class TestContinuousDistribution:
    """
    Tests the functionality of the ContinuousDistribution abstract base class
    using a concrete DummyDistribution implementation.
    """

    # Tests initialization
    # --------------------

    def test_initialization(self, dummy_dist: DummyDistribution):
        """Tests that the `_fixed_params` attribute is correctly initialized."""

        assert hasattr(dummy_dist, "_fixed_params")
        assert isinstance(dummy_dist._fixed_params, set)
        assert dummy_dist._fixed_params == set()

    # Test fix_params, unfix_params methods
    # -------------------------------------

    def test_fix_param_success(self, dummy_dist: DummyDistribution):
        """Tests that a parameter can be successfully fixed."""

        assert "param1" not in dummy_dist._fixed_params
        dummy_dist.fix_param("param1")
        assert "param1" in dummy_dist._fixed_params

    def test_fix_param_invalid_name_raises_error(self, dummy_dist: DummyDistribution):
        """Tests that fixing a non-existent parameter raises a ValueError."""

        with pytest.raises(ValueError, match="Parameter 'invalid_param' does not exist"):
            dummy_dist.fix_param("invalid_param")

    def test_unfix_param_success(self, dummy_dist: DummyDistribution):
        """Tests that a fixed parameter can be successfully unfixed."""

        dummy_dist.fix_param("param1")
        assert "param1" in dummy_dist._fixed_params
        dummy_dist.unfix_param("param1")
        assert "param1" not in dummy_dist._fixed_params

    def test_unfix_param_does_nothing_if_not_fixed(self, dummy_dist: DummyDistribution):
        """Tests that unfixing a parameter that was not fixed does not raise an error."""

        initial_fixed = dummy_dist._fixed_params.copy()
        dummy_dist.unfix_param("param2")
        dummy_dist.unfix_param("non_existent_param")
        assert dummy_dist._fixed_params == initial_fixed

    def test_params_to_optimize_property(self, dummy_dist: DummyDistribution):
        """Tests that the `params_to_optimize` property works correctly."""

        assert dummy_dist.params_to_optimize == {"param1", "param2"}

        dummy_dist.fix_param("param1")
        assert dummy_dist.params_to_optimize == {"param2"}

        dummy_dist.fix_param("param2")
        assert dummy_dist.params_to_optimize == set()

        dummy_dist.unfix_param("param1")
        assert dummy_dist.params_to_optimize == {"param1"}

    # Test get_params_vector method
    # -----------------------------

    @pytest.mark.parametrize(
        "param_names, expected_vector",
        [
            (["param1", "param2"], [10.0, 20.0]),
            (("param2", "param1"), [20.0, 10.0]),
            (["param1"], [10.0]),
        ],
    )
    def test_get_params_vector_success(self, dummy_dist: DummyDistribution, param_names, expected_vector):
        """Tests retrieving parameter values as a vector."""

        vector = dummy_dist.get_params_vector(param_names)
        assert isinstance(vector, list)
        assert np.array_equal(vector, expected_vector)

    def test_get_params_vector_invalid_name_raises_error(self, dummy_dist: DummyDistribution):
        """Tests that get_params_vector raises an error for invalid parameter names."""

        with pytest.raises(ValueError, match="Invalid parameter names provided"):
            dummy_dist.get_params_vector(["param1", "invalid_param"])

    # Test set_params_from_vector method
    # ----------------------------------

    @pytest.mark.parametrize(
        "param_names, vector_to_set",
        [
            (["param1", "param2"], [100.0, 200.0]),
            (("param2",), (99.0,)),
            (["param1", "param2"], np.array([1.5, 2.5])),
        ],
    )
    def test_set_params_from_vector_success(self, dummy_dist: DummyDistribution, param_names, vector_to_set):
        """Tests setting parameter values from a vector."""

        dummy_dist.set_params_from_vector(param_names, vector_to_set)

        retrieved_vector = dummy_dist.get_params_vector(param_names)
        assert np.array_equal(retrieved_vector, np.asarray(vector_to_set))

    def test_set_params_from_vector_order_is_respected(self, dummy_dist: DummyDistribution):
        """Tests that the order of parameters in the list is respected during setting."""

        param2, param1 = 55.0, 44.0

        dummy_dist.set_params_from_vector(["param2", "param1"], [55.0, 44.0])
        assert dummy_dist.param1 == param1
        assert dummy_dist.param2 == param2

    @pytest.mark.parametrize(
        "param_names, vector, error_msg_match",
        [
            (["param1", "invalid"], [1.0, 2.0], "Invalid parameter names provided"),
            (["param1", "param2"], [1.0], "The number of parameter names must match the number of values"),
            (["param1"], [1.0, 2.0], "The number of parameter names must match the number of values"),
        ],
    )
    def test_set_params_from_vector_raises_errors(
        self, dummy_dist: DummyDistribution, param_names, vector, error_msg_match
    ):
        """Tests that set_params_from_vector raises appropriate ValueErrors."""

        with pytest.raises(ValueError, match=error_msg_match):
            dummy_dist.set_params_from_vector(param_names, vector)
