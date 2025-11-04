"""Tests for ContinuousDistribution class"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from copy import copy
from typing import ClassVar

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from rework_pysatl_mpest.core import Parameter
from rework_pysatl_mpest.distributions import ContinuousDistribution, Exponential, Normal

# Dummy distribution classes
# --------------------------


class DummyDistribution(ContinuousDistribution):
    """
    A concrete implementation of ContinuousDistribution for testing purposes.
    This class implements all abstract methods, allowing us to instantiate it
    and test the non-abstract methods of the base class.
    """

    param1 = Parameter()
    param2 = Parameter()

    def __init__(self, param1: float = 1.0, param2: float = 2.0, name: str = "Dummy", dtype: np.floating = np.float64):
        """Initializes with two simple parameters."""

        super().__init__(dtype=dtype)
        self.param1 = param1
        self.param2 = param2
        self._name = name

    @property
    def name(self) -> str:
        return self._name

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
def dummy_float32_dist() -> DummyDistribution:
    return DummyDistribution(param1=10.0, param2=20.0, dtype=np.float32)


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

    def test_get_params_vector_returns_correct_types(self, dummy_float32_dist: DummyDistribution):
        """Tests that get_params_vector returns a list of scalars with the correct dtype."""
        param_names = ["param1", "param2"]
        vector = dummy_float32_dist.get_params_vector(param_names)
        for param in vector:
            assert isinstance(param, np.float32)

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

    @pytest.mark.parametrize(
        "param_names, vector_to_set",
        [
            (["param1", "param2"], [100.0, 200.0]),
            (("param2",), (99.0,)),
            (["param1", "param2"], np.array([1.5, 2.5])),
        ],
    )
    def test_set_params_from_vector_correct_dtype(
        self, dummy_float32_dist: DummyDistribution, param_names, vector_to_set
    ):
        """Tests setting parameter values from a vector."""

        dummy_float32_dist.set_params_from_vector(param_names, vector_to_set)

        for param_name in param_names:
            param_value = getattr(dummy_float32_dist, param_name)
            assert isinstance(param_value, np.float32)

    # Test to_dtype method
    # ----------------------------------

    def test_to_dtype_successful_conversion(self, dummy_dist: DummyDistribution):
        """
        Tests that _to_dtype creates a new instance with the correct new dtype
        and that the original instance remains unchanged.
        """
        assert dummy_dist.dtype == np.float64
        for param in dummy_dist.params:
            assert isinstance(getattr(dummy_dist, param), np.float64)

        target_dtype = np.float32
        new_dist = dummy_dist.astype(target_dtype)

        assert new_dist is not dummy_dist
        assert new_dist != dummy_dist

        assert new_dist.dtype == np.float32
        for param in new_dist.params:
            assert isinstance(getattr(new_dist, param), target_dtype)
            assert getattr(new_dist, param) == np.float32(getattr(dummy_dist, param))

        # original instance remains unchanged
        assert dummy_dist.dtype == np.float64
        for param in dummy_dist.params:
            assert isinstance(getattr(dummy_dist, param), np.float64)

    def test_to_dtype_returns_self_if_same_dtype(self, dummy_dist: DummyDistribution):
        """
        Tests that _to_dtype returns the same instance if the target dtype
        is identical to the current one, avoiding unnecessary copying.
        """
        assert dummy_dist.dtype == np.float64

        same_dtype_dist = dummy_dist.astype(np.float64)

        assert same_dtype_dist is dummy_dist

    def test_to_dtype_preserves_fixed_params(self, dummy_dist: DummyDistribution):
        """
        Tests that the set of fixed parameters is correctly copied to the
        new instance after dtype conversion.
        """
        dummy_dist.fix_param("param1")
        assert "param1" in dummy_dist._fixed_params

        new_dist = dummy_dist.astype(np.float32)

        assert new_dist.dtype == np.float32
        for param in new_dist.params:
            assert isinstance(getattr(new_dist, param), np.float32)
            assert getattr(new_dist, param) == np.float32(getattr(dummy_dist, param))

        assert "param1" in new_dist._fixed_params
        assert new_dist._fixed_params == dummy_dist._fixed_params

        assert new_dist._fixed_params is not dummy_dist._fixed_params


class TestContinuousDistributionCopying:
    """Tests the __copy__ method implementation for ContinuousDistribution."""

    def test_copy_creates_new_equal_instance(self, dummy_dist: DummyDistribution):
        """Tests that copy.copy() creates a new instance that is equal to the original."""

        copied_dist = copy(dummy_dist)

        assert copied_dist is not dummy_dist
        assert copied_dist == dummy_dist

    def test_copy_is_independent(self, dummy_dist: DummyDistribution):
        """Tests that modifying the copied instance does not affect the original."""

        MODIFIED_PARAM_VALUE = 999.0
        copied_dist = copy(dummy_dist)

        original_param1_value = dummy_dist.param1

        # Modify a parameter in the copied distribution
        copied_dist.param1 = MODIFIED_PARAM_VALUE

        # Assert that the original object's parameter remains unchanged
        assert dummy_dist.param1 == original_param1_value
        assert copied_dist.param1 == MODIFIED_PARAM_VALUE

    def test_copy_replicates_fixed_params_independently(self, dummy_dist: DummyDistribution):
        """Tests that the set of fixed parameters is also copied independently."""

        dummy_dist.fix_param("param1")
        copied_dist = copy(dummy_dist)

        assert copied_dist._fixed_params == {"param1"}

        # Modify the copy's fixed params
        copied_dist.unfix_param("param1")

        # Ensure the original is unchanged
        assert dummy_dist._fixed_params != copied_dist._fixed_params


class TestContinuousDistributionComparison:
    """Tests the __eq__ and __hash__ methods of the ContinuousDistribution base class."""

    def test_eq_same_type_and_params(self):
        """Tests that two instances with the same type and parameters are equal."""

        d1 = DummyDistribution(param1=1.0, param2=2.0)
        d2 = DummyDistribution(param1=1.0, param2=2.0)
        assert d1 == d2

    def test_neq_different_params(self):
        """Tests that two instances with different parameters are not equal."""

        d1 = DummyDistribution(param1=1.0, param2=2.0)
        d2 = DummyDistribution(param1=99.0, param2=2.0)
        assert d1 != d2

    def test_neq_different_type(self):
        """Tests that two instances with different types are not equal."""

        d1 = Normal(loc=0.0, scale=1.0)
        d2 = Exponential(loc=0.0, rate=1.0)
        assert d1 != d2

    def test_neq_other_object(self):
        """Tests that a distribution instance is not equal to an object of a different class."""

        d1 = DummyDistribution(param1=1.0, param2=2.0)
        other = "not_a_distribution"
        assert d1 != other

    def test_neq_different_dtype(self):
        """Tests that two instances with different dtypes are not equal."""

        d1 = DummyDistribution(param1=1.0, param2=2.0)
        d2 = DummyDistribution(param1=1.0, param2=2.0, dtype=np.float32)
        assert d1 != d2

    def test_hash_consistency(self):
        """Tests that two equal distribution instances have the same hash."""

        d1 = DummyDistribution(param1=1.0, param2=2.0)
        d2 = DummyDistribution(param1=1.0, param2=2.0)
        assert d1 == d2
        assert hash(d1) == hash(d2)

    def test_hash_inequality(self):
        """Tests that two non-equal distribution instances have different hashes."""

        d1 = DummyDistribution(param1=1.0, param2=2.0)
        d2 = DummyDistribution(param1=99.0, param2=2.0)
        assert d1 != d2
        assert hash(d1) != hash(d2)

    def test_hash_inequality_name(self):
        """Tests that two non-equal type distribution instances have different hashes."""

        d1 = DummyDistribution(param1=1.0, param2=2.0)
        d2 = DummyDistribution(param1=1.0, param2=2.0, name="Dummy2")
        assert d1 != d2
        assert hash(d1) != hash(d2)

    def test_hash_inequality_dtype(self):
        """Tests that two non-equal type distribution instances have different hashes."""

        d1 = DummyDistribution(param1=1.0, param2=2.0)
        d2 = DummyDistribution(param1=1.0, param2=2.0, dtype=np.float32)
        assert d1 != d2
        assert hash(d1) != hash(d2)


class DTypeHandlingMixin:
    """A test mixin to verify correct dtype handling in all subclasses of ContinuousDistribution."""

    distribution_class: ClassVar[type[ContinuousDistribution] | None] = None
    default_params: ClassVar[dict] = {}

    # Tests initialization
    # --------------------

    def check_init_with_dtype_sets_correct_types(self, dtype):
        """Tests that the constructor and the Parameter descriptor correctly cast parameter types."""

        dist = self.distribution_class(**self.default_params, dtype=dtype)

        assert dist.dtype == dtype

        for param_name in self.default_params:
            param_value = getattr(dist, param_name)
            assert isinstance(param_value, dtype)

    # Tests generate
    # --------------------

    def check_generate_returns_correct_dtype(self, size, dtype):
        """Tests the dtype of the array returned by the generate method."""
        dist = self.distribution_class(**self.default_params, dtype=dtype)

        result = dist.generate(size=size)

        assert isinstance(result, np.ndarray)
        assert result.dtype == dtype

    # Tests calculations
    # --------------------

    def check_methods_taking_x_return_correct_dtype(self, method_name, x_data, dtype):
        """Helper method with the logic for testing methods that take X."""
        dist = self.distribution_class(**self.default_params, dtype=dtype)
        method_to_test = getattr(dist, method_name)
        result = method_to_test(x_data)
        assert isinstance(result, np.ndarray)
        assert result.dtype == dtype

    def check_ppf_returns_correct_dtype(self, p_data, dtype):
        """Helper method with the logic for testing the ppf method."""
        dist = self.distribution_class(**self.default_params, dtype=dtype)
        result = dist.ppf(p_data)
        assert isinstance(result, np.ndarray)
        assert result.dtype == dtype

    def check_dlog_methods_returns_correct_dtype(self, x_data, method_name, dtype):
        """Tests that each partial derivative method (_dlog_*) returns a NumPy array with the correct dtype."""

        dist = self.distribution_class(**self.default_params, dtype=dtype)
        method = getattr(dist, method_name)

        assert method(x_data).dtype == dtype
