"""Tests for Parameter descriptor."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_mpest.core import Parameter
from tests.mocks.core.parameter import MockParameterOwner


def test_parameter_default_init() -> None:
    """Test default initialization of Parameter descriptor."""

    param = Parameter()
    assert param.invariant(1.0) is True
    assert param.invariant(-1.0) is True
    assert param.error_message == "Parameter value is not valid."


def test_parameter_custom_init() -> None:
    """Test custom initialization of Parameter descriptor."""

    param = Parameter(invariant=lambda x: x > 0, error_message="Custom error.")
    assert param.invariant(1.0) is True
    assert param.invariant(-1.0) is False
    assert param.error_message == "Custom error."


def test_parameter_set_name() -> None:
    """Test __set_name__ assigns proper names to descriptor."""

    descriptor = MockParameterOwner.__dict__["any_param"]

    assert descriptor.public_name == "any_param"
    assert descriptor.private_name == "_any_param"


def test_parameter_get_class_access() -> None:
    """Test that accessing parameter via class returns the descriptor itself."""

    descriptor = MockParameterOwner.any_param
    assert isinstance(descriptor, Parameter)


def test_parameter_get_instance_access() -> None:
    """Test that accessing parameter via instance returns the value."""

    owner = MockParameterOwner(positive_val=5.0, any_val=-3.0)

    np.testing.assert_allclose(owner.positive_param, 5.0)
    np.testing.assert_allclose(owner.any_param, -3.0)
    assert type(owner.positive_param) is float
    assert type(owner.any_param) is float


def test_parameter_set_valid() -> None:
    """Test setting valid values and casting to dtype."""

    owner = MockParameterOwner(positive_val=5.0, any_val=-3.0)

    owner.positive_param = 10
    np.testing.assert_allclose(owner.positive_param, 10.0)
    assert type(owner.positive_param) is float

    owner.any_param = 2.5
    np.testing.assert_allclose(owner.any_param, 2.5)
    assert type(owner.any_param) is float


def test_parameter_invariant_violation() -> None:
    """Test assigning a value that fails the invariant check."""

    owner = MockParameterOwner(positive_val=5.0, any_val=-3.0)

    with pytest.raises(ValueError, match="Value must be positive."):
        owner.positive_param = 0.0

    with pytest.raises(ValueError, match="Value must be positive."):
        owner.positive_param = -1.5


def test_parameter_fixed_violation() -> None:
    """Test modifying a parameter listed in _fixed_params."""

    owner = MockParameterOwner(positive_val=5.0, any_val=-3.0)
    owner._fixed_params = {"positive_param"}

    with pytest.raises(AttributeError, match="This parameter is fixed."):
        owner.positive_param = 10.0


def test_parameter_uninitialized_access() -> None:
    """Test that accessing an uninitialized parameter raises AttributeError."""

    owner = MockParameterOwner.__new__(MockParameterOwner)
    with pytest.raises(AttributeError):
        _ = owner.positive_param


def test_parameter_missing_dtype_fallback() -> None:
    """Test that parameter falls back to float if owner lacks dtype attribute."""

    class FallbackOwner:
        param = Parameter()

    owner = FallbackOwner()
    owner.param = 5

    assert type(owner.param) is float
    np.testing.assert_allclose(owner.param, 5.0)


@pytest.mark.parametrize("invalid_val", ["invalid_string", {"a": 1}])
def test_parameter_incompatible_types(invalid_val: object) -> None:
    """Test that assigning incompatible types raises conversion errors."""

    owner = MockParameterOwner(positive_val=5.0, any_val=-3.0)

    with pytest.raises((ValueError, TypeError)):
        owner.any_param = invalid_val  # type: ignore


@pytest.mark.parametrize(
    "edge_val",
    [
        0.0,
        -0.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_parameter_edge_values(edge_val: float) -> None:
    """Test parameter invariants against numerical edge cases."""

    owner = MockParameterOwner(positive_val=5.0, any_val=-3.0)

    val_casted = float(edge_val)
    # Suppress RuntimeWarning for np.nan > 0
    with np.errstate(invalid="ignore"):
        expected_pass = bool(val_casted > 0)

    if expected_pass:
        owner.positive_param = edge_val
        assert type(owner.positive_param) is float
    else:
        with pytest.raises(ValueError, match="Value must be positive."):
            owner.positive_param = edge_val


def test_parameter_tiny_value() -> None:
    """Test that the smallest normal positive value for the specific dtype passes the invariant."""

    owner = MockParameterOwner(positive_val=5.0, any_val=-3.0)
    tiny_val = np.finfo(float).tiny

    # This should pass since tiny_val > 0 is True for its respective dtype
    owner.positive_param = tiny_val
    assert type(owner.positive_param) is float
    np.testing.assert_allclose(owner.positive_param, tiny_val)


def test_parameter_array_assignment() -> None:
    """Test that assigning an array to a parameter raises a TypeError."""

    owner = MockParameterOwner(positive_val=5.0, any_val=-3.0)

    with pytest.raises(TypeError, match=r"must be a scalar, got array of shape \(2,\)."):
        owner.positive_param = [1.0, 2.0]  # type: ignore
