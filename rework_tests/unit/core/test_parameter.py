"""Tests for Parameter class"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from rework_pysatl_mpest.core import Parameter


class _OwnerClass:
    """
    A helper class to test the Parameter descriptor.
    It simulates a class (like a distribution) that uses Parameter instances
    as attributes.
    """

    positive_param = Parameter(invariant=lambda x: x > 0, error_message="Value must be positive.")
    any_param = Parameter()

    def __init__(self, positive_val: float, any_val: float):
        """
        Initializes the owner class and its parameters.
        Also initializes a set to keep track of fixed parameters.
        """
        self._fixed_params: set[str] = set()
        self.positive_param = positive_val
        self.any_param = any_val


class _OwnerClassWithDType:
    """A helper class that has a dtype attribute."""

    positive_param = Parameter(invariant=lambda x: x > 0, error_message="Value must be positive.")
    any_param = Parameter()

    def __init__(self, positive_val: float, any_val: float, dtype=np.float64):
        """
        Initializes the owner class and its parameters.
        Also initializes a set to keep track of fixed parameters.
        """
        self.dtype = dtype
        self._fixed_params: set[str] = set()
        self.positive_param = positive_val
        self.any_param = any_val


@pytest.fixture
def owner_instance() -> _OwnerClass:
    """
    Pytest fixture to provide a clean instance of _OwnerClass for each test.
    """

    return _OwnerClass(positive_val=10.0, any_val=-5.0)


@pytest.fixture
def owner_instance_with_dtype() -> _OwnerClassWithDType:
    """
    Pytest fixture to provide a clean instance of _OwnerClassWithDType for each test.
    """

    return _OwnerClassWithDType(positive_val=10.0, any_val=-5.0, dtype=np.float32)


def test_parameter_initialization():
    """
    Tests that the Parameter descriptor is initialized correctly with
    the specified invariant and error message.
    """

    def invariant(x):
        return x > 0

    error_message = "Must be a positive number."

    param = Parameter(invariant=invariant, error_message=error_message)

    assert param.invariant == invariant
    assert param.error_message == error_message


def test_parameter_initialization_defaults():
    """
    Tests that the Parameter descriptor uses correct default values
    when no invariant or error message is provided.
    """

    param = Parameter()

    assert param.invariant(1)
    assert param.invariant(0)
    assert param.invariant(-100.5)
    assert param.error_message == "Parameter value is not valid."


def test_set_name_is_called_correctly():
    """
    Tests that the __set_name__ method correctly sets the public and
    private names of the parameter attribute. This is called automatically
    by Python when the owner class is created.
    """

    positive_descriptor = _OwnerClass.positive_param
    any_descriptor = _OwnerClass.any_param

    assert positive_descriptor.public_name == "positive_param"
    assert positive_descriptor.private_name == "_positive_param"
    assert any_descriptor.public_name == "any_param"
    assert any_descriptor.private_name == "_any_param"


def test_get_from_class_returns_descriptor():
    """
    Tests that accessing the parameter from the class (not an instance)
    returns the Parameter descriptor instance itself.
    """

    result = _OwnerClass.positive_param

    assert isinstance(result, Parameter)
    assert result.error_message == "Value must be positive."


def test_get_from_instance_returns_value(owner_instance: _OwnerClass):
    """
    Tests that accessing the parameter from an instance of the owner class
    returns the actual float value stored in the instance.
    """

    positive_value = owner_instance.positive_param
    expected_positive_value = 10.0
    any_value = owner_instance.any_param
    expected_any_value = -5.0

    assert isinstance(positive_value, float)
    assert positive_value == expected_positive_value
    assert isinstance(any_value, float)
    assert any_value == expected_any_value


def test_set_valid_value(owner_instance: _OwnerClass):
    """
    Tests that a valid value that satisfies the invariant can be
    successfully assigned to the parameter.
    """

    new_positive_value = 25.5
    owner_instance.positive_param = new_positive_value
    assert owner_instance.positive_param == new_positive_value


@pytest.mark.parametrize(
    "param_name, invalid_value, error_msg",
    [
        ("positive_param", 0, "Invalid value for 'positive_param': Value must be positive."),
        ("positive_param", -100.0, "Invalid value for 'positive_param': Value must be positive."),
    ],
)
def test_set_invalid_value_raises_value_error(
    owner_instance: _OwnerClass, param_name: str, invalid_value: float, error_msg: str
):
    """
    Tests that assigning a value that violates the parameter's invariant
    raises a ValueError with the correct message.
    """

    with pytest.raises(ValueError) as exc_info:
        setattr(owner_instance, param_name, invalid_value)

    assert str(exc_info.value) == error_msg


def test_set_fixed_parameter_raises_attribute_error(owner_instance: _OwnerClass):
    """
    Tests that attempting to change a parameter that has been 'fixed'
    raises an AttributeError.
    """

    param_name_to_fix = "positive_param"
    owner_instance._fixed_params.add(param_name_to_fix)

    with pytest.raises(AttributeError) as exc_info:
        owner_instance.positive_param = 999.0

    expected_msg = f"Cannot set '{param_name_to_fix}' for instance of '_OwnerClass' class. This parameter is fixed."
    assert str(exc_info.value) == expected_msg

    expected_positive_value = 10.0
    assert owner_instance.positive_param == expected_positive_value


def test_can_set_unfixed_parameter_after_fixing_another(owner_instance: _OwnerClass):
    """
    Tests that fixing one parameter does not prevent other, unfixed
    parameters from being changed.
    """

    owner_instance._fixed_params.add("positive_param")
    owner_instance.any_param = 123.45

    expected_positive_value = 10.0
    expected_any_value = 123.45

    assert owner_instance.any_param == expected_any_value
    assert owner_instance.positive_param == expected_positive_value


def test_get_from_instance_with_dtype_returns_correct_type(owner_instance_with_dtype: _OwnerClassWithDType):
    """
    Tests that __get__ returns a value of the correct DType when the owner
    instance has a `dtype` attribute.
    """

    positive_value = owner_instance_with_dtype.positive_param
    any_value = owner_instance_with_dtype.any_param

    assert isinstance(positive_value, np.float32)
    assert isinstance(any_value, np.float32)

    assert positive_value == np.float32(10.0)
    assert any_value == np.float32(-5.0)


def test_set_and_get_with_dtype_casting(owner_instance_with_dtype: _OwnerClassWithDType):
    """
    Tests the full set -> get cycle with dtype casting.
    """

    new_positive_value = 123.45
    owner_instance_with_dtype.positive_param = new_positive_value

    retrieved_value = owner_instance_with_dtype.positive_param

    assert isinstance(retrieved_value, np.float32)
    assert retrieved_value == np.float32(new_positive_value)
