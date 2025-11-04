"""Provides a descriptor for parameters of custom distributions.

This module contains a `Parameter` descriptor class, which is used to define
and validate parameters in classes inheriting from `ContinuousDistribution`.
It allows you to set invariants for parameter values and handle assignment errors,
as well as to fix parameters from changes."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import TYPE_CHECKING, Callable, Optional, Union, overload

import numpy as np

if TYPE_CHECKING:
    from ..distributions.continuous_dist import ContinuousDistribution
from ..typings import DType


class Parameter:
    """A descriptor for validating and managing distribution parameters.

    This class implements the descriptor protocol for managing access to
    attributes representing the parameters of a statistical distribution.
    It allows you to set the conditions (invariants) that the
    parameter value must satisfy.

    Parameters
    ----------
    invariant : Callable[[float], bool], optional
        A predicate function for validating parameter values.
        Defaults to `lambda x: True`.
    error_message : str, optional
        An error message in case of failed validation.
        Defaults to "Parameter value is not valid.".

    Attributes
    ----------
    invariant : Callable[[float], bool]
        A function that checks if the parameter value is valid.
        Returns True if the value is correct, and False otherwise.
    error_message : str
        An error message that is raised if the invariant is not satisfied.
    public_name : str
        The name of the attribute as defined in the owner class.
    private_name : str
        The name of the attribute used to store the value within an instance
        of the owner class.

    Examples
    --------

    .. code-block:: python

        class NormalDistribution(ContinuousDistribution):
            # Location parameter can be any number
            loc = Parameter()
            # Scale parameter must be a positive number
            scale = Parameter(invariant=lambda s: s > 0, error_message="Standard deviation (scale) must be positive.")

    """

    def __init__(
        self,
        invariant: Callable[[float], bool] = lambda x: True,
        error_message: str = "Parameter value is not valid.",
    ):
        self.invariant = invariant
        self.error_message = error_message

    def __set_name__(self, owner: type["ContinuousDistribution[DType]"], name: str):
        """Sets the name for the public and private attributes.

        This method is automatically called when a descriptor instance is created
        in the owner class. It uses the attribute name to create the
        public and private names.

        Parameters
        ----------
        owner : type[ContinuousDistribution]
            The class that uses the descriptor.
        name : str
            The attribute name assigned to the descriptor instance.
        """

        self.public_name = name
        self.private_name = "_" + name

    @overload
    def __get__(self, instance: None, owner: type["ContinuousDistribution[DType]"]) -> "Parameter":
        """If access is via a class, return the descriptor object itself."""

    @overload
    def __get__(self, instance: "ContinuousDistribution[DType]", owner: type["ContinuousDistribution[DType]"]) -> DType:
        """If access is via an object, return the value."""

    def __get__(
        self, instance: Optional["ContinuousDistribution[DType]"], owner: type["ContinuousDistribution[DType]"]
    ) -> Union[DType, "Parameter"]:
        """Returns the parameter value or the descriptor itself.

        If access is through an instance of the class, it returns the
        parameter's value. If access is through the class itself, it returns
        the descriptor object.

        Parameters
        ----------
        instance : ContinuousDistribution, optional
            An instance of the owner class, or `None` if access
            is through the class.
        owner : type[ContinuousDistribution]
            The owner class.

        Returns
        -------
        DType or Parameter
            The value of the parameter or the descriptor itself.
        """

        if instance is None:
            return self

        return getattr(instance, self.private_name)

    def __set__(self, instance: "ContinuousDistribution[DType]", value: float):
        """Sets the parameter value after validation.

        Before setting a new value, it checks whether the parameter is
        "fixed." Then, it validates the value using the :attr:`invariant` function.

        Parameters
        ----------
        instance : "ContinuousDistribution[DType]"
            An instance of the owner class.
        value : float
            The new value for the parameter.

        Raises
        ------
        AttributeError
            If an attempt is made to change a "fixed" parameter.
        ValueError
            If the new value does not pass the :attr:`invariant` check.
        """

        if self.public_name in getattr(instance, "_fixed_params", set()):
            raise AttributeError(
                f"Cannot set '{self.public_name}' for instance of '{type(instance).__name__}' class. "
                "This parameter is fixed."
            )

        owner_dtype = getattr(instance, "dtype", np.float64)
        d_value = owner_dtype(value)

        if not self.invariant(d_value):
            raise ValueError(f"Invalid value for '{self.public_name}': {self.error_message}")

        setattr(instance, self.private_name, d_value)
