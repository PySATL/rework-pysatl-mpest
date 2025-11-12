"""A module that provides custom exceptions for the project."""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class NumericalStabilityError(Exception):
    """
    Exception raised for issues related to numerical stability, such as
    overflow or underflow, during calculations.
    """

    def __init__(self, message: str):
        super().__init__(message)
