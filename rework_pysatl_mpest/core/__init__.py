"""The "Core" module provides a mixture of distributions class,
as well as a "Parameter" descriptor for implementing your own distributions."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from .mixture import MixtureModel
from .parameter import Parameter

__all__ = ["MixtureModel", "Parameter"]
