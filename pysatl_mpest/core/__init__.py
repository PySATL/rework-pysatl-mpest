"""The "Core" module provides a mixture of distributions class,
as well as a ParametricAdapter for adapting distributions from pysatl-core."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from .mixture import MixtureModel

__all__ = ["MixtureModel"]
