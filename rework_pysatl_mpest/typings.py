"""A module that provides custom types for the project."""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TypeVar

import numpy as np

DType = TypeVar("DType", bound=np.floating)
