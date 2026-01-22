"""Module which contains interface of the classifier model and supported classifier models"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_mpest.preprocessing.components_family.classifier_models.classifier_interface import (
    IClassifier,
)
from pysatl_mpest.preprocessing.components_family.classifier_models.classifier_models import (
    XGBClassifier,
)

__all__ = ["IClassifier", "XGBClassifier"]
