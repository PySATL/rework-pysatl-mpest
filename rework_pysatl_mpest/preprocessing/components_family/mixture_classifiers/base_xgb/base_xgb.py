"""Module which contains XGB-base mixture classifier model"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path

from rework_pysatl_mpest.distributions import Beta, Cauchy, Exponential, Normal, Uniform, Weibull
from rework_pysatl_mpest.preprocessing.components_family.classifier_criterions import (
    MixtureClassifierCriterions,
)
from rework_pysatl_mpest.preprocessing.components_family.classifier_models.classifier_models import (
    XGBClassifier,
)
from rework_pysatl_mpest.preprocessing.components_family.mixture_classifiers.mixture_classifier import (
    MixtureClassifierModel,
)

XGBBaseModel = MixtureClassifierModel(
    XGBClassifier(),
    "https://drive.google.com/uc?id=1dNWfD7rRcCLawt9rJHDfCaPV7piE6jFB",
    str(Path(__file__).parent / "xgb_model.ubj"),
    str(Path(__file__).parent / "labels.csv"),
    MixtureClassifierCriterions(),
    {"G": Normal, "W": Weibull, "U": Uniform, "C": Cauchy, "E": Exponential, "B": Beta},
)
