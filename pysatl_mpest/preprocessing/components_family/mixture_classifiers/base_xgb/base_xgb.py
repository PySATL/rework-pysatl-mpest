"""Module which contains XGB-base mixture classifier model"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path

from pysatl_mpest.distributions import Beta, Cauchy, Exponential, Normal, Uniform, Weibull
from pysatl_mpest.preprocessing.components_family.classifier_criterions import (
    MixtureClassifierCriterions,
)
from pysatl_mpest.preprocessing.components_family.classifier_models.classifier_models import (
    XGBClassifier,
)
from pysatl_mpest.preprocessing.components_family.mixture_classifiers.mixture_classifier import (
    MixtureClassifierModel,
)

XGBBaseModel = MixtureClassifierModel(
    XGBClassifier(),
    "https://drive.google.com/uc?id=1dNWfD7rRcCLawt9rJHDfCaPV7piE6jFB",
    str(Path(__file__).parent / "xgb_model.ubj"),
    str(Path(__file__).parent / "labels.csv"),
    MixtureClassifierCriterions(),
    {
        "G": Normal(0.0, 1.0),
        "W": Weibull(1.0, 0.0, 1.0),
        "U": Uniform(0.0, 1.0),
        "C": Cauchy(0.0, 1.0),
        "E": Exponential(0.0, 1.0),
        "B": Beta(0.0, 0.0, 0.0, 1.0),
    },
)
