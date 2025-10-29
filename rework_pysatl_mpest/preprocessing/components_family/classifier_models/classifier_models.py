"""Module which contains all supported classifier models"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import xgboost as xgb
from rework_pysatl_mpest.preprocessing.components_family.classifier_models.classifier_interface import (
    ClassifierInterface,
)


class XGBClassifier(ClassifierInterface):
    """Implementation of XGBoosting-based classifier"""

    def __init__(self) -> None:
        super().__init__()
        self.model = xgb.Booster()

    def _load_model(self, model_path: str) -> None:
        self.model.load_model(model_path)

    def predict(self, criterions: dict[str, float]) -> np.ndarray:
        feature_names = list(criterions.keys())
        values = [criterions[name] for name in feature_names]
        features = xgb.DMatrix([values], feature_names=feature_names)

        return self.model.predict(features)
