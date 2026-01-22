"""Module which contains mixture classifier template"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import os

import gdown
import numpy as np
import pandas as pd
from pysatl_mpest.distributions import ContinuousDistribution
from pysatl_mpest.preprocessing.components_family.classifier_criterions import (
    MixtureClassifierCriterions,
)
from pysatl_mpest.preprocessing.components_family.classifier_models import (
    IClassifier,
)
from sklearn.preprocessing import LabelEncoder


class MixtureClassifierModel:
    """
    MixtureClassifierCriterions

    Parameters
    ----------
    :model:         IClassifier                      — Classifier Model
    :model_path:    str                               — Path to model folder
    :label_path:    str                               — Path to label folder
    :criterions:    MixtureClassifierCriterions       — Mixture Classifier Criterions
    :distributions: dict[str, ContinuousDistribution] — Dictionary of used distributions
    """

    def __init__(
        self,
        model: IClassifier,
        model_link: str | None,
        model_path: str,
        labels_path: str,
        criterions: MixtureClassifierCriterions,
        distributions: dict[str, ContinuousDistribution],
    ) -> None:
        self.model = model
        self.model_link = model_link
        self.model_path = model_path

        self.le = LabelEncoder()
        self.labels_path = labels_path

        self.criterions = criterions
        self.distributions = distributions

    def _download_model(self) -> None:
        """Function for installing a model from Google Drive if it is not downloaded"""

        if not os.path.exists(self.model_path):
            if not self.model_link:
                raise FileNotFoundError("The model file was not found")

            gdown.download(self.model_link, self.model_path, quiet=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Function for obtaining an unlabeled model prediction"""

        if not self.model.is_fitted:
            self._download_model()
            self.model.load_model(self.model_path)

        criterions = self.criterions.get_criterions(X)
        return self.model.predict(criterions)[0]

    def transform(self, feature_id: int) -> list[ContinuousDistribution]:
        """Function for converting a model prediction into an appropriate format"""

        if not hasattr(self.le, "classes_"):
            labels = pd.read_csv(self.labels_path)["Labels"]
            self.le.fit(labels)

        label = self.le.inverse_transform([feature_id])[0]
        return [self.distributions[d] for d in label.split(" ")]
