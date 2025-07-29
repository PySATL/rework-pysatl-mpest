from numpy import ndarray
from new_mpest.algorithms.base_estimator import BaseEstimator
from new_mpest.mpest_core.mixture_model import MixtureModel


class IterativeEstimator(BaseEstimator):

    def __init__(self) -> None:
        pass

    def fit(self, X: ndarray, mixture: MixtureModel) -> MixtureModel:
        pass
