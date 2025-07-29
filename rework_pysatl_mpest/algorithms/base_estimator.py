from abc import ABC, abstractmethod

from numpy import ndarray

from new_mpest.mpest_core.mixture_model import MixtureModel


class BaseEstimator(ABC):

    @abstractmethod
    def fit(self, X: ndarray, mixture: MixtureModel) -> MixtureModel:
        raise NotImplementedError
