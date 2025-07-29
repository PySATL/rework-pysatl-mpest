from abc import ABC, abstractmethod

from numpy import ndarray

from new_mpest.mpest_core.continuous_dist import ContinuousDistribution
from new_mpest.mpest_core.mixture_model import MixtureModel


class BaseInitializer(ABC):

    @abstractmethod
    def perform(self, X: ndarray, distributions: list[ContinuousDistribution]) -> MixtureModel:
        raise NotImplementedError
