from abc import ABC, abstractmethod

from numpy import ndarray


class ContinuousDistribution(ABC):

    def __init__(self) -> None:
        pass

    def loglikelihood(self, X: ndarray) -> ndarray:
        pass

    def q_function(self, X: ndarray, W: ndarray) -> ndarray:
        pass

    def fix_parmeter(self, name: str) -> ndarray:
        pass

    def unfix_parameter(self, name: str) -> ndarray:
        pass

    @abstractmethod
    def ppf(self, X: ndarray) -> ndarray:
        pass

    @abstractmethod
    def pdf(self, X: ndarray) -> ndarray:
        pass

    @abstractmethod
    def lpdf(self, X: ndarray) -> ndarray:
        pass

    @abstractmethod
    def log_gradients(self, X: ndarray) -> ndarray:
        pass

    @abstractmethod
    def generate(self, size: int) -> ndarray:
        pass
