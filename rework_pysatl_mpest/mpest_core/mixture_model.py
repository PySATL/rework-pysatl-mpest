from numpy import ndarray

from new_mpest.mpest_core.continuous_dist import ContinuousDistribution


class MixtureModel:

    def __init__(self, components: list[ContinuousDistribution], weights: ndarray) -> None:
        pass

    def pdf(self, X: ndarray) -> ndarray:
        pass

    def lpdf(self, X: ndarray) -> ndarray:
        pass

    def loglikelihood(self, X: ndarray) -> ndarray:
        pass

    def q_function(self, X: ndarray, W: ndarray) -> ndarray:
        pass

    def generate(self, size: int) -> ndarray:
        pass
