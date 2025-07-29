
from numpy import ndarray
from new_mpest.mpest_core.continuous_dist import ContinuousDistribution


class Uniform(ContinuousDistribution):

    def __init__(self, a, b) -> None:
        super().__init__()

    def ppf(self, X: ndarray) -> ndarray:
        pass

    def pdf(self, X: ndarray) -> ndarray:
        pass

    def lpdf(self, X: ndarray) -> ndarray:
        pass

    def log_gradients(self, X: ndarray) -> ndarray:
        pass

    def generate(self, size: int) -> ndarray:
        pass
