from abc import ABC, abstractmethod


class Initializer(ABC):
    @abstractmethod
    def perform(self, x, dists, cluster_match_info, estimate_info):
        raise NotImplementedError
