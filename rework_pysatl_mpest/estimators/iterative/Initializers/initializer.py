from abc import ABC, abstractmethod


class Initializer(ABC):

    @abstractmethod
    def perform(self, x, dists, info):
        raise NotImplementedError
