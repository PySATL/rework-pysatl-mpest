from abc import ABC, abstractmethod


class BasePruner(ABC):

    @abstractmethod
    def prune(context):
        raise NotImplementedError
