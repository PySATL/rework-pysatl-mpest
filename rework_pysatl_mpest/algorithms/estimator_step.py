from abc import ABC, abstractmethod


class EstimatorStep(ABC):

    @abstractmethod
    def run(self, context):
        raise NotImplementedError
