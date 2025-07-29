from abc import ABC, abstractmethod


class BaseBreakpointer(ABC):

    @abstractmethod
    def check(context) -> bool:
        raise NotImplementedError
