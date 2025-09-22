from enum import Enum, auto


class EstimationStrategy(Enum):
    QFUNCTION = auto()


class ClusterMatchStrategy(Enum):
    LIKELIHOOD = auto()
    AKAIKE = auto()
