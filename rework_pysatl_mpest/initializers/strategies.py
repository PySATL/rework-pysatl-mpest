"""
A module providing enumeration types for initialization strategies.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from enum import Enum, auto


class EstimationStrategy(Enum):
    """Enumeration of parameter estimation strategies for distribution components.

    This enumeration defines the available methods for estimating parameters
    of individual distribution components during mixture model initialization.

    Attributes
    ----------
    QFUNCTION : EstimationStrategy
        Uses the Q-function (expected complete data log-likelihood) for
        parameter estimation. This strategy maximizes the Q-function either
        analytically (for specific distributions) or numerically (for general
        distributions).

    Notes
    -----
    **QFUNCTION Strategy**

    The Q-function strategy:
    - For Exponential distribution: Uses analytical solution for efficiency
    - For other distributions: Falls back to numerical optimization
    - Provides maximum likelihood estimates in the EM framework
    - Handles weighted data points through the responsibility matrix

    **Future Extensions**

    Additional strategies that could be added:
    - Implementations for other types of distributions.
    """

    QFUNCTION = auto()


class MatchingMethod(Enum):
    GREEDY = auto()
    HUNGARIAN = auto()
    PERMUTATIONS = auto()


class ScoringMethod(Enum):
    AIC = auto()
    LIKELIHOOD = auto()
