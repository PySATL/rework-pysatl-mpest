"""
A module providing enumeration types for initialization strategies.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from enum import Enum, auto


class EstimationStrategy(Enum):
    """Enumeration of parameter estimation strategies for distribution components.

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
    """

    QFUNCTION = auto()


class MatchingMethod(Enum):
    """Enumeration of strategies for matching clusters to distribution models.

    Attributes
    ----------
    GREEDY : MatchingMethod
        Iteratively assigns the best remaining cluster to the best remaining model.
        Fastest method but may yield suboptimal total likelihood.
    HUNGARIAN : MatchingMethod
        Uses the Hungarian algorithm (linear sum assignment) to find the optimal
        assignment that minimizes the total cost (negative score) between models
        and clusters based on the cost matrix.
    PERMUTATIONS : MatchingMethod
        Brute-force approach that evaluates the full mixture score for every
        possible permutation of cluster-to-model assignments. Guaranteed to find
        the global optimum for the mixture score but computationally expensive
        (factorial complexity).
    """

    GREEDY = auto()
    HUNGARIAN = auto()
    PERMUTATIONS = auto()


class ScoringMethod(Enum):
    """Enumeration of scoring criteria used to evaluate model-cluster fits.

    Attributes
    ----------
    AIC : ScoringMethod
        Akaike Information Criterion. Penalizes model complexity (number of parameters)
        against the goodness of fit. Useful when comparing distributions with different
        degrees of freedom.
    LIKELIHOOD : ScoringMethod
        Uses the raw Log-Likelihood. Higher values indicate a better fit of the
        distribution to the cluster data.
    """

    AIC = auto()
    LIKELIHOOD = auto()
