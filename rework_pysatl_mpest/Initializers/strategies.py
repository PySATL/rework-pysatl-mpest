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


class ClusterMatchStrategy(Enum):
    """Enumeration of strategies for matching clusters to distribution models.

    This enumeration defines the available methods for assigning clusters
    (identified by clustering algorithms) to specific distribution models
    during mixture model initialization.

    Attributes
    ----------
    LIKELIHOOD : ClusterMatchStrategy
        Uses weighted log-likelihood criteria to match clusters to models.
        Each model is sequentially assigned to the cluster that maximizes
        its weighted log-likelihood score.

    AKAIKE : ClusterMatchStrategy
        Uses Akaike Information Criterion (AIC) to find the optimal assignment
        between clusters and models. Evaluates all possible permutations and
        selects the combination that minimizes the total AIC score.

    Notes
    -----
    **LIKELIHOOD Strategy**

    - Sequential greedy assignment
    - Computationally efficient
    - May find locally optimal but not globally optimal assignments
    - Uses normalized weighted log-likelihood as selection criteria

    **AKAIKE Strategy**

    - Evaluates all possible cluster-model permutations
    - Finds globally optimal assignment (with respect to AIC)
    - Computationally more expensive but provides better results
    - Balances model fit and complexity through AIC penalty

    **Comparison**

    - LIKELIHOOD: Faster, suitable for large numbers of components
    - AKAIKE: More accurate, recommended for smaller numbers of components
    - Choice depends on computational constraints and quality requirements

    **Future Extensions**

    Additional strategies that could be added:
    - BAYESIAN: Using Bayesian Information Criterion (BIC)
    """

    LIKELIHOOD = auto()
    AKAIKE = auto()
