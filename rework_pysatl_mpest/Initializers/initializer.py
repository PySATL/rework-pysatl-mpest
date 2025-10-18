"""
A module providing an abstract base class for mixture model initializers.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from numpy._typing import ArrayLike

from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from rework_pysatl_mpest.Initializers.strategies import ClusterMatchStrategy, EstimationStrategy


class Initializer(ABC):
    """Abstract base class for mixture model initializers.

    This class defines the interface for all initialization strategies that
    estimate initial parameters for mixture models. Subclasses must implement
    the `perform` method to provide specific initialization logic.

    Methods
    -------
    perform(X, dists, cluster_match_strategy, estimation_strategies)
        Performs initialization of mixture model parameters.

    Notes
    -----
    **Purpose**

    Initializers are responsible for providing good starting points for
    mixture model parameters before the main optimization process. This can
    significantly improve convergence speed and solution quality.

    **Implementation Requirements**

    Subclasses must implement the `perform` method to:

    - Estimate initial parameters for each distribution component
    - Calculate initial mixture weights
    - Return a properly initialized MixtureModel instance

    **Common Initialization Strategies**

    - Cluster-based initialization (using clustering algorithms)
    """

    @abstractmethod
    def perform(
        self,
        X: ArrayLike,
        dists: list[ContinuousDistribution],
        cluster_match_info: ClusterMatchStrategy,
        estimation_info: list[EstimationStrategy],
    ):
        """Performs initialization of mixture model parameters.

        Parameters
        ----------
        X : ArrayLike
            Input data points used for parameter estimation. Should be a 1D array
            of sample values from the mixture distribution.
        dists : list[ContinuousDistribution]
            List of distribution models to initialize. Each distribution
            represents one component of the mixture model. The number of
            distributions determines the number of mixture components.
        cluster_match_info : ClusterMatchStrategy
            Strategy for matching clusters to distribution models. Determines
            how clusters identified in the data are assigned to specific
            distribution components.
        estimation_info : list[EstimationStrategy]
            List of estimation strategies for each distribution model. Each
            element specifies the parameter estimation method to use for the
            corresponding distribution in the `dists` list.

        Returns
        -------
        MixtureModel
            An initialized mixture model with estimated parameters and
            normalized component weights that sum to 1.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        Notes
        -----
        The method should handle the following tasks:

        1. Validate input parameters and data consistency
        2. Estimate initial parameters for each distribution component using
           the specified estimation strategies
        3. Calculate initial mixture weights (should sum to 1)
        4. Ensure all parameters are within valid ranges for each distribution
        5. Return a properly configured MixtureModel instance

        The implementation may use various strategies to estimate good
        starting parameters for the EM algorithm or other optimization methods.
        """

        raise NotImplementedError
