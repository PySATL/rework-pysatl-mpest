"""
Initializers module for mixture model parameter initialization.

This module provides various strategies for initializing parameters of mixture models
before the main optimization process. Good initialization is crucial for achieving
fast convergence and high-quality solutions in mixture model estimation.

Submodules
----------

.. autosummary::

        clusterizeInitializer
        clusterMatchStrategy
        initializer
        q_function
        strategies

Classes
-------
.. autosummary::

        ClusterizeInitializer
        Initializer

Functions
---------
.. autosummary::

        match_clusters_for_models_akaike
        match_clusters_for_models_log_likelihood
        q_function_strategy

Enumerations
------------
.. autosummary::

        EstimationStrategy
        ClusterMatchStrategy

Notes
-----
**Purpose**

Initializers provide good starting points for EM algorithm and other optimization
methods, helping to avoid poor local optima and improving convergence.

**Q-function Strategy Specializations**

The :func:`q_function_strategy` function supports multiple implementations for different distributions:

- **Generic implementation** - Uses numerical optimization for any continuous distribution
- **Exponential distribution** - Uses analytical solution (:func:`q_function_strategy_exponential`)

**Usage Example**

.. code-block:: python

    >>> from rework_pysatl_mpest import Exponential
    >>> import numpy as np
    >>> from sklearn.cluster import KMeans
    >>> from rework_pysatl_mpest.Initializers import ClusterizeInitializer
    >>> from rework_pysatl_mpest.Initializers import ClusterMatchStrategy, EstimationStrategy

    >>> # Create initializer with KMeans clustering
    >>> initializer_cluster = ClusterizeInitializer(
    ...     is_accurate=True,
    ...     is_soft=False,
    ...     clusterizer=KMeans(n_clusters=3)
    ... )

    >>> # Create distribution models to initialize
    >>> distributions = [Exponential(loc=0.0, rate=0.1),
    >>>Exponential(loc=5.0, rate=0.05), Exponential(loc=10.0, rate=0.01)]

    >>> # Generate sample data
    >>> X = np.linspace(0.01, 25.0, 300)

    >>> # Perform initialization
    >>> mixture_model = initializer_cluster.perform(
    ...     X=X,
    ...     dists=distributions,
    ...     cluster_match_info=ClusterMatchStrategy.AKAIKE,
    ...     estimation_info=[EstimationStrategy.QFUNCTION] * len(distributions)
    ... )

    >>> # The mixture model is now initialized with estimated parameters
    >>> print(f"Number of components: {len(mixture_model.components)}")
    >>> print(f"Weights: {mixture_model.weights}")

**See Also**
------------
:class:`rework_pysatl_mpest.MixtureModel`
    The mixture model class that uses these initializers.
:class:`rework_pysatl_mpest.estimators.BaseEstimator`
    Estimators that utilize initialization strategies.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .cluster_match_strategy import (
    match_clusters_for_models_akaike,
    match_clusters_for_models_log_likelihood,
)
from .clusterize_initializer import ClusterizeInitializer
from .initializer import Initializer
from .q_function import q_function_strategy, q_function_strategy_exponential
from .strategies import ClusterMatchStrategy, EstimationStrategy

__all__ = [
    "ClusterMatchStrategy",
    "ClusterizeInitializer",
    "EstimationStrategy",
    "Initializer",
    "match_clusters_for_models_akaike",
    "match_clusters_for_models_log_likelihood",
    "q_function_strategy",
    "q_function_strategy_exponential",
]
