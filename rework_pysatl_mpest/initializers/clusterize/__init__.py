"""
A subpackage providing cluster-based initialization logic.
"""

__author__ = "Viktor Khanukaev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .cluster_match_algorithms import match_clusters_for_models
from .clusterize_initializer import ClusterizeInitializer
from .strategies import EstimationStrategy, MatchingMethod, ScoringMethod

__all__ = [
    "ClusterizeInitializer",
    "EstimationStrategy",
    "MatchingMethod",
    "ScoringMethod",
    "match_clusters_for_models",
]
