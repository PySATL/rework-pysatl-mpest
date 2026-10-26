"""
Internal helpers for working with mixture components, which are `pysatl-core` distributions.

All access to `pysatl-core` distribution internals is kept in this module so that a change of the
`pysatl-core` API affects a single place.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy.typing import ArrayLike
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import CharacteristicName

from ..typings import FloatArray


def lpdf(component: ParametricFamilyDistribution, X: ArrayLike) -> np.float64 | FloatArray:
    """Logarithm of the probability density function of a component.

    Parameters
    ----------
    component : ParametricFamilyDistribution
        The distribution whose log-density is evaluated.
    X : ArrayLike
        The points at which to evaluate the log-density.

    Returns
    -------
    np.float64 | FloatArray
        The log-density values with the same shape as :attr:`X`. Points outside
        the support of the component give ``-inf``.
    """

    result = component.calculate_characteristic(CharacteristicName.LPDF, np.asarray(X, dtype=np.float64))
    return np.asarray(result, dtype=np.float64)
