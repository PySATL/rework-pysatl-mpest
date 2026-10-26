"""
Internal helpers for working with mixture components, which are `pysatl-core` distributions.

All access to `pysatl-core` distribution internals is kept in this module so that a change of the
`pysatl-core` API affects a single place.

A component created from ``family.view(...)`` stores only its free parameters; the fixed ones live in the
view. `pysatl-core` has no public way to get from such a component to its view
(``component.family`` returns the base family from the registry), so ``_family`` reads it from the
parametrization class.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy.typing import ArrayLike
from pysatl_core.families import ParametricFamily, ParametricFamilyDistribution
from pysatl_core.families.parametric_family import PartialParametricFamily
from pysatl_core.types import CharacteristicName

from ..typings import FloatArray

type ComponentKey = tuple[str, str, tuple[tuple[str, float], ...]]


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

    return _evaluate(component, CharacteristicName.LPDF, X)


def pdf(component: ParametricFamilyDistribution, X: ArrayLike) -> np.float64 | FloatArray:
    """Probability density function of a component.

    Parameters
    ----------
    component : ParametricFamilyDistribution
        The distribution whose density is evaluated.
    X : ArrayLike
        The points at which to evaluate the density.

    Returns
    -------
    np.float64 | FloatArray
        The density values with the same shape as :attr:`X`.
    """

    return _evaluate(component, CharacteristicName.PDF, X)


def free_parameters(component: ParametricFamilyDistribution) -> dict[str, float]:
    """Parameters of a component that can be changed.

    Parameters
    ----------
    component : ParametricFamilyDistribution
        The component to inspect.

    Returns
    -------
    dict[str, float]
        Names and values of the free parameters, in the parametrization of the component.
        For a component created without ``view`` these are all of its parameters.
    """

    return dict(component.parameters)


def fixed_parameters(component: ParametricFamilyDistribution) -> dict[str, float]:
    """Parameters of a component that were fixed with ``family.view(...)``.

    Parameters
    ----------
    component : ParametricFamilyDistribution
        The component to inspect.

    Returns
    -------
    dict[str, float]
        Names and values of the fixed parameters, in the parametrization of the component.
        Empty for a component created without ``view``.
    """

    family = _family(component)
    if isinstance(family, PartialParametricFamily):
        return dict(family.fixed_parameters)
    return {}


def all_parameters(component: ParametricFamilyDistribution) -> dict[str, float]:
    """Fixed and free parameters of a component together.

    Parameters
    ----------
    component : ParametricFamilyDistribution
        The component to inspect.

    Returns
    -------
    dict[str, float]
        Names and values of all parameters of the component's parametrization.
    """

    return {**fixed_parameters(component), **free_parameters(component)}


def with_parameters(component: ParametricFamilyDistribution, **updates: float) -> ParametricFamilyDistribution:
    """Rebuild a component with new values of some of its free parameters.

    The result belongs to the same family and parametrization, and the parameters fixed with ``view``
    keep their values. The component passed in is not changed. Sampling and computation strategies are
    not carried over, because `pysatl-core` makes them unique for each distribution.

    Parameters
    ----------
    component : ParametricFamilyDistribution
        The component to take the other parameters from.
    **updates : float
        New values of free parameters, by name.

    Returns
    -------
    ParametricFamilyDistribution
        A new component with the updated parameters.

    Raises
    ------
    ValueError
        If a name in :attr:`updates` is not a free parameter of the component (it is unknown or fixed),
        or if the new values violate the constraints of the parametrization.
    """

    free = free_parameters(component)
    if unknown := set(updates) - set(free):
        raise ValueError(f"Cannot update {sorted(unknown)}: the free parameters of the component are {sorted(free)}.")

    return _family(component).distribution(
        parametrization_name=component.parametrization_name,
        sampling_strategy=None,
        computation_strategy=None,
        **{**free, **updates},
    )


def component_key(component: ParametricFamilyDistribution) -> ComponentKey:
    """Hashable key identifying a component by its family, parametrization and parameter values.

    Two components have the same key if they belong to the same family, use the same parametrization
    and have equal values of all parameters, whether those are free or fixed with ``view``.
    The key is not normalized across parametrizations: ``meanStd(sigma=2)`` and ``meanVar(var=4)``
    have different keys.

    Parameters
    ----------
    component : ParametricFamilyDistribution
        The component to describe.

    Returns
    -------
    ComponentKey
        A tuple of the family name, the parametrization name and the sorted ``(name, value)`` pairs of
        all parameters.
    """

    return (
        component.family_name,
        component.parametrization_name,
        tuple(sorted(all_parameters(component).items())),
    )


def _evaluate(
    component: ParametricFamilyDistribution, characteristic: CharacteristicName, X: ArrayLike
) -> np.float64 | FloatArray:
    result = np.asarray(
        component.calculate_characteristic(characteristic, np.asarray(X, dtype=np.float64)), dtype=np.float64
    )
    return result[()] if result.ndim == 0 else result


def _family(component: ParametricFamilyDistribution) -> ParametricFamily:
    """For a component created from ``view`` this is the view itself, otherwise the base family."""

    return type(component.parametrization).__family__
