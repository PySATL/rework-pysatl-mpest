"""Defines configuration structures for the maximization step in a pipeline.

This module provides data structures used to configure and control the
maximization of an iterative estimation process, such as the one
implemented by :class:`MaximizationStep`.
It includes an enumeration of possible optimization strategies and a dataclass
to define a specific optimization task for a component's parameters.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from dataclasses import dataclass
from enum import Enum, auto

from pysatl_core.families import ParametricFamilyDistribution

from ....core import _components


class MaximizationStrategy(Enum):
    """Enumerates the available strategies for the :class:`MaximizationStep`.

    This enumeration defines the objective functions that can be used to
    update the parameters of mixture components during the maximization step.

    Attributes
    ----------
    QFUNCTION
        Indicates that the optimization should maximize the Q-function,
        which is the expected value of the complete-data log-likelihood.
    OBSERVED_DATA_LIKELIHOOD
        Indicates that the optimization should maximize the observed-data likelihood
        of the mixture.
    MOMENTS
        Indicates that the optimization should use the Method of Moments,
        matching theoretical moments to empirical weighted moments.
    """

    QFUNCTION = auto()
    OBSERVED_DATA_LIKELIHOOD = auto()
    MOMENTS = auto()


@dataclass
class OptimizationBlock:
    """A configuration block for optimizing a specific component's parameters.

    This dataclass specifies a single optimization task within a :class:`~MaximizationStep`.
    It defines which component to update, which of its parameters to optimize,
    and which objective function (strategy) to use for the optimization.

    Parameters
    ----------
    component_id : int
        The zero-based index of the component within the
        :class:`~pysatl_mpest.core.MixtureModel` to be optimized.
    params_to_optimize : set[str] | None
        A set of parameter names (as strings) for the specified component
        that should be included in the optimization, or ``None`` to optimize
        all of its free parameters. Parameters fixed with ``family.view(...)``
        are never free, so they cannot be listed here.
    maximization_strategy : MaximizationStrategy
        The optimization strategy to apply, chosen from the
        :class:`MaximizationStrategy` enum.
    """

    component_id: int
    params_to_optimize: set[str] | None
    maximization_strategy: MaximizationStrategy

    def params_to_update(self, component: ParametricFamilyDistribution) -> list[str]:
        """Names of the parameters of a component that this block optimizes.

        Parameters
        ----------
        component : ParametricFamilyDistribution
            The component the block refers to.

        Returns
        -------
        list[str]
            The sorted names of the parameters to optimize: all free parameters of the
            component if :attr:`params_to_optimize` is ``None``, otherwise the requested ones.

        Raises
        ------
        ValueError
            If :attr:`params_to_optimize` contains a name that is not a free parameter of
            the component: either the parametrization of the component has no such parameter
            or it was fixed with ``family.view(...)``.
        """

        free = set(_components.free_parameters(component))
        if self.params_to_optimize is None:
            return sorted(free)

        not_free = self.params_to_optimize - free
        if not_free:
            fixed = not_free & set(_components.fixed_parameters(component))
            unknown = not_free - fixed
            problems = []
            if unknown:
                problems.append(
                    f"{sorted(unknown)} are not parameters of the '{component.parametrization_name}' "
                    f"parametrization of '{component.family_name}'"
                )
            if fixed:
                problems.append(f"{sorted(fixed)} are fixed and cannot be optimized")
            raise ValueError(
                f"Cannot optimize parameters of component {self.component_id}: {'; '.join(problems)}. "
                f"Its free parameters are {sorted(free)}."
            )

        return sorted(self.params_to_optimize)
