"""
Validation of per-component parameter groups and construction of their optimization block chains.

``parameter_groups`` is the facade-level way to request conditional maximization: for a component it
is an ordered list of parameter groups, and each group becomes a separate maximization block. This
module is shared by the ECM and ECME facades so that the validation rules do not diverge.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping, Sequence

from pysatl_core.families import ParametricFamilyDistribution

from ..core import MixtureModel, _components
from .iterative import MaximizationStrategy, OptimizationBlock


def build_optimization_blocks(
    mixture: MixtureModel,
    parameter_groups: Mapping[int, Sequence[set[str]]] | None,
    strategies: Sequence[MaximizationStrategy],
) -> list[OptimizationBlock]:
    """Validate parameter groups and build ordered component blocks.

    Parameters
    ----------
    mixture : MixtureModel
        Mixture whose components are optimized.
    parameter_groups : Mapping[int, Sequence[set[str]]] | None
        Ordered parameter groups by component, or None for joint steps.
    strategies : Sequence[MaximizationStrategy]
        Strategy for each component, in component order.

    Returns
    -------
    list[OptimizationBlock]
        Blocks in component and group order, with copied group sets.

    Raises
    ------
    ValueError
        If the number of strategies differs from the number of components,
        or if parameter groups are invalid.
    """

    if len(strategies) != mixture.n_components:
        raise ValueError("The number of strategies must match the number of components.")

    validate_parameter_groups(mixture, parameter_groups)

    return [
        block
        for component_id, strategy in enumerate(strategies)
        for block in component_block_chain(
            component_id, None if parameter_groups is None else parameter_groups.get(component_id), strategy
        )
    ]


def validate_parameter_groups(
    mixture: MixtureModel,
    parameter_groups: Mapping[int, Sequence[set[str]]] | None,
) -> None:
    """Validate ``parameter_groups`` keys and per-component group lists.

    Keys must be non-negative component indices inside the mixture. For every
    listed component the group list must be non-empty, its groups must be
    non-empty and pairwise disjoint, and their union must equal the component's
    free parameters exactly — the condition for the conditional steps to form a
    partition of the component's maximization problem.

    Parameters
    ----------
    mixture : MixtureModel
        The mixture the indices refer to.
    parameter_groups : Mapping[int, Sequence[set[str]]] | None
        Ordered parameter groups per component index, or None.

    Raises
    ------
    ValueError
        If a key is negative or out of bounds, a component's group list is
        empty, a group is empty, groups overlap, or the union of a component's
        groups differs from its free parameters (unknown, fixed, or uncovered
        parameter names).
    """

    if parameter_groups is None:
        return

    n_components = mixture.n_components
    for component_id, groups in parameter_groups.items():
        if component_id < 0:
            raise ValueError(
                f"parameter_groups key {component_id} is negative; component indices must be non-negative."
            )
        if component_id >= n_components:
            raise ValueError(
                f"parameter_groups key {component_id} is out of bounds for a mixture with {n_components} components."
            )
        _validate_component_groups(mixture.components[component_id], component_id, groups)


def _validate_component_groups(
    component: ParametricFamilyDistribution, component_id: int, groups: Sequence[set[str]]
) -> None:
    if len(groups) == 0:
        raise ValueError(f"parameter_groups[{component_id}]: the component's group list must not be empty.")

    for group in groups:
        if len(group) == 0:
            raise ValueError(
                f"parameter_groups[{component_id}] contains an empty group; every group must be non-empty."
            )

    union: set[str] = set()
    for group in groups:
        shared = union & group
        if shared:
            raise ValueError(f"parameter_groups[{component_id}] groups overlap on parameters {sorted(shared)}.")
        union |= group

    free = set(_components.free_parameters(component))
    if union == free:
        return

    extra = union - free
    fixed = extra & set(_components.fixed_parameters(component))
    unknown = extra - fixed
    uncovered = free - union

    problems = []
    if unknown:
        problems.append(
            f"{sorted(unknown)} are not parameters of the '{component.parametrization_name}' "
            f"parametrization of '{component.family_name}'"
        )
    if fixed:
        problems.append(f"{sorted(fixed)} are fixed and cannot be optimized")
    if uncovered:
        problems.append(f"{sorted(uncovered)} are not covered by any group")

    raise ValueError(
        f"Invalid parameter_groups for component {component_id}: {'; '.join(problems)}. "
        f"Its free parameters are {sorted(free)}."
    )


def component_block_chain(
    component_id: int,
    groups: Sequence[set[str]] | None,
    strategy: MaximizationStrategy,
) -> list[OptimizationBlock]:
    """Build the ordered chain of optimization blocks for one component.

    Parameters
    ----------
    component_id : int
        Zero-based index of the component.
    groups : Sequence[set[str]] | None
        Ordered parameter groups of the component, or None for a single joint
        step over all its free parameters.
    strategy : MaximizationStrategy
        Strategy assigned to every block of the chain.

    Returns
    -------
    list[OptimizationBlock]
        Blocks in conditional-step order; the group sets are copied.
    """

    if groups is None:
        return [OptimizationBlock(component_id, None, strategy)]
    return [OptimizationBlock(component_id, set(group), strategy) for group in groups]
