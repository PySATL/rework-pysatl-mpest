"""
Integration tests for the strategies registered for the built-in pysatl-core families.

They check which implementation is selected for real components, including components created
through ``family.view(...)``.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

import pytest
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.estimators.iterative._strategies._dispatch import FamilyDispatch
from pysatl_mpest.estimators.iterative._strategies.moments import moments_strategy
from pysatl_mpest.estimators.iterative._strategies.q_function import q_function_strategy
from tests.helpers.families import family

ANALYTICAL = [
    pytest.param(lambda: family(FamilyName.NORMAL)(mu=0.0, sigma=1.0), ("Normal", "meanStd"), id="normal-meanStd"),
    pytest.param(
        lambda: family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0), ("Normal", "meanStd"), id="normal-meanStd-view"
    ),
    pytest.param(lambda: family(FamilyName.EXPONENTIAL)(lambda_=1.0), ("Exponential", "rate"), id="exponential-rate"),
]

GENERIC = [
    pytest.param(
        lambda: family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0), id="normal-meanVar"
    ),
    pytest.param(
        lambda: family(FamilyName.EXPONENTIAL)(parametrization_name="scale", beta=1.0), id="exponential-scale"
    ),
    pytest.param(lambda: family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0), id="uniform"),
]

STRATEGIES = [
    pytest.param(q_function_strategy, id="q_function"),
    pytest.param(moments_strategy, id="moments"),
]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_registered_keys(strategy: FamilyDispatch) -> None:
    assert sorted(strategy.registry) == [("Exponential", "rate"), ("Normal", "meanStd")]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(("make_component", "key"), ANALYTICAL)
def test_analytical_implementation_is_selected(
    strategy: FamilyDispatch, make_component: Callable[[], ParametricFamilyDistribution], key: tuple[str, str]
) -> None:
    selected = strategy.dispatch(make_component())

    assert selected is strategy.registry[key]
    assert selected is not strategy.__wrapped__


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("make_component", GENERIC)
def test_default_implementation_is_selected_otherwise(
    strategy: FamilyDispatch, make_component: Callable[[], ParametricFamilyDistribution]
) -> None:
    assert strategy.dispatch(make_component()) is strategy.__wrapped__
