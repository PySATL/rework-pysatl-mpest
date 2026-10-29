"""
Integration tests of the thresholds of the analytical strategies, exactly at the boundary.

The numerical tolerance is replaced by 0.25, which is exact in binary floating point, so that the data can be
chosen to hit the threshold exactly and the strict or non-strict comparisons can be told apart.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock
from pysatl_mpest.estimators.iterative._strategies import moments as moments_module
from pysatl_mpest.estimators.iterative._strategies import moments_strategy, q_function_strategy
from pysatl_mpest.estimators.iterative._strategies import q_function as q_function_module
from tests.helpers.families import family
from tests.helpers.pipeline import make_state

TOLERANCE = 0.25

STRATEGIES = pytest.mark.parametrize(
    ("strategy", "strategy_kind"),
    [(q_function_strategy, MaximizationStrategy.QFUNCTION), (moments_strategy, MaximizationStrategy.MOMENTS)],
    ids=["q_function", "moments"],
)


@pytest.fixture(autouse=True)
def coarse_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q_function_module, "NUMERICAL_TOLERANCE", TOLERANCE)
    monkeypatch.setattr(moments_module, "NUMERICAL_TOLERANCE", TOLERANCE)


def exponential(lambda_: float) -> ParametricFamilyDistribution:
    return family(FamilyName.EXPONENTIAL)(lambda_=lambda_)


def normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


def run(strategy, kind, component, X, H, params=None):
    block = OptimizationBlock(0, params, kind)
    return strategy(component, make_state(np.array(X), np.array(H)), block, optimizer=None)[1]


@STRATEGIES
class TestTotalResponsibility:
    @pytest.mark.parametrize(
        "make_component", [lambda: exponential(2.5), lambda: normal(0.0, 2.0)], ids=["exp", "normal"]
    )
    def test_at_the_tolerance_nothing_is_updated(self, strategy, strategy_kind, make_component) -> None:
        assert run(strategy, strategy_kind, make_component(), [1.0], [[TOLERANCE]]) == {}

    @pytest.mark.parametrize(
        "make_component", [lambda: exponential(2.5), lambda: normal(0.0, 2.0)], ids=["exp", "normal"]
    )
    def test_above_the_tolerance_the_parameters_are_updated(self, strategy, strategy_kind, make_component) -> None:
        assert run(strategy, strategy_kind, make_component(), [1.0], [[0.5]]) != {}


@STRATEGIES
class TestExponentialMean:
    def test_at_the_tolerance_the_rate_is_kept(self, strategy, strategy_kind) -> None:
        assert run(strategy, strategy_kind, exponential(2.5), [TOLERANCE], [[1.0]]) == {"lambda_": 2.5}

    def test_above_the_tolerance_the_rate_is_updated(self, strategy, strategy_kind) -> None:
        assert run(strategy, strategy_kind, exponential(2.5), [0.5], [[1.0]]) == {"lambda_": 2.0}


@STRATEGIES
class TestNormalVariance:
    def test_at_the_tolerance_the_standard_deviation_is_kept(self, strategy, strategy_kind) -> None:
        # The mean is 0.5 and the variance is exactly 0.25.
        result = run(strategy, strategy_kind, normal(0.0, 2.0), [0.0, 1.0], [[0.5], [0.5]])

        assert result == {"mu": 0.5, "sigma": 2.0}

    def test_above_the_tolerance_the_standard_deviation_is_updated(self, strategy, strategy_kind) -> None:
        # The mean is 1 and the variance is 1.
        result = run(strategy, strategy_kind, normal(0.0, 2.0), [0.0, 2.0], [[0.5], [0.5]])

        assert result == {"mu": 1.0, "sigma": 1.0}
