"""
Integration tests for the default Method of Moments strategy, which has no generic implementation.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock
from pysatl_mpest.estimators.iterative._strategies import moments_strategy
from tests.helpers.families import family
from tests.helpers.pipeline import make_state


@pytest.mark.parametrize(
    ("make_component", "message"),
    [
        pytest.param(
            lambda: family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0),
            "'meanVar' parametrization of the 'Normal' family",
            id="normal-meanVar",
        ),
        pytest.param(
            lambda: family(FamilyName.EXPONENTIAL)(parametrization_name="scale", beta=1.0),
            "'scale' parametrization of the 'Exponential' family",
            id="exponential-scale",
        ),
        pytest.param(
            lambda: family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0),
            "'standard' parametrization of the 'ContinuousUniform' family",
            id="uniform",
        ),
    ],
)
def test_raises_not_implemented_for_unregistered_components(
    make_component: type[ParametricFamilyDistribution], message: str
) -> None:
    state = make_state(X=np.array([0.5]), H=np.ones((1, 1)))
    block = OptimizationBlock(0, None, MaximizationStrategy.MOMENTS)

    with pytest.raises(NotImplementedError, match=f"Moments strategy is not implemented for the {message}"):
        moments_strategy(make_component(), state, block, optimizer=None)  # type: ignore[arg-type]
