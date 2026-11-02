"""Integration tests for MaximizationStep with real strategies and components."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators.iterative import MaximizationStep, MaximizationStrategy, OptimizationBlock, PipelineState
from pysatl_mpest.optimizers import ScipyNelderMead

from tests.helpers.families import family


def test_moments_reject_unsupported_normal_parametrization() -> None:
    component = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0)
    mixture = MixtureModel([component])
    state = PipelineState(np.array([1.0, 2.0]), np.ones((2, 1)), None, mixture, None)
    step = MaximizationStep([OptimizationBlock(0, None, MaximizationStrategy.MOMENTS)], ScipyNelderMead())

    with pytest.raises(NotImplementedError, match="Moments strategy is not implemented for the 'meanVar'"):
        step.run(state)

    assert state.error is None


def test_q_function_rejects_unknown_parameter() -> None:
    mixture = MixtureModel([family(FamilyName.NORMAL)(mu=0.0, sigma=1.0)])
    state = PipelineState(np.array([1.0, 2.0, 3.0]), np.ones((3, 1)), None, mixture, None)
    step = MaximizationStep(
        [OptimizationBlock(0, {"invalid_param"}, MaximizationStrategy.QFUNCTION)], ScipyNelderMead()
    )

    with pytest.raises(ValueError, match="invalid_param.*are not parameters"):
        step.run(state)

    assert state.error is None


def test_view_component_density_matches_plain_component() -> None:
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    mixture = MixtureModel([family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)])
    state = PipelineState(x, np.full((len(x), 1), 1.0), None, mixture, None)
    step = MaximizationStep([OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION)], ScipyNelderMead())

    step.run(state)

    expected = family(FamilyName.NORMAL)(mu=1.0, sigma=1.25)
    assert cc.fixed_parameters(mixture[0]) == {"mu": 1.0}
    assert cc.free_parameters(mixture[0]) == {"sigma": 1.25}
    assert_allclose(cc.lpdf(mixture[0], x), cc.lpdf(expected, x))


def test_run_refreshes_previously_cached_mixture_density() -> None:
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    normal = family(FamilyName.NORMAL)
    mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0)], weights=[0.5, 0.5])
    mixture.pdf(x)
    state = PipelineState(x, np.full((len(x), 2), 0.5), None, mixture, None)
    step = MaximizationStep([OptimizationBlock(0, {"mu", "sigma"}, MaximizationStrategy.QFUNCTION)], ScipyNelderMead())

    step.run(state)

    expected = MixtureModel(
        [normal(mu=0.375, sigma=np.sqrt(4.6875 / 4)), normal(mu=5.0, sigma=1.0)],
        weights=[0.5, 0.5],
    )
    assert_allclose(mixture.pdf(x), expected.pdf(x))
