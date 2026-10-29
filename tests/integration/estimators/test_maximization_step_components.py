"""
Integration tests for ``MaximizationStep`` applying parameters to real pysatl-core components.

The strategies are replaced with scripted ones, so these tests check how the step turns the
parameters returned by a strategy into new components of the mixture.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators.iterative import (
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    PipelineState,
)
from pysatl_mpest.optimizers import Optimizer
from pytest_mock import MockerFixture
from tests.helpers.families import family

X = np.array([-1.0, 0.0, 0.5, 2.0])


def make_state(mixture: MixtureModel, H: np.ndarray | None = None) -> PipelineState:
    H = np.full((len(X), mixture.n_components), 1.0 / mixture.n_components) if H is None else H
    return PipelineState(X=X, H=H, prev_mixture=None, curr_mixture=mixture, error=None)


def make_step(mocker: MockerFixture, blocks: list[OptimizationBlock], *updates: dict[str, float]) -> MaximizationStep:
    """A step whose strategy returns ``updates`` for the blocks, in order."""

    returned = iter(updates)

    def scripted(
        component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
    ) -> tuple[int, dict[str, float]]:
        return block.component_id, next(returned)

    mocker.patch.object(MaximizationStep, "_strategies", {MaximizationStrategy.QFUNCTION: scripted})
    return MaximizationStep(blocks, mocker.MagicMock(spec=Optimizer))


def block(component_id: int, params: set[str] | None = None) -> OptimizationBlock:
    return OptimizationBlock(component_id, params, MaximizationStrategy.QFUNCTION)


@pytest.fixture
def mixture() -> MixtureModel:
    normal = family(FamilyName.NORMAL)
    return MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0)], weights=[0.5, 0.5])


def test_returned_parameters_replace_the_component(mocker: MockerFixture, mixture: MixtureModel) -> None:
    state = make_state(mixture)
    untouched, original = mixture[1], mixture[0]

    make_step(mocker, [block(0)], {"mu": 1.0, "sigma": 2.0}).run(state)

    assert cc.all_parameters(mixture[0]) == {"mu": 1.0, "sigma": 2.0}
    assert mixture[1] is untouched
    assert cc.all_parameters(original) == {"mu": 0.0, "sigma": 1.0}


def test_fixed_parameters_of_a_view_component_are_kept(mocker: MockerFixture) -> None:
    mixture = MixtureModel([family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)])
    state = make_state(mixture)

    make_step(mocker, [block(0, {"sigma"})], {"sigma": 3.0}).run(state)

    assert cc.fixed_parameters(mixture[0]) == {"mu": 1.0}
    assert cc.free_parameters(mixture[0]) == {"sigma": 3.0}
    assert_allclose(cc.lpdf(mixture[0], X), cc.lpdf(family(FamilyName.NORMAL)(mu=1.0, sigma=3.0), X))


def test_updates_of_two_blocks_of_one_component_are_combined(mocker: MockerFixture, mixture: MixtureModel) -> None:
    state = make_state(mixture)
    step = make_step(mocker, [block(0, {"mu"}), block(0, {"sigma"})], {"mu": 2.0}, {"sigma": 3.0})

    step.run(state)

    assert cc.all_parameters(mixture[0]) == {"mu": 2.0, "sigma": 3.0}


def test_weights_are_set_from_responsibilities(mocker: MockerFixture, mixture: MixtureModel) -> None:
    H = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]])

    make_step(mocker, []).run(make_state(mixture, H))

    assert_allclose(mixture.weights, [0.45, 0.55], atol=1e-12)


def test_the_cache_of_the_mixture_is_refreshed(mocker: MockerFixture, mixture: MixtureModel) -> None:
    mixture.pdf(X)
    state = make_state(mixture)

    make_step(mocker, [block(0)], {"mu": 1.0, "sigma": 2.0}).run(state)

    normal = family(FamilyName.NORMAL)
    expected = MixtureModel([normal(mu=1.0, sigma=2.0), normal(mu=5.0, sigma=1.0)], weights=mixture.weights)
    assert_allclose(mixture.pdf(X), expected.pdf(X))


def test_updating_a_fixed_parameter_is_rejected(mocker: MockerFixture) -> None:
    component = family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)
    mixture = MixtureModel([component])

    with pytest.raises(ValueError, match=r"Cannot update \['mu'\]"):
        make_step(mocker, [block(0)], {"mu": 5.0}).run(make_state(mixture))

    assert mixture[0] is component
