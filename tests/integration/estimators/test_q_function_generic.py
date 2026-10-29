"""
Integration tests for the generic (numerical) Q-function strategy on real pysatl-core components.

The generic strategy is used for the families and parametrizations that have no analytical solution
registered, here the ``meanVar`` parametrization of the normal family and the ``scale`` parametrization
of the exponential one.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators.iterative import PipelineState
from pysatl_mpest.estimators.iterative._strategies import q_function_strategy
from pysatl_mpest.optimizers import Optimizer, ScipyNelderMead, ScipyPowell
from pytest_mock import MockerFixture
from scipy import stats
from tests.helpers.families import family
from tests.helpers.pipeline import make_state, q_block


@pytest.fixture
def component() -> ParametricFamilyDistribution:
    """A normal component in the ``meanVar`` parametrization, which has no analytical strategy."""
    return family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=1.0, var=2.0)


@pytest.fixture
def state() -> PipelineState:
    return make_state(X=np.array([0.5, 1.0, 3.0]), H=np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]], dtype=np.float64))


@pytest.fixture
def optimizer(mocker: MockerFixture) -> Optimizer:
    """An optimizer that returns the values 3.0 and 4.0 (or just 3.0 for a single parameter)."""
    mock = mocker.create_autospec(Optimizer, instance=True)
    mock.minimize.side_effect = lambda target, start: [3.0, 4.0][: len(start)]
    return mock


def expected_q(state: PipelineState, mu: float, var: float) -> float:
    """The negative Q-function of the first component, computed with scipy."""
    assert state.H is not None
    return -float(np.dot(state.H[:, 0], stats.norm(loc=mu, scale=np.sqrt(var)).logpdf(state.X)))


def test_does_not_modify_the_original_component(
    component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
) -> None:
    q_function_strategy(component, state, q_block(), optimizer)

    assert cc.all_parameters(component) == {"mu": 1.0, "var": 2.0}


def test_calls_the_optimizer_once_from_the_current_values(
    component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
) -> None:
    q_function_strategy(component, state, q_block(), optimizer)

    optimizer.minimize.assert_called_once()  # type: ignore[attr-defined]
    _, start = optimizer.minimize.call_args.args  # type: ignore[attr-defined]
    assert start == [1.0, 2.0]


def test_returns_the_optimizer_result_by_parameter_name(
    component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
) -> None:
    component_id, new_params = q_function_strategy(component, state, q_block(), optimizer)

    assert component_id == 0
    assert new_params == {"mu": 3.0, "var": 4.0}
    assert all(type(value) is float for value in new_params.values())


def test_raises_if_h_is_not_computed(component: ParametricFamilyDistribution, optimizer: Optimizer) -> None:
    state = make_state(X=np.array([1.0]), H=None)

    with pytest.raises(ValueError, match=r"^Responsibility matrix H is not computed\.$"):
        q_function_strategy(component, state, q_block(), optimizer)


def test_target_is_the_negative_q_function(
    component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
) -> None:
    q_function_strategy(component, state, q_block(), optimizer)
    target, _ = optimizer.minimize.call_args.args  # type: ignore[attr-defined]

    assert target([0.5, 3.0]) == pytest.approx(expected_q(state, mu=0.5, var=3.0), rel=1e-12)


def test_target_is_infinite_where_a_constraint_is_violated(
    component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
) -> None:
    q_function_strategy(component, state, q_block(), optimizer)
    target, _ = optimizer.minimize.call_args.args  # type: ignore[attr-defined]

    assert target([0.5, -1.0]) == np.inf
    assert target([0.5, 0.0]) == np.inf


def test_only_the_requested_parameters_are_optimized(
    component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
) -> None:
    _, new_params = q_function_strategy(component, state, q_block(params={"var"}), optimizer)
    target, start = optimizer.minimize.call_args.args  # type: ignore[attr-defined]

    assert new_params == {"var": 3.0}
    assert start == [2.0]
    assert target([3.0]) == pytest.approx(expected_q(state, mu=1.0, var=3.0), rel=1e-12)  # mu keeps its value


def test_fixed_parameters_are_kept(state: PipelineState, optimizer: Optimizer) -> None:
    fixed_mu = family(FamilyName.NORMAL).view(parametrization_name="meanVar", mu=1.0)(var=2.0)

    _, new_params = q_function_strategy(fixed_mu, state, q_block(), optimizer)
    target, start = optimizer.minimize.call_args.args  # type: ignore[attr-defined]

    assert new_params == {"var": 3.0}
    assert start == [2.0]
    assert target([3.0]) == pytest.approx(expected_q(state, mu=1.0, var=3.0), rel=1e-12)


def test_nothing_to_optimize_skips_the_optimizer(
    component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
) -> None:
    assert q_function_strategy(component, state, q_block(params=set()), optimizer) == (0, {})

    optimizer.minimize.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.parametrize("params", [{"sigma"}, {"mu", "sigm"}])
def test_unknown_parameters_are_rejected(
    component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer, params: set[str]
) -> None:
    with pytest.raises(ValueError, match="are not parameters of the 'meanVar' parametrization"):
        q_function_strategy(component, state, q_block(params=params), optimizer)

    optimizer.minimize.assert_not_called()  # type: ignore[attr-defined]


def test_fixed_parameters_cannot_be_requested(state: PipelineState, optimizer: Optimizer) -> None:
    fixed_mu = family(FamilyName.NORMAL).view(parametrization_name="meanVar", mu=1.0)(var=2.0)

    with pytest.raises(ValueError, match=r"\['mu'\] are fixed and cannot be optimized"):
        q_function_strategy(fixed_mu, state, q_block(params={"mu", "var"}), optimizer)


def test_points_with_zero_responsibility_do_not_matter(optimizer: Optimizer) -> None:
    uniform = family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0)
    state = make_state(X=np.array([0.5, 5.0]), H=np.array([[1.0, 0.0], [0.0, 1.0]]))

    q_function_strategy(uniform, state, q_block(params={"upper_bound"}), optimizer)
    target, _ = optimizer.minimize.call_args.args  # type: ignore[attr-defined]

    assert target([2.0]) == pytest.approx(np.log(2.0))
    assert target([0.4]) == np.inf  # the point 0.5 leaves the support


def sample_state(true_mu: float, true_var: float, size: int = 2000) -> PipelineState:
    rng = np.random.default_rng(7)
    X = rng.normal(true_mu, np.sqrt(true_var), size)
    return make_state(X=X, H=np.column_stack([np.ones_like(X), np.zeros_like(X)]))


@pytest.mark.parametrize("optimizer_class", [ScipyNelderMead, ScipyPowell])
def test_recovers_the_parameters_of_a_sample(optimizer_class: type[Optimizer]) -> None:
    state = sample_state(true_mu=2.0, true_var=2.25)
    start = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0)

    _, new_params = q_function_strategy(start, state, q_block(), optimizer_class())

    assert new_params["mu"] == pytest.approx(np.mean(state.X), abs=1e-3)
    assert new_params["var"] == pytest.approx(np.var(state.X), rel=1e-3)


@pytest.mark.parametrize("optimizer_class", [ScipyNelderMead, ScipyPowell])
def test_survives_steps_outside_the_constraints(optimizer_class: type[Optimizer]) -> None:
    # Without the infinite target such a step would raise ValueError from pysatl-core.
    state = sample_state(true_mu=0.0, true_var=1e-4)
    start = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0)

    _, new_params = q_function_strategy(start, state, q_block(params={"var"}), optimizer_class())

    assert new_params["var"] == pytest.approx(np.mean(state.X**2), rel=1e-3)


def weighted_state(size: int = 3000) -> PipelineState:
    rng = np.random.default_rng(0)
    X = rng.normal(2.0, 1.5, size)
    H_j = rng.random(size)
    H_j[::7] = 0.0
    return make_state(X=np.abs(X) + 0.1, H=np.column_stack([H_j, 1.0 - H_j]))


def test_agrees_with_the_analytical_solution_for_the_normal_family() -> None:
    state = weighted_state()
    normal = family(FamilyName.NORMAL)

    _, analytical = q_function_strategy(normal(mu=0.0, sigma=1.0), state, q_block(), ScipyNelderMead())  # meanStd
    _, numerical = q_function_strategy(
        normal(parametrization_name="meanVar", mu=0.0, var=1.0), state, q_block(), ScipyNelderMead()
    )

    assert numerical["mu"] == pytest.approx(analytical["mu"], rel=1e-3)
    assert numerical["var"] == pytest.approx(analytical["sigma"] ** 2, rel=1e-3)


def test_agrees_with_the_analytical_solution_for_the_exponential_family() -> None:
    state = weighted_state()
    exponential = family(FamilyName.EXPONENTIAL)

    _, analytical = q_function_strategy(exponential(lambda_=1.0), state, q_block(), ScipyNelderMead())  # rate
    _, numerical = q_function_strategy(
        exponential(parametrization_name="scale", beta=1.0), state, q_block(), ScipyNelderMead()
    )

    assert 1.0 / numerical["beta"] == pytest.approx(analytical["lambda_"], rel=1e-3)
    assert_allclose(numerical["beta"], 1.0 / analytical["lambda_"], rtol=1e-3)
