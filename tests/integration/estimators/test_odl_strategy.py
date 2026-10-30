"""
Integration tests for the observed data likelihood strategy on mixtures of real pysatl-core components.

The reference values are computed with ``scipy.stats`` and ``scipy.special.logsumexp``.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from copy import copy

import numpy as np
import pytest
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock
from pysatl_mpest.estimators.iterative._strategies import observed_data_likelihood_strategy
from pysatl_mpest.estimators.iterative._strategies.observed_data_likelihood import LOG_FLOOR
from pysatl_mpest.optimizers import Optimizer, ScipyNelderMead
from pytest_mock import MockerFixture
from scipy import stats
from scipy.special import logsumexp
from tests.helpers.families import family
from tests.helpers.pipeline import make_state

X = np.array([-1.0, 0.0, 0.5, 2.0, 4.5, 6.0])


def odl_block(component_id: int = 1, params: set[str] | None = None) -> OptimizationBlock:
    return OptimizationBlock(component_id, params, MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD)


def normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


def negative_log_likelihood(weights: list[float], normals: list[tuple[float, float]], data: np.ndarray = X) -> float:
    """The negative log-likelihood of a mixture of normals, computed with scipy."""

    terms = [np.log(w) + stats.norm(mu, sigma).logpdf(data) for w, (mu, sigma) in zip(weights, normals)]
    return -float(np.sum(logsumexp(terms, axis=0)))


@pytest.fixture
def mixture() -> MixtureModel:
    return MixtureModel([normal(0.0, 1.0), normal(3.0, 2.0)], weights=[0.4, 0.6])


@pytest.fixture
def optimizer(mocker: MockerFixture) -> Optimizer:
    """An optimizer that returns the starting values, so that the target can be inspected."""
    mock = mocker.create_autospec(Optimizer, instance=True)
    mock.minimize.side_effect = lambda target, start: list(start)
    return mock


def captured_target(optimizer: Optimizer):
    return optimizer.minimize.call_args.args[0]  # type: ignore[attr-defined]


def test_does_not_modify_the_component_or_the_mixture(mixture: MixtureModel, optimizer: Optimizer) -> None:
    before = copy(mixture)
    component = mixture[1]

    observed_data_likelihood_strategy(component, make_state(X, None, mixture), odl_block(), optimizer)

    assert cc.all_parameters(component) == {"mu": 3.0, "sigma": 2.0}
    assert mixture == before


def test_calls_the_optimizer_once_from_the_current_values(mixture: MixtureModel, optimizer: Optimizer) -> None:
    observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)

    optimizer.minimize.assert_called_once()  # type: ignore[attr-defined]
    assert optimizer.minimize.call_args.args[1] == [3.0, 2.0]  # type: ignore[attr-defined]


def test_returns_the_component_id_and_floats(mixture: MixtureModel, optimizer: Optimizer) -> None:
    optimizer.minimize.side_effect = lambda target, start: [1.5, 2.5]  # type: ignore[attr-defined]

    component_id, new_params = observed_data_likelihood_strategy(
        mixture[1], make_state(X, None, mixture), odl_block(), optimizer
    )

    assert component_id == 1
    assert new_params == {"mu": 1.5, "sigma": 2.5}
    assert all(type(value) is float for value in new_params.values())


def test_responsibilities_are_not_needed(mixture: MixtureModel, optimizer: Optimizer) -> None:
    state = make_state(X, None, mixture)

    assert state.H is None
    assert observed_data_likelihood_strategy(mixture[1], state, odl_block(), optimizer)[0] == 1


def test_target_is_the_negative_log_likelihood_of_the_mixture(mixture: MixtureModel, optimizer: Optimizer) -> None:
    observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)
    target = captured_target(optimizer)

    assert target([2.0, 1.5]) == pytest.approx(negative_log_likelihood([0.4, 0.6], [(0.0, 1.0), (2.0, 1.5)]), rel=1e-12)


def test_target_for_the_first_component_uses_the_others_as_background(
    mixture: MixtureModel, optimizer: Optimizer
) -> None:
    observed_data_likelihood_strategy(mixture[0], make_state(X, None, mixture), odl_block(component_id=0), optimizer)

    assert captured_target(optimizer)([-1.0, 0.5]) == pytest.approx(
        negative_log_likelihood([0.4, 0.6], [(-1.0, 0.5), (3.0, 2.0)]), rel=1e-12
    )


def test_target_of_a_single_component_mixture(optimizer: Optimizer) -> None:
    mixture = MixtureModel([normal(0.0, 1.0)])

    observed_data_likelihood_strategy(mixture[0], make_state(X, None, mixture), odl_block(component_id=0), optimizer)

    assert captured_target(optimizer)([1.0, 2.0]) == pytest.approx(-float(np.sum(stats.norm(1.0, 2.0).logpdf(X))))


def test_target_uses_the_normalized_weights_of_the_mixture(mixture: MixtureModel, optimizer: Optimizer) -> None:
    mixture.log_weights = np.log([2.0, 6.0])

    observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)

    assert captured_target(optimizer)([2.0, 1.5]) == pytest.approx(
        negative_log_likelihood([0.25, 0.75], [(0.0, 1.0), (2.0, 1.5)]), rel=1e-12
    )


def test_target_is_finite_and_exact_in_the_far_tail(mixture: MixtureModel, optimizer: Optimizer) -> None:
    far = np.array([-100.0, 100.0])
    assert np.all(mixture.pdf(far) == 0.0)

    observed_data_likelihood_strategy(mixture[1], make_state(far, None, mixture), odl_block(), optimizer)
    value = captured_target(optimizer)([3.0, 2.0])

    assert np.isfinite(value)
    assert value == pytest.approx(negative_log_likelihood([0.4, 0.6], [(0.0, 1.0), (3.0, 2.0)], far), rel=1e-12)


def test_target_clips_finite_log_densities_below_the_floor(optimizer: Optimizer) -> None:
    component = normal(0.0, 1.0)
    mixture = MixtureModel([component])
    data = np.array([200.0])
    log_density = cc.lpdf(component, data)
    assert np.all(np.isfinite(log_density))
    assert np.all(log_density < LOG_FLOOR)

    observed_data_likelihood_strategy(component, make_state(data, None, mixture), odl_block(component_id=0), optimizer)

    assert captured_target(optimizer)([0.0, 1.0]) == -LOG_FLOOR


def test_target_is_infinite_when_the_log_density_is_nan(optimizer: Optimizer) -> None:
    component = normal(0.0, 1.0)
    mixture = MixtureModel([component])
    data = np.array([np.nan])
    assert np.all(np.isnan(cc.lpdf(component, data)))

    observed_data_likelihood_strategy(component, make_state(data, None, mixture), odl_block(component_id=0), optimizer)

    with np.errstate(invalid="ignore"):
        assert captured_target(optimizer)([0.0, 1.0]) == np.inf


def test_target_of_a_component_with_zero_weight_does_not_depend_on_it(optimizer: Optimizer) -> None:
    mixture = MixtureModel([normal(0.0, 1.0), normal(3.0, 2.0)], weights=[1.0, 0.0])

    observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)
    target = captured_target(optimizer)

    assert target([3.0, 2.0]) == pytest.approx(target([-5.0, 0.3]))


def test_target_is_infinite_where_a_constraint_is_violated(mixture: MixtureModel, optimizer: Optimizer) -> None:
    observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)

    assert captured_target(optimizer)([2.0, -1.0]) == np.inf


@pytest.mark.filterwarnings("error")
def test_target_penalizes_points_outside_every_support(optimizer: Optimizer) -> None:
    exponential = family(FamilyName.EXPONENTIAL)(lambda_=1.0)
    uniform = family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0)
    mixture = MixtureModel([exponential, uniform])
    state = make_state(np.array([-1.0, 0.5]), None, mixture)

    observed_data_likelihood_strategy(uniform, state, odl_block(params={"upper_bound"}), optimizer)
    target = captured_target(optimizer)

    observed_data_likelihood_strategy(
        uniform, make_state(np.array([0.5]), None, mixture), odl_block(params={"upper_bound"}), optimizer
    )

    assert target([2.0]) - captured_target(optimizer)([2.0]) == pytest.approx(-LOG_FLOOR)


def test_target_is_finite_with_unsupported_outliers(optimizer: Optimizer) -> None:
    mixture = MixtureModel([family(FamilyName.EXPONENTIAL)(lambda_=0.5), family(FamilyName.EXPONENTIAL)(lambda_=3.0)])
    data = np.array([-0.5, 0.2, 1.0, 2.0])

    observed_data_likelihood_strategy(mixture[1], make_state(data, None, mixture), odl_block(), optimizer)

    assert np.isfinite(captured_target(optimizer)([3.0]))


@pytest.mark.parametrize("rate", [1.0, 3.0])
def test_unsupported_point_adds_a_constant_penalty(optimizer: Optimizer, rate: float) -> None:
    mixture = MixtureModel([family(FamilyName.EXPONENTIAL)(lambda_=0.5), family(FamilyName.EXPONENTIAL)(lambda_=3.0)])
    observed_data_likelihood_strategy(
        mixture[1], make_state(np.array([-0.5, 0.2, 1.0, 2.0]), None, mixture), odl_block(), optimizer
    )
    target_with_outlier = captured_target(optimizer)
    observed_data_likelihood_strategy(
        mixture[1], make_state(np.array([0.2, 1.0, 2.0]), None, mixture), odl_block(), optimizer
    )

    assert target_with_outlier([rate]) - captured_target(optimizer)([rate]) == pytest.approx(-LOG_FLOOR)


def test_recovers_the_parameters_in_presence_of_an_unsupported_point() -> None:
    truth = MixtureModel([family(FamilyName.EXPONENTIAL)(lambda_=0.5), family(FamilyName.EXPONENTIAL)(lambda_=1.5)])
    sample = np.asarray(truth.generate(size=1000, random_state=42))
    start = family(FamilyName.EXPONENTIAL)(lambda_=3.0)
    mixture = MixtureModel([truth[0], start])
    _, clean = observed_data_likelihood_strategy(
        start, make_state(sample, None, mixture), odl_block(), ScipyNelderMead()
    )
    _, contaminated = observed_data_likelihood_strategy(
        start, make_state(np.concatenate((sample, [-0.5])), None, mixture), odl_block(), ScipyNelderMead()
    )

    assert contaminated["lambda_"] == pytest.approx(clean["lambda_"], rel=1e-4)
    assert contaminated["lambda_"] != pytest.approx(3.0)


def test_uniform_component_extends_its_support_over_points_outside_it() -> None:
    sample = np.random.default_rng(42).uniform(0.0, 10.0, size=1000)
    component = family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=4.0, upper_bound=6.0)
    mixture = MixtureModel([component])

    _, fitted = observed_data_likelihood_strategy(
        component, make_state(sample, None, mixture), odl_block(component_id=0), ScipyNelderMead()
    )

    assert fitted["lower_bound"] <= sample.min()
    assert fitted["upper_bound"] >= sample.max()


def test_background_may_contain_components_with_fixed_parameters(optimizer: Optimizer) -> None:
    fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)
    mixture = MixtureModel([fixed_mu, normal(3.0, 2.0)], weights=[0.4, 0.6])

    observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)

    assert captured_target(optimizer)([2.0, 1.5]) == pytest.approx(
        negative_log_likelihood([0.4, 0.6], [(0.0, 1.0), (2.0, 1.5)]), rel=1e-12
    )


def test_only_the_requested_parameters_are_optimized(mixture: MixtureModel, optimizer: Optimizer) -> None:
    _, new_params = observed_data_likelihood_strategy(
        mixture[1], make_state(X, None, mixture), odl_block(params={"sigma"}), optimizer
    )

    assert new_params == {"sigma": 2.0}
    assert optimizer.minimize.call_args.args[1] == [2.0]  # type: ignore[attr-defined]
    assert captured_target(optimizer)([1.5]) == pytest.approx(
        negative_log_likelihood([0.4, 0.6], [(0.0, 1.0), (3.0, 1.5)]), rel=1e-12
    )


def test_fixed_parameters_of_the_component_are_kept(optimizer: Optimizer) -> None:
    fixed_mu = family(FamilyName.NORMAL).view(mu=3.0)(sigma=2.0)
    mixture = MixtureModel([normal(0.0, 1.0), fixed_mu], weights=[0.4, 0.6])

    _, new_params = observed_data_likelihood_strategy(fixed_mu, make_state(X, None, mixture), odl_block(), optimizer)

    assert new_params == {"sigma": 2.0}
    assert captured_target(optimizer)([1.5]) == pytest.approx(
        negative_log_likelihood([0.4, 0.6], [(0.0, 1.0), (3.0, 1.5)]), rel=1e-12
    )


def test_nothing_to_optimize_skips_the_optimizer(mixture: MixtureModel, optimizer: Optimizer) -> None:
    result = observed_data_likelihood_strategy(
        mixture[1], make_state(X, None, mixture), odl_block(params=set()), optimizer
    )

    assert result == (1, {})
    optimizer.minimize.assert_not_called()  # type: ignore[attr-defined]


def test_unknown_parameters_are_rejected(mixture: MixtureModel, optimizer: Optimizer) -> None:
    with pytest.raises(ValueError, match=r"\['var'\] are not parameters of the 'meanStd' parametrization"):
        observed_data_likelihood_strategy(
            mixture[1], make_state(X, None, mixture), odl_block(params={"var"}), optimizer
        )

    optimizer.minimize.assert_not_called()  # type: ignore[attr-defined]


def test_fixed_parameters_cannot_be_requested(optimizer: Optimizer) -> None:
    fixed_mu = family(FamilyName.NORMAL).view(mu=3.0)(sigma=2.0)
    mixture = MixtureModel([normal(0.0, 1.0), fixed_mu])

    with pytest.raises(ValueError, match=r"\['mu'\] are fixed and cannot be optimized"):
        observed_data_likelihood_strategy(
            fixed_mu, make_state(X, None, mixture), odl_block(params={"mu", "sigma"}), optimizer
        )


@pytest.fixture
def sample_mixture() -> tuple[MixtureModel, np.ndarray]:
    mixture = MixtureModel([normal(0.0, 1.0), normal(5.0, 1.0)], weights=[0.5, 0.5])
    return mixture, np.asarray(mixture.generate(size=4000, random_state=1))


def test_recovers_the_parameters_of_a_component_from_a_sample(sample_mixture: tuple[MixtureModel, np.ndarray]) -> None:
    mixture, sample = sample_mixture
    start = normal(4.0, 2.0)

    _, new_params = observed_data_likelihood_strategy(
        start, make_state(sample, None, mixture), odl_block(), ScipyNelderMead()
    )

    assert new_params["mu"] == pytest.approx(5.0, abs=0.15)
    assert new_params["sigma"] == pytest.approx(1.0, abs=0.1)


def test_recovers_the_scale_of_a_component_with_fixed_mean(sample_mixture: tuple[MixtureModel, np.ndarray]) -> None:
    mixture, sample = sample_mixture
    start = family(FamilyName.NORMAL).view(mu=5.0)(sigma=2.0)

    _, new_params = observed_data_likelihood_strategy(
        start, make_state(sample, None, mixture), odl_block(), ScipyNelderMead()
    )

    assert set(new_params) == {"sigma"}
    assert new_params["sigma"] == pytest.approx(1.0, abs=0.1)
