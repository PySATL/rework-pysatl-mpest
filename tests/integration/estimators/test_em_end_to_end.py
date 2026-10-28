"""
End-to-end tests of the ECM and ECME estimators on real pysatl-core components.

Nothing is mocked: the estimators run the pipeline with the real E- and M-steps, strategies and optimizers.
The analytical strategies must reproduce a plain numpy implementation of the EM algorithm iteration by
iteration, and the numerical ones must reach the same estimates up to the accuracy of the optimizer.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from copy import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators import ECM, ECME
from pysatl_mpest.estimators.iterative import LikelihoodBreakpointer, PriorThresholdPruner, StepBreakpointer
from pysatl_mpest.optimizers import Optimizer, ScipyNelderMead, ScipyPowell
from scipy import stats
from scipy.special import logsumexp
from tests.helpers.families import family

N_SAMPLES = 2000
TRUE_WEIGHTS = [0.4, 0.6]
LIKELIHOOD_SLACK = 1e-9  # a numerical error of a likelihood that is never meant to decrease


def normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


def exponential(lambda_: float) -> ParametricFamilyDistribution:
    return family(FamilyName.EXPONENTIAL)(lambda_=lambda_)


def parameters(mixture: MixtureModel) -> list[dict[str, float]]:
    return [cc.all_parameters(component) for component in mixture]


def sample_of(mixture: MixtureModel, seed: int = 0) -> np.ndarray:
    return np.asarray(mixture.generate(size=N_SAMPLES, random_state=seed))


@pytest.fixture
def true_mixture() -> MixtureModel:
    return MixtureModel([normal(0.0, 1.0), normal(6.0, 1.5)], weights=TRUE_WEIGHTS)


@pytest.fixture
def X(true_mixture: MixtureModel) -> np.ndarray:
    return sample_of(true_mixture)


@pytest.fixture
def start() -> MixtureModel:
    return MixtureModel([normal(1.0, 2.0), normal(4.0, 2.0)], weights=[0.5, 0.5])


def ecm_for(iterations: int, optimizer: Optimizer | None = None, tolerance: float | None = None) -> ECM:
    """An ECM that runs exactly ``iterations`` iterations, or fewer if the likelihood stops changing."""

    breakpointers = [StepBreakpointer(iterations)]
    if tolerance is not None:
        breakpointers.append(LikelihoodBreakpointer(tolerance))
    return ECM(breakpointers=breakpointers, pruners=[], optimizer=optimizer or ScipyNelderMead())


def reference_em(
    X: np.ndarray, mixture: MixtureModel, iterations: int, fixed_mu: tuple[bool, ...] = ()
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The EM algorithm for a mixture of normals written directly in numpy.

    Returns the means, standard deviations and weights after ``iterations`` iterations. ``fixed_mu`` marks the
    components whose mean is not updated; their standard deviation is measured around the fixed mean.
    """

    params = parameters(mixture)
    mu = np.array([p["mu"] for p in params])
    sigma = np.array([p["sigma"] for p in params])
    weights = np.array(mixture.weights)
    frozen = np.array(fixed_mu or (False,) * len(mu))

    for _ in range(iterations):
        joint = np.log(weights) + stats.norm(mu, sigma).logpdf(X[:, None])
        resp = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
        total = resp.sum(axis=0)
        weights = total / len(X)
        mu = np.where(frozen, mu, resp.T @ X / total)
        sigma = np.sqrt((resp * (X[:, None] - mu) ** 2).sum(axis=0) / total)

    return mu, sigma, weights


def fitted_arrays(mixture: MixtureModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = parameters(mixture)
    return (
        np.array([p["mu"] for p in params]),
        np.array([p["sigma"] for p in params]),
        np.array(mixture.weights),
    )


# --- ECM with analytical strategies ---


@pytest.mark.parametrize("iterations", [1, 5, 25])
def test_ecm_reproduces_the_reference_em_iteration_by_iteration(
    X: np.ndarray, start: MixtureModel, iterations: int
) -> None:
    fitted = ecm_for(iterations).fit(X, start)

    for actual, expected in zip(fitted_arrays(fitted), reference_em(X, start, iterations)):
        assert_allclose(actual, expected, rtol=1e-9)


def test_ecm_recovers_the_parameters_of_a_mixture(X: np.ndarray, start: MixtureModel) -> None:
    fitted = ecm_for(300, tolerance=1e-9).fit(X, start)
    mu, sigma, weights = fitted_arrays(fitted)

    assert_allclose(mu, [0.0, 6.0], atol=0.15)
    assert_allclose(sigma, [1.0, 1.5], atol=0.15)
    assert_allclose(weights, TRUE_WEIGHTS, atol=0.03)


def test_ecm_never_decreases_the_log_likelihood(X: np.ndarray, start: MixtureModel) -> None:
    ecm = ecm_for(40)
    fitted = ecm.fit(X, start)

    likelihoods = [record.mixture.loglikelihood(X) for record in (ecm.history[i] for i in range(len(ecm.history)))]

    assert len(likelihoods) == 40  # noqa: PLR2004
    assert np.all(np.diff(likelihoods) >= -LIKELIHOOD_SLACK)
    assert likelihoods[-1] > start.loglikelihood(X)
    assert fitted.loglikelihood(X) == pytest.approx(likelihoods[-1])


def test_ecm_reaches_a_likelihood_not_below_the_truth(
    X: np.ndarray, true_mixture: MixtureModel, start: MixtureModel
) -> None:
    fitted = ecm_for(300, tolerance=1e-9).fit(X, start)

    assert fitted.loglikelihood(X) >= true_mixture.loglikelihood(X) - 1e-6


def test_ecm_keeps_a_parameter_fixed_through_a_view(X: np.ndarray) -> None:
    fixed = family(FamilyName.NORMAL).view(mu=0.0)(sigma=2.0)
    start = MixtureModel([fixed, normal(4.0, 2.0)], weights=[0.5, 0.5])

    fitted = ecm_for(25).fit(X, start)

    assert cc.fixed_parameters(fitted[0]) == {"mu": 0.0}
    assert cc.all_parameters(fitted[0])["mu"] == 0.0
    expected = reference_em(X, start, 25, fixed_mu=(True, False))
    for actual, reference in zip(fitted_arrays(fitted), expected):
        assert_allclose(actual, reference, rtol=1e-9)


def test_ecm_does_not_modify_the_initial_mixture(X: np.ndarray, start: MixtureModel) -> None:
    before, components = copy(start), start.components

    ecm_for(5).fit(X, start)

    assert start == before
    assert start.components == components


def test_ecm_is_deterministic(X: np.ndarray, start: MixtureModel) -> None:
    assert ecm_for(10).fit(X, start) == ecm_for(10).fit(X, start)


# --- stopping ---


def test_step_breakpointer_sets_the_number_of_iterations(X: np.ndarray, start: MixtureModel) -> None:
    ecm = ecm_for(7)

    ecm.fit(X, start)

    assert len(ecm.history) == 7  # noqa: PLR2004


def test_likelihood_breakpointer_stops_before_the_step_limit(X: np.ndarray, start: MixtureModel) -> None:
    ecm = ecm_for(500, tolerance=1e-6)

    fitted = ecm.fit(X, start)

    assert 1 < len(ecm.history) < 500  # noqa: PLR2004
    assert_allclose(fitted_arrays(fitted)[2], TRUE_WEIGHTS, atol=0.03)


# --- ECM with numerical strategies ---


@pytest.mark.parametrize("optimizer_class", [ScipyNelderMead, ScipyPowell])
def test_ecm_with_numerical_strategy_matches_the_analytical_one(
    X: np.ndarray, start: MixtureModel, optimizer_class: type[Optimizer]
) -> None:
    """The first component is given in the ``meanVar`` parametrization, which has no analytical solution."""

    mean_var = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=1.0, var=4.0)
    numerical_start = MixtureModel([mean_var, start[1]], weights=[0.5, 0.5])
    iterations = 15

    numerical = ecm_for(iterations, optimizer=optimizer_class()).fit(X, numerical_start)
    analytical = ecm_for(iterations).fit(X, start)

    numerical_params, analytical_params = parameters(numerical), parameters(analytical)
    assert numerical[0].parametrization_name == "meanVar"
    assert numerical_params[0]["mu"] == pytest.approx(analytical_params[0]["mu"], abs=5e-3)
    assert numerical_params[0]["var"] == pytest.approx(analytical_params[0]["sigma"] ** 2, rel=5e-3)
    assert numerical_params[1] == pytest.approx(analytical_params[1], rel=5e-3)
    assert_allclose(numerical.weights, analytical.weights, atol=5e-3)


def test_ecm_with_numerical_strategy_keeps_a_fixed_parameter(X: np.ndarray, start: MixtureModel) -> None:
    fixed = family(FamilyName.NORMAL).view(parametrization_name="meanVar", mu=0.0)(var=4.0)
    numerical_start = MixtureModel([fixed, start[1]], weights=[0.5, 0.5])

    fitted = ecm_for(10).fit(X, numerical_start)

    assert cc.all_parameters(fitted[0])["mu"] == 0.0
    assert cc.all_parameters(fitted[0])["var"] != cc.all_parameters(fixed)["var"]


# --- a mixture of different families ---


@pytest.fixture
def exponential_normal_sample() -> tuple[MixtureModel, np.ndarray]:
    truth = MixtureModel([exponential(1.5), normal(8.0, 1.0)], weights=[0.5, 0.5])
    return truth, sample_of(truth)


def test_ecm_recovers_a_mixture_of_an_exponential_and_a_normal(
    exponential_normal_sample: tuple[MixtureModel, np.ndarray],
) -> None:
    _, X = exponential_normal_sample
    start = MixtureModel([exponential(0.5), normal(6.0, 2.0)], weights=[0.5, 0.5])

    fitted = ecm_for(300, tolerance=1e-9).fit(X, start)

    assert cc.all_parameters(fitted[0])["lambda_"] == pytest.approx(1.5, rel=0.1)
    assert cc.all_parameters(fitted[1]) == pytest.approx({"mu": 8.0, "sigma": 1.0}, abs=0.15)
    assert_allclose(fitted.weights, [0.5, 0.5], atol=0.03)


# --- ECME ---


def test_ecme_with_q_function_strategy_equals_ecm(X: np.ndarray, start: MixtureModel) -> None:
    ecme = ECME(
        breakpointers=[StepBreakpointer(10)], pruners=[], optimizer=ScipyNelderMead(), default_strategy="q-func"
    )

    assert ecme.fit(X, start) == ecm_for(10).fit(X, start)


@pytest.mark.parametrize(
    ("default_strategy", "q_indices", "odl_indices"),
    [("odl", None, None), ("q-func", None, 1), ("q-func", 0, 1)],
    ids=["odl everywhere", "odl for the second", "q-function and odl"],
)
def test_ecme_with_observed_data_likelihood_improves_the_fit(
    X: np.ndarray,
    start: MixtureModel,
    default_strategy: str,
    q_indices: int | None,
    odl_indices: int | None,
) -> None:
    ecme = ECME(
        breakpointers=[StepBreakpointer(25)],
        pruners=[],
        optimizer=ScipyNelderMead(),
        default_strategy=default_strategy,  # type: ignore[arg-type]
    )

    fitted = ecme.fit(X, start, q_indices_raw=q_indices, odl_indices_raw=odl_indices)

    assert fitted.loglikelihood(X) > start.loglikelihood(X)
    mu, sigma, weights = fitted_arrays(fitted)
    assert_allclose(mu, [0.0, 6.0], atol=0.2)
    assert_allclose(sigma, [1.0, 1.5], atol=0.2)
    assert_allclose(weights, TRUE_WEIGHTS, atol=0.05)
    assert len(ecme.history) == 25  # noqa: PLR2004


# --- pruning ---

PRUNING_THRESHOLD = 0.05


def stray_start(stray_positions: tuple[int, ...]) -> MixtureModel:
    """The two true clusters plus components with a negligible weight that are inserted at the given positions."""

    stray_weight = 0.01
    components = [normal(0.0, 1.0), normal(6.0, 1.0)]
    weights = [(1.0 - stray_weight * len(stray_positions)) / 2] * 2
    for position, mu in zip(stray_positions, (3.0, -3.0, 9.0)):
        components.insert(position, normal(mu, 1.0))
        weights.insert(position, stray_weight)
    return MixtureModel(components, weights=weights)


def pruning_ecm(iterations: int = 15) -> ECM:
    return ECM(
        breakpointers=[StepBreakpointer(iterations)],
        pruners=[PriorThresholdPruner(PRUNING_THRESHOLD)],
        optimizer=ScipyNelderMead(),
    )


@pytest.mark.parametrize("position", [0, 1, 2], ids=["first", "middle", "last"])
def test_ecm_prunes_a_component_with_a_negligible_weight(X: np.ndarray, position: int) -> None:
    fitted = pruning_ecm().fit(X, stray_start((position,)))

    mu, sigma, weights = fitted_arrays(fitted)
    assert fitted.n_components == 2  # noqa: PLR2004
    assert_allclose(mu, [0.0, 6.0], atol=0.15)
    assert_allclose(sigma, [1.0, 1.5], atol=0.15)
    assert_allclose(weights, TRUE_WEIGHTS, atol=0.03)


@pytest.mark.parametrize("positions", [(0, 2), (1, 3), (0, 1)], ids=["0 and 2", "1 and 3", "0 and 1"])
def test_ecm_prunes_several_components_at_once(X: np.ndarray, positions: tuple[int, ...]) -> None:
    fitted = pruning_ecm().fit(X, stray_start(positions))

    assert fitted.n_components == 2  # noqa: PLR2004
    assert_allclose(fitted_arrays(fitted)[0], [0.0, 6.0], atol=0.15)


def test_ecm_with_pruning_continues_like_a_fit_of_the_pruned_mixture(X: np.ndarray) -> None:
    """After the pruner removes a component the fit goes on exactly as it would for the smaller mixture."""

    start = stray_start((1,))
    after_first = ecm_for(1).fit(X, start)
    assert after_first.weights[1] < PRUNING_THRESHOLD
    pruned = copy(after_first)
    pruned.remove_component(1)

    with_pruner = pruning_ecm(iterations=10).fit(X, start)
    by_hand = ecm_for(9).fit(X, pruned)

    for actual, expected in zip(fitted_arrays(with_pruner), fitted_arrays(by_hand)):
        assert_allclose(actual, expected, rtol=1e-9)


def test_ecm_with_pruning_keeps_the_history_consistent(X: np.ndarray) -> None:
    ecm = pruning_ecm(iterations=6)

    ecm.fit(X, stray_start((1,)))

    records = [ecm.history[i] for i in range(len(ecm.history))]
    assert [record.mixture.n_components for record in records][-1] == 2  # noqa: PLR2004
    for record in records:
        assert record.H is not None
        assert record.H.shape == (N_SAMPLES, record.mixture.n_components)


def test_ecm_with_pruning_keeps_a_fixed_parameter(X: np.ndarray) -> None:
    fixed = family(FamilyName.NORMAL).view(mu=6.0)(sigma=2.0)
    start = MixtureModel([normal(0.0, 1.0), normal(3.0, 1.0), fixed], weights=[0.49, 0.02, 0.49])

    fitted = pruning_ecm().fit(X, start)

    assert fitted.n_components == 2  # noqa: PLR2004
    assert cc.fixed_parameters(fitted[1]) == {"mu": 6.0}
    assert cc.all_parameters(fitted[1])["sigma"] == pytest.approx(1.5, abs=0.15)


@pytest.mark.parametrize("default_strategy", ["q-func", "odl"])
def test_ecme_prunes_a_component_with_a_negligible_weight(X: np.ndarray, default_strategy: str) -> None:
    ecme = ECME(
        breakpointers=[StepBreakpointer(12)],
        pruners=[PriorThresholdPruner(PRUNING_THRESHOLD)],
        optimizer=ScipyNelderMead(),
        default_strategy=default_strategy,  # type: ignore[arg-type]
    )

    fitted = ecme.fit(X, stray_start((0,)))

    assert fitted.n_components == 2  # noqa: PLR2004
    assert_allclose(fitted_arrays(fitted)[0], [0.0, 6.0], atol=0.2)
    assert_allclose(fitted.weights, TRUE_WEIGHTS, atol=0.05)


def test_fit_with_a_point_outside_the_support_keeps_pdf_equal_to_exp_lpdf() -> None:
    """The M-step normalizes weights when observations outside the support contribute no mass."""
    start = MixtureModel([exponential(1.0), exponential(3.0)])
    data = np.array([-1.0, -2.0, 0.5, 1.0, 2.0, 0.1])

    fitted = ecm_for(1).fit(data, start)

    assert np.sum(np.exp(fitted.log_weights)) == pytest.approx(1.0)
    on_support = data[2:]
    assert fitted.pdf(on_support) == pytest.approx(np.exp(fitted.lpdf(on_support)))
