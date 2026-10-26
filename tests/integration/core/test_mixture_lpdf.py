"""
Integration tests for ``MixtureModel.pdf`` / ``lpdf`` / ``loglikelihood`` on real pysatl-core families.

Unlike ``tests/unit/core/test_mixture.py`` these tests do not mock ``finite_mixture``, so they exercise
the actual ``FiniteMixtureDistribution`` from pysatl-core. The reference values are computed with
``scipy.stats`` and ``numpy.logaddexp``.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from scipy import stats
from tests.helpers.families import family
from tests.helpers.math_assertions import assert_is_array_type, assert_is_scalar_type

RTOL, ATOL = 1e-10, 1e-12

# pysatl-core does not implement ``lpdf`` for a finite mixture (``LPDF = "lpdf"  # unimplemented in graph yet``).
# Remove this marker when ``MixtureModel.lpdf`` computes it itself (step 2 of MIGRATION-PLAN.md).
lpdf_not_implemented = pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="finite mixture of pysatl-core has no lpdf",
)

Component = tuple[ParametricFamilyDistribution, "stats.rv_continuous"]


def _normal(mu: float, sigma: float) -> Component:
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma), stats.norm(loc=mu, scale=sigma)


def _exponential(lambda_: float) -> Component:
    return family(FamilyName.EXPONENTIAL)(lambda_=lambda_), stats.expon(scale=1.0 / lambda_)


def _uniform(lower: float, upper: float) -> Component:
    return (
        family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=lower, upper_bound=upper),
        stats.uniform(loc=lower, scale=upper - lower),
    )


class Case:
    """A mixture together with scipy reference densities for its components."""

    def __init__(self, components: Sequence[Component], weights: Sequence[float]) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.references = [reference for _, reference in components]
        self.mixture = MixtureModel([component for component, _ in components], weights=self.weights)

    def reference_lpdf(self, x: np.ndarray) -> np.ndarray:
        """Log-density of the mixture computed with scipy and ``logaddexp``."""

        terms = [np.log(w) + ref.logpdf(x) for w, ref in zip(self.weights, self.references)]
        return np.logaddexp.reduce(terms, axis=0)

    def reference_pdf(self, x: np.ndarray) -> np.ndarray:
        """Density of the mixture computed with scipy."""

        return sum(w * ref.pdf(x) for w, ref in zip(self.weights, self.references))


CASE_FACTORIES: dict[str, Callable[[], Case]] = {
    "normal+normal": lambda: Case([_normal(0.0, 1.0), _normal(3.0, 2.0)], [0.4, 0.6]),
    "normal+exponential": lambda: Case([_normal(0.0, 1.0), _exponential(1.5)], [0.3, 0.7]),
    "normal+uniform": lambda: Case([_normal(2.0, 1.0), _uniform(0.0, 1.0)], [0.5, 0.5]),
}

cases = pytest.mark.parametrize("make_case", CASE_FACTORIES.values(), ids=CASE_FACTORIES.keys())

# Fixed grid: covers the exponential support boundary and the uniform support boundaries.
X_GRID = np.linspace(-3.0, 6.0, 25)


@cases
def test_pdf_matches_reference(make_case: Callable[[], Case]) -> None:
    """The density delegated to pysatl-core agrees with the scipy reference."""

    case = make_case()
    assert_allclose(case.mixture.pdf(X_GRID), case.reference_pdf(X_GRID), rtol=RTOL, atol=ATOL)


@lpdf_not_implemented
@cases
def test_lpdf_matches_reference(make_case: Callable[[], Case]) -> None:
    """The log-density agrees with ``logaddexp(log w + scipy logpdf)``, including points outside some supports."""

    case = make_case()
    assert_allclose(case.mixture.lpdf(X_GRID), case.reference_lpdf(X_GRID), rtol=RTOL, atol=ATOL)


@lpdf_not_implemented
@cases
def test_pdf_is_exp_of_lpdf(make_case: Callable[[], Case]) -> None:
    """``pdf`` and ``lpdf`` are consistent."""

    case = make_case()
    assert_allclose(case.mixture.pdf(X_GRID), np.exp(case.mixture.lpdf(X_GRID)), rtol=RTOL, atol=ATOL)


@lpdf_not_implemented
def test_lpdf_far_tail_is_finite() -> None:
    """Far in the tails ``pdf`` underflows to 0 while ``lpdf`` must stay finite and exact."""

    case = CASE_FACTORIES["normal+normal"]()
    x = np.array([-100.0, 100.0])

    assert np.all(case.mixture.pdf(x) == 0.0)  # documents why lpdf cannot be computed as log(pdf)
    lpdf = case.mixture.lpdf(x)
    assert np.all(np.isfinite(lpdf))
    assert_allclose(lpdf, case.reference_lpdf(x), rtol=RTOL, atol=ATOL)


@lpdf_not_implemented
@pytest.mark.filterwarnings("error")
def test_lpdf_outside_every_support_is_neg_inf() -> None:
    """Where every component has zero density the log-density is ``-inf`` without NaN or warnings."""

    case = Case([_exponential(1.0), _uniform(0.0, 1.0)], [0.5, 0.5])
    lpdf = case.mixture.lpdf(np.array([-1.0, -0.5]))

    assert np.all(np.isneginf(lpdf))


@lpdf_not_implemented
@cases
def test_lpdf_return_types(make_case: Callable[[], Case]) -> None:
    """A scalar input gives ``np.float64``; an array input gives a ``float64`` array of the same shape."""

    mixture = make_case().mixture

    assert_is_scalar_type(mixture.lpdf(0.5))
    assert_is_array_type(mixture.lpdf(X_GRID), X_GRID.shape)


@lpdf_not_implemented
@cases
def test_loglikelihood_is_sum_of_lpdf(make_case: Callable[[], Case]) -> None:
    """``loglikelihood`` equals the sum of the reference log-densities."""

    case = make_case()
    x = np.linspace(0.1, 5.0, 40)
    assert_allclose(case.mixture.loglikelihood(x), np.sum(case.reference_lpdf(x)), rtol=RTOL, atol=ATOL)
