"""
Integration tests for ``MixtureModel`` equality, hashing and sampling on real pysatl-core families.

Components created through ``family.view(...)`` are compared and sampled the same way as plain ones.
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
from tests.helpers.families import family
from tests.helpers.math_assertions import assert_is_array_type, assert_is_scalar_type


def _normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


def _normal_fixed_mu(mu: float, sigma: float) -> ParametricFamilyDistribution:
    """A normal component whose ``mu`` is fixed through ``view``; only ``sigma`` is free."""
    return family(FamilyName.NORMAL).view(mu=mu)(sigma=sigma)


def _exponential(lambda_: float) -> ParametricFamilyDistribution:
    return family(FamilyName.EXPONENTIAL)(lambda_=lambda_)


def test_independently_built_mixtures_are_equal() -> None:
    first = MixtureModel([_normal(0.0, 1.0), _exponential(2.0)], weights=[0.3, 0.7])
    second = MixtureModel([_normal(0.0, 1.0), _exponential(2.0)], weights=[0.3, 0.7])

    assert first == second
    assert hash(first) == hash(second)


def test_mixtures_with_view_components_are_equal() -> None:
    first = MixtureModel([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], weights=[0.4, 0.6])
    second = MixtureModel([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], weights=[0.4, 0.6])

    assert first == second
    assert hash(first) == hash(second)


def test_view_component_equals_full_component_with_same_values() -> None:
    with_view = MixtureModel([_normal_fixed_mu(1.0, 2.0)])
    full = MixtureModel([_normal(1.0, 2.0)])

    assert with_view == full
    assert hash(with_view) == hash(full)


def test_equality_ignores_component_order() -> None:
    first = MixtureModel([_normal(0.0, 1.0), _exponential(2.0)], weights=[0.3, 0.7])
    second = MixtureModel([_exponential(2.0), _normal(0.0, 1.0)], weights=[0.7, 0.3])

    assert first == second
    assert hash(first) == hash(second)


def test_mixtures_with_different_fixed_value_differ() -> None:
    first = MixtureModel([_normal_fixed_mu(1.0, 2.0)])
    second = MixtureModel([_normal_fixed_mu(2.0, 2.0)])

    assert first != second


def test_mixtures_with_different_free_value_differ() -> None:
    first = MixtureModel([_normal_fixed_mu(1.0, 2.0)])
    second = MixtureModel([_normal_fixed_mu(1.0, 3.0)])

    assert first != second


def test_zero_weight_component_does_not_make_mixtures_equal() -> None:
    component, extra = _normal(0.0, 1.0), _normal(5.0, 1.0)

    assert MixtureModel([component, extra], weights=[1.0, 0.0]) != MixtureModel([component])


def test_copy_with_arbitrary_weights_is_equal_and_hashes_equal() -> None:
    """Copying passes the weights through ``log`` and ``softmax``, which changes them by about 1e-16."""
    components = [_normal(0.0, 1.0), _normal_fixed_mu(3.0, 2.0), _exponential(2.0)]
    rng = np.random.default_rng(0)

    for weights in rng.dirichlet(np.ones(3), size=500):
        mixture = MixtureModel(components, weights=weights)
        duplicate = copy(mixture)

        assert duplicate == mixture
        assert hash(duplicate) == hash(mixture)


def test_mixture_is_usable_as_dict_key() -> None:
    mixture = MixtureModel([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], weights=[0.4, 0.6])
    twin = MixtureModel([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], weights=[0.4, 0.6])

    assert {mixture: "fitted"}[twin] == "fitted"


@pytest.mark.parametrize(
    "components",
    [
        pytest.param([_normal(0.0, 1.0), _exponential(2.0)], id="plain"),
        pytest.param([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], id="view"),
    ],
)
def test_generate_is_reproducible_with_seed(components: list[ParametricFamilyDistribution]) -> None:
    mixture = MixtureModel(components, weights=[0.5, 0.5])
    seed, other_seed, size = 42, 43, 200

    first = mixture.generate(size=size, random_state=seed)
    second = mixture.generate(size=size, random_state=seed)
    from_generator = mixture.generate(size=size, random_state=np.random.default_rng(seed))
    other = mixture.generate(size=size, random_state=other_seed)

    assert_allclose(first, second)
    assert_allclose(first, from_generator)
    assert not np.allclose(first, other)


def test_generate_shapes() -> None:
    mixture = MixtureModel([_normal_fixed_mu(1.0, 2.0), _exponential(2.0)], weights=[0.5, 0.5])
    seed = 0

    assert_is_scalar_type(mixture.generate(random_state=seed))
    assert_is_array_type(mixture.generate(size=7, random_state=seed), (7,))
    assert_is_array_type(mixture.generate(size=(3, 4), random_state=seed), (3, 4))


def test_generate_follows_the_mixture() -> None:
    """With well separated components and equal weights about half of the sample lies on each side."""
    mixture = MixtureModel([_normal(0.0, 1.0), _normal_fixed_mu(20.0, 1.0)], weights=[0.5, 0.5])
    size, boundary, tolerance = 20_000, 10.0, 0.02

    sample = np.asarray(mixture.generate(size=size, random_state=2026))
    left, right = sample[sample < boundary], sample[sample >= boundary]

    assert abs(len(left) / size - 0.5) < tolerance
    assert_allclose(np.mean(left), 0.0, atol=0.05)
    assert_allclose(np.mean(right), 20.0, atol=0.05)
