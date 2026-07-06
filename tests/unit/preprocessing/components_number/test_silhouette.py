"""Unit test module which test the Silhouette method"""

import pytest
from pysatl_mpest.distributions import Normal
from pysatl_mpest.preprocessing.components_number import Silhouette
from tests.unit.preprocessing.components_number.components_num_utils import run_test


@pytest.mark.parametrize(
    "components, weights, size, kmax",
    [
        (
            [Normal(mu=0.0, sigma=1.0), Normal(mu=10.0, sigma=1.0)],
            [0.6, 0.4],
            200,
            10,
        ),
        (
            [Normal(mu=0.0, sigma=1.0), Normal(mu=10.0, sigma=1.0), Normal(mu=20.0, sigma=1.0)],
            [0.3, 0.3, 0.4],
            500,
            10,
        ),
        (
            [
                Normal(mu=0.0, sigma=1.0),
                Normal(mu=10.0, sigma=1.0),
                Normal(mu=20.0, sigma=1.0),
                Normal(mu=30.0, sigma=1.0),
            ],
            [0.5, 0.3, 0.1, 0.1],
            500,
            10,
        ),
    ],
)
def test_correct_estimating(components, weights, size, kmax):
    """Runs the Silhouette method with a positive outcome"""
    assert run_test(components, weights, size, Silhouette(kmax, random_state=42)) == len(components)


@pytest.mark.parametrize(
    "components, weights, size, kmax",
    [
        (
            [Normal(mu=0.0, sigma=5.0), Normal(mu=0.0, sigma=5.0), Normal(mu=0.0, sigma=5.0)],
            [0.33, 0.33, 0.34],
            200,
            10,
        ),
        ([Normal(mu=0.0, sigma=1.0)], [1.0], 500, 10),
        (
            [Normal(mu=0.0, sigma=5.0), Normal(mu=0.0, sigma=5.0), Normal(mu=0.0, sigma=5.0)],
            [0.4, 0.5, 0.1],
            1000,
            10,
        ),
    ],
)
def test_incorrect_estimating(components, weights, size, kmax):
    """Runs the Silhouette method with a negative outcome"""
    assert run_test(components, weights, size, Silhouette(kmax, random_state=42)) != len(components)
