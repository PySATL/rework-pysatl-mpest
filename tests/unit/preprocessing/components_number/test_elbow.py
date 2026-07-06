"""Unit test module which test the Elbow method"""

import pytest
from pysatl_mpest.distributions import Normal, Uniform
from pysatl_mpest.preprocessing.components_number import Elbow
from tests.unit.preprocessing.components_number.components_num_utils import run_test


@pytest.mark.parametrize(
    "components, weights, size, kmax",
    [
        (
            [Uniform(lower_bound=0.0, upper_bound=2.0), Normal(mu=5.0, sigma=1.0), Normal(mu=15.0, sigma=2.0)],
            [0.33, 0.34, 0.33],
            200,
            15,
        ),
        (
            [Normal(mu=5.0, sigma=2.0), Normal(mu=15.0, sigma=2.0)],
            [0.6, 0.4],
            500,
            15,
        ),
        (
            [
                Normal(mu=0.0, sigma=1.0),
                Normal(mu=10.0, sigma=1.0),
                Normal(mu=20.0, sigma=1.0),
                Normal(mu=30.0, sigma=1.0),
            ],
            [0.25, 0.25, 0.25, 0.25],
            1000,
            20,
        ),
    ],
)
def test_correct_estimating(components, weights, size, kmax):
    """Runs the Elbow method with a positive outcome"""
    assert run_test(components, weights, size, Elbow(kmax, random_state=42)) == len(components)


@pytest.mark.parametrize(
    "components, weights, size, kmax",
    [
        (
            [Normal(mu=0.0, sigma=1.0), Normal(mu=0.0, sigma=1.0), Normal(mu=0.0, sigma=1.0)],
            [0.6, 0.2, 0.2],
            200,
            20,
        ),
        (
            [Normal(mu=5.0, sigma=2.0), Uniform(lower_bound=4.0, upper_bound=10.0)],
            [0.5, 0.5],
            500,
            15,
        ),
        (
            [Normal(mu=0.0, sigma=5.0), Normal(mu=0.5, sigma=5.0), Normal(mu=1.0, sigma=5.0)],
            [0.33, 0.33, 0.34],
            1000,
            15,
        ),
    ],
)
def test_incorrect_estimating(components, weights, size, kmax):
    """Runs the Elbow method with a negative outcome"""
    assert run_test(components, weights, size, Elbow(kmax, random_state=42)) != len(components)
