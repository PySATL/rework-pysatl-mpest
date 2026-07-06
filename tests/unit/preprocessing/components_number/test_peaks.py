"""Unit test module which test the Peak method"""

import pytest
from pysatl_mpest.distributions import Normal
from pysatl_mpest.preprocessing.components_number import Peaks
from tests.unit.preprocessing.components_number.components_num_utils import run_test


@pytest.mark.parametrize(
    "components, weights, size",
    [
        (
            [Normal(mu=5.0, sigma=1.0), Normal(mu=15.0, sigma=1.0)],
            [0.5, 0.5],
            2000,
        ),
        (
            [Normal(mu=0.0, sigma=1.0), Normal(mu=10.0, sigma=1.0), Normal(mu=20.0, sigma=1.0)],
            [0.33, 0.33, 0.34],
            2000,
        ),
        (
            [Normal(mu=0.0, sigma=1.0), Normal(mu=10.0, sigma=1.0), Normal(mu=20.0, sigma=1.0)],
            [0.2, 0.4, 0.4],
            2000,
        ),
    ],
)
def test_correct_estimating(components, weights, size):
    """Runs the Peak method with a positive outcome"""
    assert run_test(components, weights, size, Peaks()) == len(components)


@pytest.mark.parametrize(
    "components, weights, size",
    [
        (
            [Normal(mu=0.0, sigma=1.0), Normal(mu=0.5, sigma=1.0), Normal(mu=1.0, sigma=1.0)],
            [0.2, 0.4, 0.4],
            2000,
        ),
        (
            [
                Normal(mu=0.0, sigma=1.0),
                Normal(mu=0.5, sigma=1.0),
                Normal(mu=1.0, sigma=1.0),
                Normal(mu=1.5, sigma=1.0),
            ],
            [0.1, 0.2, 0.4, 0.3],
            5000,
        ),
        (
            [Normal(mu=0.0, sigma=1.0), Normal(mu=0.5, sigma=1.0)],
            [0.5, 0.5],
            2000,
        ),
    ],
)
def test_incorrect_estimating(components, weights, size):
    """Runs the Peak method with a negative outcome"""
    assert run_test(components, weights, size, Peaks()) != len(components)
