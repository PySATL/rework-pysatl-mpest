"""Unit test module which test the Peak method"""

import pytest
from rework_pysatl_mpest.distributions import Exponential, Normal, Weibull
from rework_pysatl_mpest.preprocessing.components_number import Peaks
from rework_tests.unit.preprocessing.components_number.components_num_utils import run_test


@pytest.mark.parametrize(
    "components, weights, size",
    [
        ([Normal(5.0, 2.0)], [1.0], 200),
        (
            [Weibull(5.0, 0.0, 2.0), Weibull(7.0, 0.0, 1.0), Weibull(11.0, 0.0, 3.0)],
            [0.33, 0.33, 0.34],
            500,
        ),
        (
            [Weibull(4.0, 0.0, 2.0), Normal(7.5, 2.5), Weibull(10.0, 0.0, 4.0)],
            [0.2, 0.4, 0.4],
            1000,
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
            [Weibull(10.0, 0.0, 1.0), Weibull(4.0, 0.0, 6.0), Exponential(0.0, 3.5)],
            [0.2, 0.4, 0.4],
            200,
        ),
        (
            [Exponential(0.0, 0.5), Exponential(0.0, 3.5), Normal(9.0, 3.5), Normal(3.0, 6.0)],
            [0.1, 0.2, 0.4, 0.3],
            5000,
        ),
        (
            [Normal(3.0, 1.5), Weibull(7.0, 0.0, 2.0)],
            [0.7, 0.3],
            1000,
        ),
    ],
)
def test_incorrect_estimating(components, weights, size):
    """Runs the Peak method with a negative outcome"""
    assert run_test(components, weights, size, Peaks()) != len(components)
