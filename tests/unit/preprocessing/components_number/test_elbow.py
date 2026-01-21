"""Unit test module which test the Elbow method"""

import pytest
from rework_pysatl_mpest.distributions import Exponential, Normal, Weibull
from rework_pysatl_mpest.preprocessing.components_number import Elbow
from rework_tests.unit.preprocessing.components_number.components_num_utils import run_test


@pytest.mark.parametrize(
    "components, weights, size, kmax",
    [
        (
            [Weibull(1.0, 0.0, 0.5), Normal(5.0, 1.0), Normal(15.0, 2.0)],
            [0.33, 0.34, 0.33],
            200,
            15,
        ),
        (
            [Normal(5.0, 2.0), Normal(15.0, 2.0)],
            [0.6, 0.4],
            500,
            15,
        ),
        (
            [Weibull(11.0, 0.0, 2.5), Normal(5.0, 3.0), Exponential(0.0, 0.25), Weibull(18.0, 0.0, 2.0)],
            [0.2, 0.2, 0.4, 0.2],
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
            [Normal(5.0, 2.0), Normal(10.0, 2.0), Normal(15.0, 2.0)],
            [0.6, 0.2, 0.2],
            200,
            20,
        ),
        (
            [Normal(5.0, 2.0), Weibull(7.0, 0.0, 3.0)],
            [0.5, 0.5],
            500,
            15,
        ),
        (
            [Exponential(0.0, 0.5), Weibull(6.0, 0.0, 5.0), Weibull(7.0, 0.0, 5.0)],
            [0.1, 0.3, 0.6],
            1000,
            15,
        ),
    ],
)
def test_incorrect_estimating(components, weights, size, kmax):
    """Runs the Elbow method with a negative outcome"""
    assert run_test(components, weights, size, Elbow(kmax, random_state=42)) != len(components)
