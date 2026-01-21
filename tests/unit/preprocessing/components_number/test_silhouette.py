"""Unit test module which test the Silhouette method"""

import pytest
from rework_pysatl_mpest.distributions import Exponential, Normal, Weibull
from rework_pysatl_mpest.preprocessing.components_number import Silhouette
from rework_tests.unit.preprocessing.components_number.components_num_utils import run_test


@pytest.mark.parametrize(
    "components, weights, size, kmax",
    [
        (
            [Weibull(2.0, 0.0, 10.0), Normal(5.0, 1.0)],
            [0.6, 0.4],
            200,
            10,
        ),
        (
            [Normal(-5.0, 3.0), Normal(2.0, 1.0), Normal(10.0, 2.0)],
            [0.3, 0.3, 0.4],
            500,
            10,
        ),
        (
            [Exponential(0.0, 0.5), Normal(1.0, 3.0), Normal(3.0, 10.0), Normal(5.0, 1.0)],
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
            [Weibull(1.0, 0.0, 2.0), Weibull(5.0, 0.0, 1.0), Exponential(0.0, 1.0)],
            [0.33, 0.33, 0.34],
            200,
            10,
        ),
        ([Exponential(0.0, 0.5)], [1.0], 500, 10),
        (
            [Exponential(0.0, 0.5), Exponential(0.0, 3.0), Weibull(3.0, 0.0, 0.5)],
            [0.4, 0.5, 0.1],
            1000,
            10,
        ),
    ],
)
def test_incorrect_estimating(components, weights, size, kmax):
    """Runs the Silhouette method with a negative outcome"""
    assert run_test(components, weights, size, Silhouette(kmax, random_state=42)) != len(components)
