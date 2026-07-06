"""Unit test module which test the Peak method"""

import pytest
from pysatl_mpest.distributions import Normal
from pysatl_mpest.preprocessing.components_number import Peaks
from tests.unit.preprocessing.components_number.components_num_utils import run_test


@pytest.mark.parametrize(
    "component_params, weights, size",
    [
        (
            [(5.0, 1.0), (15.0, 1.0)],
            [0.5, 0.5],
            2000,
        ),
        (
            [(0.0, 1.0), (10.0, 1.0), (20.0, 1.0)],
            [0.33, 0.33, 0.34],
            2000,
        ),
        (
            [(0.0, 1.0), (10.0, 1.0), (20.0, 1.0)],
            [0.2, 0.4, 0.4],
            2000,
        ),
    ],
)
def test_correct_estimating(component_params, weights, size):
    """Runs the Peak method with a positive outcome"""
    components = [Normal(mu=mu, sigma=sigma) for mu, sigma in component_params]
    assert run_test(components, weights, size, Peaks()) == len(components)


@pytest.mark.parametrize(
    "component_params, weights, size",
    [
        (
            [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)],
            [0.2, 0.4, 0.4],
            2000,
        ),
        (
            [
                (0.0, 1.0),
                (0.5, 1.0),
                (1.0, 1.0),
                (1.5, 1.0),
            ],
            [0.1, 0.2, 0.4, 0.3],
            5000,
        ),
        (
            [(0.0, 1.0), (0.5, 1.0)],
            [0.5, 0.5],
            2000,
        ),
    ],
)
def test_incorrect_estimating(component_params, weights, size):
    """Runs the Peak method with a negative outcome"""
    components = [Normal(mu=mu, sigma=sigma) for mu, sigma in component_params]
    assert run_test(components, weights, size, Peaks()) != len(components)
