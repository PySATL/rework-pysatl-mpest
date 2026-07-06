import unittest.mock

import numpy as np
import pytest
from pysatl_core.types import CharacteristicName
from pysatl_mpest.core._sampling import DeterministicSampling


def test_sampling_calls_ppf_characteristic() -> None:
    """Test that sample() requests PPF from the distribution correctly."""
    sampler = DeterministicSampling()
    distr_mock = unittest.mock.MagicMock()

    # Let mock return a predefined array to verify mapping
    distr_mock.calculate_characteristic.return_value = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    n = 5
    result = sampler.sample(n=n, distr=distr_mock)

    # Assert return value matches the mock's output
    np.testing.assert_array_equal(result, [2.0, 4.0, 6.0, 8.0, 10.0])

    # Verify calculate_characteristic was called exactly once
    distr_mock.calculate_characteristic.assert_called_once()

    # Verify the arguments
    args, _ = distr_mock.calculate_characteristic.call_args
    assert args[0] == CharacteristicName.PPF

    # The second argument should be an array of random uniforms of size n
    random_array = args[1]
    assert isinstance(random_array, np.ndarray)
    assert random_array.shape == (n,)
    assert np.all((random_array >= 0.0) & (random_array < 1.0))


def test_sampling_with_random_state_integer() -> None:
    """Test that integer random_state produces deterministic results."""
    sampler = DeterministicSampling()
    distr_mock = unittest.mock.MagicMock()

    # Make the mock just return the array it receives
    distr_mock.calculate_characteristic.side_effect = lambda char, arr: arr

    n = 10
    seed = 42

    result1 = sampler.sample(n=n, distr=distr_mock, random_state=seed)
    result2 = sampler.sample(n=n, distr=distr_mock, random_state=seed)

    # With the same seed, the uniform arrays passed to the mock (and returned by it) must be identical
    np.testing.assert_array_equal(result1, result2)


def test_sampling_with_random_state_rng_instance() -> None:
    """Test that passing a numpy Generator instance works correctly."""
    sampler = DeterministicSampling()
    distr_mock = unittest.mock.MagicMock()
    distr_mock.calculate_characteristic.side_effect = lambda char, arr: arr

    n = 3
    seed = 123

    # Use generator inside sample
    rng_provided = np.random.default_rng(seed)
    result = sampler.sample(n=n, distr=distr_mock, random_state=rng_provided)

    # Replicate manually
    rng_manual = np.random.default_rng(seed)
    expected = rng_manual.random(n)

    np.testing.assert_array_equal(result, expected)


def test_sampling_without_random_state() -> None:
    """Test that sample works without random_state, producing varying outputs."""
    sampler = DeterministicSampling()
    distr_mock = unittest.mock.MagicMock()
    distr_mock.calculate_characteristic.side_effect = lambda char, arr: arr

    n = 5

    result1 = sampler.sample(n=n, distr=distr_mock)
    result2 = sampler.sample(n=n, distr=distr_mock)

    # Extremely unlikely to be exactly equal unless RNG is broken/fixed
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(result1, result2)
