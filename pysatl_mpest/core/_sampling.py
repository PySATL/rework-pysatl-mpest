"""
Module providing the sampling strategy for deterministic sampling.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import numpy as np
from pysatl_core.distributions.distribution import Distribution
from pysatl_core.distributions.strategies import SamplingStrategy
from pysatl_core.types import CharacteristicName, NumericArray


class DeterministicSampling(SamplingStrategy):
    """
    Sampling strategy that produces deterministic samples based on a random state.
    """

    def sample(self, n: int, distr: Distribution, **options: Any) -> NumericArray:
        """
        Samples values from the distribution using the Percent Point Function (PPF).

        Parameters
        ----------
        n : int
            The number of samples to generate.
        distr : Distribution
            The distribution to sample from.
        **options : Any
            Additional options for sampling, such as 'random_state'.

        Returns
        -------
        NumericArray
            An array of generated samples.
        """
        seed = options.get("random_state")
        rng = np.random.default_rng(seed)

        return distr.calculate_characteristic(CharacteristicName.PPF, rng.random(n))
