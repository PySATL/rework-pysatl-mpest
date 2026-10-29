"""Sampling strategy for deterministic sampling."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import numpy as np
from pysatl_core.distributions.distribution import Distribution
from pysatl_core.distributions.strategies import SamplingStrategy
from pysatl_core.types import CharacteristicName, NumericArray


class DeterministicSampling(SamplingStrategy):
    """Deterministic sampling strategy using PPF inversion."""

    def sample(self, n: int, distr: Distribution, **options: Any) -> NumericArray:
        seed = options.get("random_state")
        rng = np.random.default_rng(seed)

        return distr.calculate_characteristic(CharacteristicName.PPF, rng.random(n))
