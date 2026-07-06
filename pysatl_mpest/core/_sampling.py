__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from pysatl_core.distributions.strategies import SamplingStrategy
from pysatl_core.types import CharacteristicName


class DeterministicSampling(SamplingStrategy):
    def sample(self, n, distr, **options):
        seed = options.get("random_state")
        rng = np.random.default_rng(seed)

        return distr.calculate_characteristic(CharacteristicName.PPF, rng.random(n))
