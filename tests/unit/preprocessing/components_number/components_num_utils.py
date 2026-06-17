"""Module which contain utility for testing methods of estimating the number of components"""

import numpy as np
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.distributions import ContinuousDistribution
from pysatl_mpest.preprocessing.components_number import AComponentsNumber


def run_test(
    components: list[ContinuousDistribution],
    weights: list[float],
    size: int,
    method: AComponentsNumber,
) -> int:
    """Run a test scenario"""

    np.random.seed(42)

    mixture = MixtureModel(components=components, weights=weights)

    X = mixture.generate(size)
    result = method.estimate(X)
    return result
