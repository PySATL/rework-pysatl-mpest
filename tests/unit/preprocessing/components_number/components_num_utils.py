"""Module which contain utility for testing methods of estimating the number of components"""

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

    mixture = MixtureModel(components=components, weights=weights)

    X = mixture.generate(size, random_state=43)
    result = method.estimate(X)
    return result
