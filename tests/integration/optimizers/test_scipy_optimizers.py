"""Integration tests for SciPy optimizers near an objective's boundary."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from numpy.typing import NDArray
from pysatl_mpest.optimizers.scipy_nelder_mead import ScipyNelderMead
from pysatl_mpest.optimizers.scipy_powell import ScipyPowell


def _bounded_objective(params: NDArray[np.float64]) -> float:
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return float((mu - 1) ** 2 + (sigma - 0.01) ** 2 + 1e-3 / sigma)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("optimizer", [ScipyPowell(), ScipyNelderMead()])
def test_optimizes_objective_with_infinite_boundary(optimizer: ScipyPowell | ScipyNelderMead) -> None:
    result = optimizer.minimize(_bounded_objective, [0.5, 0.05])

    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(1.0, abs=1e-3)
