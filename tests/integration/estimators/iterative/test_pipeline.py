"""Integration tests for complete iterative pipeline behavior."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators.iterative import (
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    Pipeline,
    PriorThresholdPruner,
    StepBreakpointer,
)
from pysatl_mpest.optimizers import ScipyNelderMead

from tests.helpers.families import family


def test_real_maximization_step_prunes_components_without_optimization_blocks() -> None:
    """Continue fitting sparse blocks after real pruning without mutating caller blocks."""
    mixture = MixtureModel(
        [family(FamilyName.EXPONENTIAL)(lambda_=float(index + 1)) for index in range(4)],
        weights=[0.3, 1e-8, 0.3, 0.4 - 1e-8],
    )
    blocks = [OptimizationBlock(index, {"lambda_"}, MaximizationStrategy.QFUNCTION) for index in [0, 3]]
    maximization = MaximizationStep(blocks, ScipyNelderMead())
    iterations = 3
    remaining_components = 3
    pipeline = Pipeline(
        [ExpectationStep(), maximization],
        [StepBreakpointer(iterations)],
        [PriorThresholdPruner(1e-4)],
    )

    fitted = pipeline.fit(np.array([0.5, 1.0, 1.5, 2.0]), mixture)

    assert fitted.n_components == remaining_components
    assert len(pipeline.history) == iterations
    assert all(record.error is None for record in pipeline.history)
    assert all(record.mixture.n_components == remaining_components for record in pipeline.history)
    assert all(record.H is not None and record.H.shape == (4, remaining_components) for record in pipeline.history)
    assert [block.component_id for block in maximization.blocks] == [0, 2]
    assert [block.component_id for block in blocks] == [0, 3]
    assert maximization.blocks[1] is not blocks[1]
    assert cc.all_parameters(fitted[1]) == cc.all_parameters(mixture[2])
