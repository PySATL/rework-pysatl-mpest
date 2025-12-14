__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from typing import Literal

from numpy.typing import ArrayLike

from rework_pysatl_mpest.core.mixture import MixtureModel
from rework_pysatl_mpest.estimators.iterative._logger import IterationsHistory
from rework_pysatl_mpest.estimators.iterative.pipeline import Pipeline
from rework_pysatl_mpest.estimators.iterative.steps.block import MaximizationStrategy, OptimizationBlock
from rework_pysatl_mpest.estimators.iterative.steps.expectation_step import ExpectationStep
from rework_pysatl_mpest.estimators.iterative.steps.maximization_step import MaximizationStep
from rework_pysatl_mpest.optimizers.optimizer import Optimizer

from ..typings import DType
from .base_estimator import BaseEstimator
from .iterative import (
    Breakpointer,
    Pruner,
)


class ECME(BaseEstimator[DType]):
    def __init__(
        self,
        breakpointers: Sequence[Breakpointer[DType]],
        pruners: Sequence[Pruner[DType]],
        optimizer: Optimizer[DType],
        default_strategy: Literal["q-func", "odl"] = "odl",
    ):
        self.breakpointers = list(breakpointers)
        self.pruners = list(pruners)
        self.optimizer = optimizer

        if default_strategy not in ("odl", "q-func"):
            raise ValueError(f"Unknown default_strategy '{default_strategy}'. Expected 'odl' or 'q-func'.")

        self.default_strategy_name = default_strategy

        self._history: IterationsHistory[DType] | None = None

    @property
    def history(self) -> IterationsHistory[DType]:
        if self._history is None:
            raise AttributeError("History is not available. Call the 'fit' method first.")
        return self._history

    def _normalize_indices(self, indices: Sequence[int] | int | None) -> set[int]:
        if indices is None:
            return set()
        if isinstance(indices, int):
            return {indices}
        return set(indices)

    def _resolve_index(self, idx: int, n_components: int) -> int:
        original_idx = idx
        if idx < 0:
            idx += n_components

        if not (0 <= idx < n_components):
            raise ValueError(f"Component index {original_idx} is out of bounds for mixture size {n_components}.")
        return idx

    def fit(
        self,
        X: ArrayLike,
        mixture: MixtureModel[DType],
        q_indices_raw: Sequence[int] | int | None = None,
        odl_indices_raw: Sequence[int] | int | None = None,
        once_in_iterations: int = 1,
    ) -> MixtureModel[DType]:
        n_components = mixture.n_components

        # 1. Resolve indices relative to the provided mixture
        q_indices = {self._resolve_index(i, n_components) for i in self._normalize_indices(q_indices_raw)}
        odl_indices = {self._resolve_index(i, n_components) for i in self._normalize_indices(odl_indices_raw)}

        # 2. Validation
        intersection = q_indices.intersection(odl_indices)
        if intersection:
            raise ValueError(f"Indices {intersection} specified for both Q-function and ODL strategies.")

        # 3. Build optimization blocks
        blocks = []

        # Map string config to Enum
        default_enum = (
            MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD
            if self.default_strategy_name == "odl"
            else MaximizationStrategy.QFUNCTION
        )

        for i, comp in enumerate(mixture):
            if i in q_indices:
                strategy = MaximizationStrategy.QFUNCTION
            elif i in odl_indices:
                strategy = MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD
            else:
                strategy = default_enum

            blocks.append(
                OptimizationBlock(
                    component_id=i, params_to_optimize=comp.params_to_optimize, maximization_strategy=strategy
                )
            )

        pipeline: Pipeline[DType] = Pipeline(
            [ExpectationStep(), MaximizationStep(blocks, self.optimizer)],
            self.breakpointers,
            self.pruners,
            once_in_iterations,
        )

        return pipeline.fit(X, mixture)
