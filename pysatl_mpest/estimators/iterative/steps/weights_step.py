"""Weights update step for iterative estimation pipelines."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np

from ..pipeline_state import PipelineState
from ..pipeline_step import PipelineStep


class WeightsStep(PipelineStep):
    """Mixture weights update step for iterative estimation pipelines.

    Updates mixture weights from the responsibility matrix H computed by
    :class:`ExpectationStep`. It is placed between ExpectationStep and
    MaximizationStep in the pipeline.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        run
    """

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        """list[type[PipelineStep]]: Steps permitted to follow WeightsStep."""

        from pysatl_mpest.estimators.iterative.steps.maximization_step import MaximizationStep

        return [MaximizationStep]

    def run(self, state: PipelineState) -> PipelineState:
        """Update mixture weights from responsibilities and store them in the mixture.

        The state and the current mixture are modified in place. Missing
        responsibilities set ``state.error`` instead of raising.

        Parameters
        ----------
        state : PipelineState
            Current state containing responsibility matrix H and current mixture.

        Returns
        -------
        PipelineState
            Updated state with modified mixture weights.
        """

        if state.H is None:
            state.error = ValueError("Responsibility matrix H is not computed.")
            return state

        responsibilities_sum = np.sum(state.H, axis=0)
        new_weights = responsibilities_sum / state.X.shape[0]
        state.curr_mixture.log_weights = np.log(new_weights + 1e-30)

        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass
