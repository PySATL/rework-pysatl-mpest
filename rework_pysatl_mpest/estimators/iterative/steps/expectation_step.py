"""Provides the Expectation-step for an iterative estimation pipeline."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from scipy.special import logsumexp

from ....typings import DType
from ..pipeline_state import PipelineState
from ..pipeline_step import PipelineStep


class ExpectationStep(PipelineStep[DType]):
    """A pipeline step that performs the Expectation (E-step).

    This step calculates the responsibility matrix H, where H[i, j] is
    the posterior probability that the i-th data point belongs to the j-th
    mixture component. It can perform either a soft (probabilistic) or hard
    (winner-takes-all) assignment.

    Parameters
    ----------
    is_soft : bool, optional
        If True (default), performs a soft assignment where H contains probabilities.
        If False, performs a hard assignment where each data point
        is assigned to the single most likely component (i.e., H contains
        only 0s and 1s).

    Attributes
    ----------
    is_soft : bool
        Flag indicating whether to perform soft or hard assignment.

    Methods
    -------

    .. autosummary::
        :toctree: generated/

        run
    """

    def __init__(self, is_soft: bool = True):
        self.is_soft = is_soft

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        """list[type[PipelineStep]]: Defines the valid subsequent steps.

        Specifies that an :class:`ExpectationStep` must be followed by a
        :class:`MaximizationStep` to form a standard EM iteration.
        """

        from rework_pysatl_mpest.estimators.iterative.steps.maximization_step import MaximizationStep

        return [MaximizationStep]

    def run(self, state: PipelineState[DType]) -> PipelineState[DType]:
        """Executes the E-step by calculating the responsibility matrix H.

        This method computes the log-likelihood of each data point under each
        component, incorporates the component weights, and normalizes to find
        the posterior probabilities (responsibilities). The resulting matrix H
        is then stored in the pipeline state.

        Parameters
        ----------
        state : PipelineState[DType]
            The current state of the pipeline, which must contain the input
            data X and the current mixture model curr_mixture.

        Returns
        -------
        PipelineState[DType]
            The updated pipeline state with the H attribute computed and set.
        """

        X, mixture = state.X, state.curr_mixture

        dtype = mixture.dtype

        log_p_xij_matrix = np.array([comp.lpdf(X) for comp in mixture.components])
        log_p_xij_matrix = log_p_xij_matrix.T

        log_weighted_likelihoods = log_p_xij_matrix + mixture.log_weights
        log_denominator = logsumexp(log_weighted_likelihoods, axis=1, keepdims=True)

        valid_mask = log_denominator != dtype(-np.inf)
        H_soft = np.zeros_like(log_weighted_likelihoods, dtype=dtype)
        log_H = log_weighted_likelihoods[valid_mask.flatten(), :] - log_denominator[valid_mask.flatten(), :]

        H_soft[valid_mask.flatten(), :] = np.exp(log_H)
        H_soft[np.isnan(H_soft)] = dtype(0.0)

        if not self.is_soft:
            n_samples = X.shape[0]
            H_hard = np.zeros_like(H_soft, dtype=dtype)

            max_indices = np.argmax(H_soft, axis=1)
            H_hard[np.arange(n_samples), max_indices] = dtype(1.0)

            state.H = H_hard
        else:
            state.H = H_soft

        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass
