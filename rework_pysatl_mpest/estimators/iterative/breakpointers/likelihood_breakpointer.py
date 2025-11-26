"""Module that provides a :class:`rework_pysatl_mpest.estimators.iterative.Pipeline`
stopping strategy based on when log-likelihood of the mixture converges"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Optional

from ..breakpointer import Breakpointer
from ..pipeline_state import PipelineState


class LikelihoodBreakpointer(Breakpointer):
    """Stops the pipeline when the log-likelihood of the mixture converges.

    This breakpointer terminates the iterative estimation process when the
    absolute difference between the current and previous log-likelihood values
    falls below a specified threshold:

        |L_{t+1} - L_t| < threshold

    It tracks the log-likelihood of the current mixture model on the observed
    data at each iteration and compares it to the previous value.

    Parameters
    ----------
    threshold : float
        The convergence threshold for the log-likelihood difference.
        Must be a positive number.

    Attributes
    ----------
    threshold : float
        The convergence threshold.

    Raises
    ------
    ValueError
        If `threshold` is not greater than 0.

    Methods
    -------
    check(state: PipelineState) -> bool
        Returns True if convergence is detected, False otherwise.
    """

    def __init__(self, threshold: float):
        self._validate(threshold)
        self.threshold = threshold
        self._L_old: Optional[float] = None
        self._L_new: Optional[float] = None

    def _validate(self, threshold: float):
        """Validates the threshold parameter."""
        if threshold <= 0:
            raise ValueError("The threshold must be greater than 0")

    def check(self, state: PipelineState) -> bool:
        """Checks if the log-likelihood has converged.

        Computes the current log-likelihood of the mixture on the data in
        the pipeline state and compares it with the previous value.

        Parameters
        ----------
        state : PipelineState
            The current state of the pipeline, which must contain a valid
            `curr_mixture` and data `X`.

        Returns
        -------
        bool
            True if |L_new - L_old| < threshold (converged), False otherwise.
            On the first call (no previous likelihood), returns False and
            initializes internal state.
        """
        self._L_new = state.curr_mixture.loglikelihood(state.X)

        # First iteration: cannot compare, so just store and continue
        if self._L_old is None:
            self._L_old = self._L_new
            return False

        if abs(self._L_new - self._L_old) < self.threshold:
            self._L_old = None
            self._L_new = None
            return True
        else:
            self._L_old = self._L_new
            return False
