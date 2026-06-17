"""Common utilities for maximization strategies."""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from ....exceptions import NumericalStabilityError
from ....typings import FloatingType
from ..pipeline_state import PipelineState


def handle_numerical_overflow[FloatT: FloatingType](
    state: PipelineState[FloatT], context: str = "optimization"
) -> None:
    """Creates and registers a numerical stability error in the pipeline state.

    This helper function is called when a numerical instability (e.g., infinity)
    is detected during the M-step. It creates a `NumericalStabilityError`,
    places it in `state.error`.

    The presence of this error in the state signals the `Pipeline` class to
    take corrective action, such as restarting the fitting process with a
    higher floating-point precision (e.g., `np.float64`).

    Parameters
    ----------
    state : PipelineState[FloatT]
        The current pipeline state where the error will be recorded.
    context : str
        The context name to include in the error message (e.g. 'Moments optimization', 'Q-function optimization').
    """
    error = NumericalStabilityError(
        f"Overflow detected during {context}. The pipeline will attempt to restart with higher precision."
    )
    state.error = error
