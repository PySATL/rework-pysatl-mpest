"""Dummy class for testing Parameter descriptor."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from pysatl_mpest.core import Parameter
from pysatl_mpest.distributions import ContinuousDistribution
from pysatl_mpest.typings import ArrayLike, FloatArray, FloatingType


class MockParameterOwner[FloatT: FloatingType](ContinuousDistribution[FloatT]):
    """A helper class to test the Parameter descriptor.

    It simulates a class (like a distribution) that uses Parameter instances
    as attributes.

    Parameters
    ----------
    positive_val : float
        A strictly positive parameter.
    any_val : float
        Any parameter without constraints.
    dtype : type[FloatT]
        The numpy dtype.
    """

    positive_param = Parameter(invariant=lambda x: x > 0, error_message="Value must be positive.")
    any_param = Parameter()

    def __init__(self, positive_val: float, any_val: float, dtype: type[FloatT]):
        super().__init__(dtype)

        self.positive_param = positive_val
        self.any_param = any_val

    @property
    def name(self) -> str:
        """Returns the name of the dummy parameter owner.

        Returns
        -------
        str
            The string 'Dummy'.
        """

        return "Dummy"

    @property
    def params(self) -> set[str]:
        """Returns the names of the parameters.

        Returns
        -------
        set[str]
            A set containing 'positive_param' and 'any_param'.
        """

        return {"positive_param", "any_param"}

    def pdf(self, X: ArrayLike) -> FloatT | FloatArray[FloatT]:
        """Dummy PDF implementation returning X.

        Parameters
        ----------
        X : ArrayLike
            Input dataset.

        Returns
        -------
        FloatT | FloatArray[FloatT]
            The dummy PDF evaluation (returns X).
        """

        return X  # type: ignore

    def ppf(self, P: ArrayLike) -> FloatT | FloatArray[FloatT]:
        """Dummy PPF implementation returning P.

        Parameters
        ----------
        P : ArrayLike
            Input probabilities.

        Returns
        -------
        FloatT | FloatArray[FloatT]
            The dummy PPF evaluation (returns P).
        """

        return P  # type: ignore

    def lpdf(self, X: ArrayLike) -> FloatT | FloatArray[FloatT]:
        """Dummy LPDF implementation returning X.

        Parameters
        ----------
        X : ArrayLike
            Input dataset.

        Returns
        -------
        FloatT | FloatArray[FloatT]
            The dummy LPDF evaluation (returns X).
        """

        return X  # type: ignore

    def log_gradients(self, X: ArrayLike) -> FloatArray[FloatT]:
        """Dummy gradient implementation returning zeros.

        Parameters
        ----------
        X : ArrayLike
            Input dataset.

        Returns
        -------
        FloatArray[FloatT]
            Zeros array matching the gradient shape.
        """

        return np.zeros_like(X)

    def generate(self, size: int | tuple[int, ...] | None = None) -> FloatT | FloatArray[FloatT]:
        """Dummy data generator.

        Parameters
        ----------
        size : int | tuple[int, ...], optional
            Output shape, by default None.

        Returns
        -------
        FloatT | FloatArray[FloatT]
            Always 0.0.
        """

        return 0.0  # type: ignore
