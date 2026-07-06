"""Dummy class for testing Parameter descriptor."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from pysatl_mpest.core import Parameter
from pysatl_mpest.distributions import ContinuousDistribution
from pysatl_mpest.typings import ArrayLike, FloatArray


class MockParameterOwner(ContinuousDistribution):
    """A helper class to test the Parameter descriptor.

    It simulates a class (like a distribution) that uses Parameter instances
    as attributes.

    Parameters
    ----------
    positive_val : float
        A strictly positive parameter.
    any_val : float
        Any parameter without constraints.
    """

    positive_param = Parameter(invariant=lambda x: x > 0, error_message="Value must be positive.")
    any_param = Parameter()

    def __init__(self, positive_val: float, any_val: float):
        super().__init__()

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

    def clone_with_params(self, param_names: list[str], vector: list[float]) -> "MockParameterOwner":
        new_dist = MockParameterOwner(self.param1, self.param2)
        for name, value in zip(param_names, vector):
            setattr(new_dist, name, float(value))
        return new_dist

    def pdf(self, X: ArrayLike) -> np.float64 | FloatArray:
        """Dummy PDF implementation returning X.

        Parameters
        ----------
        X : ArrayLike
            Input dataset.

        Returns
        -------
        np.float64 | FloatArray
            The dummy PDF evaluation (returns X).
        """

        return X  # type: ignore

    def ppf(self, P: ArrayLike) -> np.float64 | FloatArray:
        """Dummy PPF implementation returning P.

        Parameters
        ----------
        P : ArrayLike
            Input probabilities.

        Returns
        -------
        np.float64 | FloatArray
            The dummy PPF evaluation (returns P).
        """

        return P  # type: ignore

    def lpdf(self, X: ArrayLike) -> np.float64 | FloatArray:
        """Dummy LPDF implementation returning X.

        Parameters
        ----------
        X : ArrayLike
            Input dataset.

        Returns
        -------
        np.float64 | FloatArray
            The dummy LPDF evaluation (returns X).
        """

        return X  # type: ignore

    def log_gradients(self, X: ArrayLike) -> FloatArray:
        """Dummy gradient implementation returning zeros.

        Parameters
        ----------
        X : ArrayLike
            Input dataset.

        Returns
        -------
        FloatArray
            Zeros array matching the gradient shape.
        """

        return np.zeros_like(X)

    def generate(self, size: int | tuple[int, ...] | None = None) -> np.float64 | FloatArray:
        """Dummy data generator.

        Parameters
        ----------
        size : int | tuple[int, ...], optional
            Output shape, by default None.

        Returns
        -------
        np.float64 | FloatArray
            Always 0.0.
        """

        return 0.0  # type: ignore
