"""Dummy distribution for testing purposes."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from pysatl_mpest.core import Parameter
from pysatl_mpest.distributions import ContinuousDistribution
from pysatl_mpest.typings import ArrayLike, FloatArray, FloatingType


class MockContinuousDistribution[FloatT: FloatingType](ContinuousDistribution[FloatT]):
    """A concrete implementation of ContinuousDistribution for testing purposes.

    This class implements all abstract methods, allowing to instantiate it
    and test the non-abstract methods of the base class.

    Parameters
    ----------
    param1 : float, optional
        First parameter, by default 1.0
    param2 : float, optional
        Second parameter, by default 2.0
    name : str, optional
        Name of the distribution, by default "Dummy"
    dtype : type[FloatT], optional
        Floating point precision type, by default np.float64
    """

    param1 = Parameter()
    param2 = Parameter()

    def __init__(self, param1: float = 1.0, param2: float = 2.0, name: str = "Dummy", dtype: type[FloatT] = np.float64):  # type: ignore[assignment]
        super().__init__(dtype=dtype)
        self.param1 = param1
        self.param2 = param2
        self._name = name

    @property
    def name(self) -> str:
        """Returns the name of the distribution.

        Returns
        -------
        str
            The internal name.
        """

        return self._name

    @property
    def params(self) -> set[str]:
        """Returns the names of the parameters.

        Returns
        -------
        set[str]
            A set containing 'param1' and 'param2'.
        """

        return {"param1", "param2"}

    def pdf(self, X: ArrayLike) -> FloatT | FloatArray[FloatT]:
        """Predictable PDF returning 1.0 for all elements.

        Parameters
        ----------
        X : ArrayLike
            Input dataset.

        Returns
        -------
        FloatT | FloatArray[FloatT]
            1.0 for each element.
        """

        X_arr = np.asarray(X, dtype=self.dtype)
        if X_arr.ndim == 0:
            return self.dtype(1.0)
        return np.ones_like(X_arr, dtype=self.dtype)

    def ppf(self, P: ArrayLike) -> FloatT | FloatArray[FloatT]:
        """Predictable PPF returning 1.0 for all elements.

        Parameters
        ----------
        P : ArrayLike
            Input probabilities.

        Returns
        -------
        FloatT | FloatArray[FloatT]
            1.0 for each element.
        """

        P_arr = np.asarray(P, dtype=self.dtype)
        if P_arr.ndim == 0:
            return self.dtype(1.0)
        return np.ones_like(P_arr, dtype=self.dtype)

    def lpdf(self, X: ArrayLike) -> FloatT | FloatArray[FloatT]:
        """Predictable LPDF returning log(1 + X).

        A predictable lpdf is needed to test q_function.
        Returns log1p(X) for simplicity and handles X=0 to avoid -inf.

        Parameters
        ----------
        X : ArrayLike
            Input dataset.

        Returns
        -------
        FloatT | FloatArray[FloatT]
            The log-probability values.
        """

        X_arr = np.asarray(X, dtype=self.dtype)
        res = np.log1p(X_arr)
        if res.ndim == 0:
            return self.dtype(res)
        return res.astype(self.dtype)

    def log_gradients(self, X: ArrayLike) -> FloatArray[FloatT]:
        """Predictable gradients returning zeros.

        Parameters
        ----------
        X : ArrayLike
            Input dataset.

        Returns
        -------
        FloatArray[FloatT]
            An array of zeros matching the shape of (len(X), num_params).
        """

        X_arr = np.atleast_1d(X)
        num_params = len(self.params_to_optimize)
        return np.zeros((len(X_arr), num_params), dtype=self.dtype)

    def generate(self, size: int | tuple[int, ...] | None = None) -> FloatT | FloatArray[FloatT]:
        """Predictable generator returning sequential numbers.

        Parameters
        ----------
        size : int | tuple[int, ...], optional
            Output shape, by default None.

        Returns
        -------
        FloatT | FloatArray[FloatT]
            Sequential values up to the requested size.
        """

        if size is None:
            return self.dtype(0.0)

        if isinstance(size, tuple):
            n = int(np.prod(size))
            return np.arange(n, dtype=self.dtype).reshape(size)
        return np.arange(size, dtype=self.dtype)


class MockInfLpdfContinuousDistribution[FloatT: FloatingType](MockContinuousDistribution[FloatT]):
    """A dummy distribution that returns -inf for negative inputs."""

    def lpdf(self, X: ArrayLike) -> FloatT | FloatArray[FloatT]:
        """Predictable LPDF returning -inf if X < 0, else log1p(X).

        Parameters
        ----------
        X : ArrayLike
            Input dataset.

        Returns
        -------
        FloatT | FloatArray[FloatT]
            The log-probability values, with -inf where X < 0.
        """

        X_arr = np.asarray(X, dtype=self.dtype)
        res = np.where(X_arr < 0, -np.inf, np.log1p(X_arr))
        if res.ndim == 0:
            return self.dtype(res)
        return res.astype(self.dtype)
