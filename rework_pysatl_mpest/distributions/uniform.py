"""Module providing uniform distribution class"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from scipy.stats import uniform

from ..core import Parameter
from ..typings import DType
from .continuous_dist import ContinuousDistribution


class Uniform(ContinuousDistribution[DType]):
    """
    The Uniform continuous probability distribution.

    The uniform distribution describes an experiment where there is an arbitrary
    outcome that lies between certain bounds. The probability is constant between
    these bounds and zero elsewhere.

    Parameters
    ----------
    left_border : float
        Left border of section [a, b]. Can be any real number.
    right_border : float
        Right border of section [a, b]. Can be any real number.

    Attributes
    ----------
    left_border : float
        Left border of section [a, b].
    right_border : float
        Right border of section [a, b].

    Raises
    ------
    ValueError
        If left_border is greater than or equal to right_border, or if either
        parameter is not finite.

    Methods
    -------

    .. autosummary::
        :toctree: generated/

        ppf
        pdf
        lpdf
        log_gradients
        generate
    """

    LEFT_BORDER = "left_border"
    RIGHT_BORDER = "right_border"

    left_border = Parameter()
    right_border = Parameter()

    def __init__(self, left_border: float, right_border: float, dtype: type[DType] = np.float64):  # type: ignore[assignment]
        super().__init__(dtype=dtype)
        if left_border >= right_border:
            raise ValueError("right_border parameter must be strictly greater than left_border")
        if not (np.isfinite(left_border) and np.isfinite(right_border)):
            raise ValueError("Both borders should be finite values")

        self.left_border = left_border
        self.right_border = right_border

    @property
    def name(self) -> str:
        return "Uniform"

    @property
    def params(self) -> set[str]:
        return {self.LEFT_BORDER, self.RIGHT_BORDER}

    def pdf(self, X):
        """Probability density function (PDF).

        The PDF for the unifrom distribution is:

        .. math::

            f(x | \\alpha, \\beta) = frac{1}{\\beta - \\alpha}

        where :math:`\\alpha` is the left_border parameter and :math:`\\beta` is the
        right_border parameter. The function is zero for :math:`x < \\alpha` or :math:`x > \\beta`.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the PDF.

        Returns
        -------
        NDArray[np.float64]
            The PDF values corresponding to each point in :attr:`X`.
        """
        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        return np.where(
            (self.left_border <= X) & (self.right_border >= X),
            dtype(1.0) / (self.right_border - self.left_border),
            dtype(0.0),
        )

    def ppf(self, P):
        """Percent Point Function (PPF) or quantile function.

        The PPF for the uniform distribution is:

        .. math::

            Q(p | \\alpha, \\beta) = \\alpha + p \\cdot (\\beta - \\alpha)

        where :math:`\\alpha` is the left_border parameter and :math:`\\beta` is the
        right_border parameter.

        Parameters
        ----------
        P : ArrayLike
            The probability values (between 0 and 1) at which to evaluate the PPF.

        Returns
        -------
        NDArray[np.float64]
            The PPF values corresponding to each probability in :attr:`P`.
        """
        P = np.asarray(P, dtype=self.dtype)
        dtype = self.dtype

        return np.where(
            (P >= 0) & (P <= 1), self.left_border + P * (self.right_border - self.left_border), dtype(np.nan)
        )

    def lpdf(self, X):
        """Log of the Probability Density Function (LPDF).

        The log-PDF for the uniform distribution is:

        .. math::

            \\ln f(x | \\alpha, \\beta) = -\\ln(\\beta - \\alpha)

        where :math:`\\alpha` is the left_border parameter and :math:`\\beta` is the
        right_border parameter. The function is -inf for :math:`\\alpha >= \\beta` or
        when x not in range.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the LPDF.

        Returns
        -------
        NDArray[np.float64]
            The log-PDF values corresponding to each point in :attr:`X`.
        """
        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        in_range = (self.left_border <= X) & (self.right_border >= X)
        valid_dist = self.right_border > self.left_border
        return np.where(in_range & valid_dist, -np.log(self.right_border - self.left_border), dtype(-np.inf))

    def _dlog_left_border(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`left_border` parameter.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta)}{\\partial \\alpha} = frac{1.0}{(\\beta - \\alpha)}

        where :math:`\\alpha` is the left_border parameter and :math:`\\beta` is the
        right_border parameter. The derivative is non-zero only for `left_border <= X <= right_border`.
        """

        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        in_range = (self.left_border <= X) & (self.right_border >= X)
        return np.where(in_range, dtype(1.0) / (self.right_border - self.left_border), dtype(0.0))

    def _dlog_right_border(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`right_border` parameter.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta)}{\\partial \\beta} = frac{-1.0}{(\\beta - \\alpha)}

        where :math:`\\alpha` is the left_border parameter and :math:`\\beta` is the
        right_border parameter. The derivative is non-zero only for `left_border <= X <= right_border`.
        """

        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        in_range = (self.left_border <= X) & (self.right_border >= X)
        return np.where(in_range, dtype(-1.0) / (self.right_border - self.left_border), dtype(0.0))

    def log_gradients(self, X):
        """Calculates the gradients of the log-PDF w.r.t. its parameters.

        The gradients are computed for the parameters that are not fixed.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to calculate the gradients.

        Returns
        -------
        NDArray[np.float64]
            An array where each row corresponds to a data point in :attr:`X`
            and each column corresponds to the gradient with respect to a
            specific optimizable parameter. The order of columns corresponds
            to the sorted order of :attr:`self.params_to_optimize`.
        """

        X = np.asarray(X, dtype=self.dtype)

        gradient_calculators = {
            self.LEFT_BORDER: self._dlog_left_border,
            self.RIGHT_BORDER: self._dlog_right_border,
        }

        optimizable_params = sorted(list(self.params_to_optimize))

        if not optimizable_params:
            return np.empty((len(X), 0), dtype=self.dtype)

        gradients = [gradient_calculators[param](X) for param in optimizable_params]

        return np.stack(gradients, axis=1)

    def generate(self, size: int):
        """Generates random samples from the distribution.

        Parameters
        ----------
        size : int
            The number of random samples to generate.

        Returns
        -------
        NDArray[np.float64]
            A NumPy array containing the generated samples.
        """

        return np.asarray(
            uniform.rvs(loc=self.left_border, scale=self.right_border - self.left_border, size=size), dtype=self.dtype
        )

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Uniform(left_border=0.0, right_border=2.0)".
        """

        return f"{self.__class__.__name__}(left_border={self.left_border}, right_border={self.right_border})"
