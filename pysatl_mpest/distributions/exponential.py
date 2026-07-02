"""Module providing exponential distribution class"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from scipy.stats import expon

from ..core import Parameter
from .continuous_dist import ContinuousDistribution


class Exponential(ContinuousDistribution):
    """Class for the two-parameter exponential distribution.

    Parameters
    ----------
    loc : float
        Location parameter. Can be any real number.
    rate : float
        Rate parameter (lambda). Must be positive.

    Attributes
    ----------
    loc : float
        Location parameter.
    rate : float
        Rate parameter.


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

    PARAM_LOC = "loc"
    PARAM_RATE = "rate"

    loc = Parameter()
    rate = Parameter(lambda x: x > 0, "Rate parameter must be a positive")

    def __init__(self, loc: float, rate: float):
        super().__init__()
        self.loc = loc
        self.rate = rate

    @property
    def name(self) -> str:
        return "Exponential"

    @property
    def params(self) -> set[str]:
        return {self.PARAM_LOC, self.PARAM_RATE}

    def pdf(self, X):
        """Probability density function (PDF).

        The PDF for the two-parameter exponential distribution is:

        .. math::

            f(x | \\alpha, \\beta) = \\alpha \\cdot e^{-\\alpha \\cdot (x - \\beta)}

        where :math:`\\alpha` is the rate parameter and :math:`\\beta` is the
        location parameter. The function is zero for :math:`x < \\beta`.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the PDF.

        Returns
        -------
        np.float64 | FloatArray
            The PDF values corresponding to each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=np.float64)
        return np.exp(self.lpdf(X))

    def ppf(self, P):
        """Percent Point Function (PPF) or quantile function.

        The PPF for the two-parameter exponential distribution is:

        .. math::

            Q(p | \\alpha, \\beta) = \\beta - \\frac{\\ln(1 - p)}{\\alpha}

        Parameters
        ----------
        P : ArrayLike
            The probability values (between 0 and 1) at which to evaluate the PPF.

        Returns
        -------
        np.float64 | FloatArray
            The PPF values corresponding to each probability in :attr:`P`.
            Return a scalar when given a scalar, and to return an array when given an array.
        """

        is_scalar = np.isscalar(P)
        P = np.asarray(P, dtype=np.float64)

        result = np.where((P >= 0) & (P <= 1), self.loc - np.log(np.float64(1) - P) / self.rate, np.float64(np.nan))
        if is_scalar:
            return result[()]
        return result

    def lpdf(self, X):
        """Log of the Probability Density Function (LPDF).

        The log-PDF for the two-parameter exponential distribution is:

        .. math::

            \\ln f(x | \\alpha, \\beta) = \\ln(\\alpha) - \\alpha \\cdot (x - \\beta)

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the LPDF.

        Returns
        -------
        np.float64 | FloatArray
            The log-PDF values corresponding to each point in :attr:`X`.
            Return a scalar when given a scalar, and to return an array when given an array.
        """

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=np.float64)

        result = np.where(self.loc <= X, np.log(self.rate) - self.rate * (X - self.loc), np.float64(-np.inf))
        if is_scalar:
            return result[()]
        return result

    def _dlog_loc(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`loc` parameter.

        The derivative is non-zero only for `X >= loc`.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta)}{\\partial \\beta} = \\alpha

        where :math:`\\alpha` is the rate and :math:`\\beta` is the location.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        np.float64 | FloatArray
            The gradient of the lpdf with respect to :attr:`loc` for each point in :attr:`X`.
            Return a scalar when given a scalar, and to return an array when given an array.
        """

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=np.float64)

        result = np.where(self.loc <= X, self.rate, np.float64(0.0))
        if is_scalar:
            return result[()]
        return result

    def _dlog_rate(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`rate` parameter.

        The derivative is non-zero only for `X >= loc`.

        .. math::

             \\frac{\\partial \\ln f(x | \\alpha, \\beta)}{\\partial \\alpha} = \\frac{1}{\\alpha} - (x - \\beta)

        where :math:`\\alpha` is the rate and :math:`\\beta` is the location.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        np.float64 | FloatArray
            The gradient of the lpdf with respect to :attr:`rate` for each point in :attr:`X`.
            Return a scalar when given a scalar, and to return an array when given an array.
        """

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=np.float64)

        result = np.where(self.loc <= X, np.float64(1.0) / self.rate - (X - self.loc), np.float64(0.0))
        if is_scalar:
            return result[()]
        return result

    def log_gradients(self, X):
        """Calculates the gradients of the log-PDF w.r.t. its parameters.

        The gradients are computed for the parameters that are not fixed.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to calculate the gradients.

        Returns
        -------
        FloatArray
            An array where each row corresponds to a data point in :attr:`X`
            and each column corresponds to the gradient with respect to a
            specific optimizable parameter. The order of columns corresponds
            to the sorted order of :attr:`self.params_to_optimize`.
            Returns a 1D array if X is a scalar.
        """

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=np.float64)

        gradient_calculators = {
            self.PARAM_LOC: self._dlog_loc,
            self.PARAM_RATE: self._dlog_rate,
        }

        optimizable_params = sorted(list(self.params_to_optimize))

        if not optimizable_params:
            return np.empty((len(X), 0), dtype=np.float64)

        gradients = [gradient_calculators[param](X) for param in optimizable_params]

        if is_scalar:
            return np.array(gradients)
        return np.stack(gradients, axis=1)

    def generate(self, size: int | tuple[int, ...] | None = None):
        """Generates random samples from the distribution.

        Parameters
        ----------
        size : int | tuple[int, ...] | None, optional
            Defining number of random variates.
            - If None (default), returns a single scalar.
            - If int, returns a 1D array of that length.
            - If tuple, returns an array of that shape.

        Returns
        -------
        np.float64 | FloatArray
            A scalar or NumPy array containing the generated samples.
        """

        samples = expon.rvs(loc=self.loc, scale=1 / self.rate, size=size)

        if size is None:
            return np.float64(samples)
        return np.asarray(samples, dtype=np.float64)

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Exponential(loc=0.0, rate=2.0)".
        """

        return f"{self.__class__.__name__}(loc={self.loc}, rate={self.rate})"
