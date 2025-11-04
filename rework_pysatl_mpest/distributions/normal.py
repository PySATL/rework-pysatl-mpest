"""Module providing normal (Gaussian) distribution class"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from scipy.stats import norm

from ..core import Parameter
from ..typings import DType
from .continuous_dist import ContinuousDistribution


class Normal(ContinuousDistribution[DType]):
    """Class for the Normal (Gaussian) distribution.

    Parameters
    ----------
    loc : float
        Mean of the distribution (mu). Can be any real number.
    scale : float
        Standard deviation of the distribution (sigma). Must be positive.

    Attributes
    ----------
    loc : float
        Mean of the distribution.
    scale : float
        Standard deviation of the distribution.


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
    PARAM_SCALE = "scale"

    loc = Parameter()
    scale = Parameter(lambda x: x > 0, "Scale parameter must be positive")

    def __init__(self, loc: float, scale: float, dtype: type[DType] = np.float64):  # type: ignore[assignment]
        super().__init__(dtype=dtype)
        self.loc = loc
        self.scale = scale

    @property
    def name(self) -> str:
        return "Normal"

    @property
    def params(self) -> set[str]:
        return {self.PARAM_LOC, self.PARAM_SCALE}

    def pdf(self, X):
        """Probability density function (PDF).

        The PDF for the Normal distribution is:

        .. math::

            f(x | \\mu, \\sigma) = \\frac{1}{\\sigma \\sqrt{2\\pi}}
            \\exp\\left( -\\frac{(x - \\mu)^2}{2\\sigma^2} \\right)

        where :math:`\\mu` is the mean (loc) and :math:`\\sigma` is the
        standard deviation (scale).

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the PDF.

        Returns
        -------
        NDArray[DType]
            The PDF values corresponding to each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        z = (X - self.loc) / self.scale
        return np.exp(-(z**2) / dtype(2.0)) / (self.scale * np.sqrt(dtype(2.0) * dtype(np.pi)))

    def ppf(self, P):
        """Percent Point Function (PPF) or quantile function.

        The PPF is the inverse of the Cumulative Distribution Function (CDF).
        This implementation relies on `scipy.stats.norm.ppf` for accuracy
        and robustness.

        Parameters
        ----------
        P : ArrayLike
            The probability values (between 0 and 1) at which to evaluate the PPF.

        Returns
        -------
        NDArray[DType]
            The PPF values corresponding to each probability in :attr:`P`.
        """

        P = np.asarray(P, dtype=self.dtype)
        result = norm.ppf(P, loc=self.loc, scale=self.scale)

        return np.asarray(result, dtype=self.dtype)

    def lpdf(self, X):
        """Log of the Probability Density Function (LPDF).

        The log-PDF for the Normal distribution is:

        .. math::

            \\ln f(x) = -\\ln(\\sigma) - \\frac{1}{2} \\ln(2\\pi) -
            \\frac{(x - \\mu)^2}{2\\sigma^2}

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the LPDF.

        Returns
        -------
        NDArray[DType]
            The log-PDF values corresponding to each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        z = (X - self.loc) / self.scale
        return -np.log(self.scale) - dtype(0.5) * np.log(dtype(2.0) * dtype(np.pi)) - dtype(0.5) * z**2

    def _dlog_loc(self, X):
        """Partial derivative of the lpdf w.r.t. the loc parameter."""

        X = np.asarray(X, dtype=self.dtype)
        return (X - self.loc) / (self.scale**2)

    def _dlog_scale(self, X):
        """Partial derivative of the lpdf w.r.t. the scale parameter."""

        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        z_sq = ((X - self.loc) / self.scale) ** 2
        return (z_sq - dtype(1.0)) / self.scale

    def log_gradients(self, X):
        """Calculates the gradients of the log-PDF w.r.t. its parameters.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to calculate the gradients.

        Returns
        -------
        NDArray[DType]
            An array where each row corresponds to a data point in :attr:`X`
            and each column corresponds to the gradient with respect to a
            specific optimizable parameter. The order of columns corresponds
            to the sorted order of :attr:`self.params_to_optimize`.
        """

        X = np.asarray(X, dtype=self.dtype)

        gradient_calculators = {
            self.PARAM_LOC: self._dlog_loc,
            self.PARAM_SCALE: self._dlog_scale,
        }

        optimizable_params = sorted(list(self.params_to_optimize))

        if not optimizable_params:
            return np.empty((len(X), 0), dtype=self.dtype)

        gradients = [gradient_calculators[param](X) for param in optimizable_params]
        return np.stack(gradients, axis=1)

    def generate(self, size: int):
        """Generates random samples from the distribution.

        This implementation relies on `scipy.stats.norm.rvs`.

        Parameters
        ----------
        size : int
            The number of random samples to generate.

        Returns
        -------
        NDArray[DType]
            A NumPy array containing the generated samples.
        """

        return np.asarray(norm.rvs(loc=self.loc, scale=self.scale, size=size), dtype=self.dtype)

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Normal(loc=0.0, scale=1.0)".
        """

        return f"{self.__class__.__name__}(loc={self.loc}, scale={self.scale})"
