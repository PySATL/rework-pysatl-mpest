"""Module providing three-parametric weibull distribution class"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from numpy import float64
from scipy.stats import weibull_min

from rework_pysatl_mpest.core.parameter import Parameter
from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution


class Weibull(ContinuousDistribution):
    """Class for the three-parameter Weibull distribution.

    Parameters
    ----------
    shape : float
        Shape parameter (k). Must be positive.
    loc : float
        Location parameter (gamma). Can be any real number.
    scale : float
        Scale parameter (lambda). Must be positive.

    Attributes
    ----------
    shape : float
        Shape parameter.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

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

    PARAM_SHAPE = "shape"
    PARAM_LOC = "loc"
    PARAM_SCALE = "scale"

    shape = Parameter(lambda x: x > 0, "Shape parameter must be positive")
    loc = Parameter()
    scale = Parameter(lambda x: x > 0, "Scale parameter must be positive")

    def __init__(self, shape: float, loc: float, scale: float):
        super().__init__()
        self.shape = shape
        self.loc = loc
        self.scale = scale

    @property
    def name(self) -> str:
        return "Weibull"

    @property
    def params(self) -> set[str]:
        return {self.PARAM_SHAPE, self.PARAM_LOC, self.PARAM_SCALE}

    def pdf(self, X):
        """Probability density function (PDF).

        The PDF for the three-parameter Weibull distribution is:

        .. math::

            f(x | k, \\lambda, \\gamma) = \\frac{k}{\\lambda}
            \\left( \\frac{x - \\gamma}{\\lambda} \\right)^{k-1}
            e^{-((x - \\gamma) / \\lambda)^k}

        where :math:`k` is the shape, :math:`\\lambda` is the scale, and
        :math:`\\gamma` is the location parameter.
        The function is zero for :math:`x < \\gamma`.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the PDF.

        Returns
        -------
        NDArray[np.float64]
            The PDF values corresponding to each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=float64)
        z = (X - self.loc) / self.scale

        # PDF is 0 for x < loc, and handle cases where z=0 and shape<1
        # which would lead to division by zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            pdf_vals = (self.shape / self.scale) * np.power(z, self.shape - 1) * np.exp(-np.power(z, self.shape))

        return np.where(self.loc <= X, np.nan_to_num(pdf_vals, nan=0.0, posinf=np.inf), 0.0)

    def ppf(self, P):
        """Percent Point Function (PPF) or quantile function.

        The PPF for the three-parameter Weibull distribution is:

        .. math::

            Q(p | k, \\lambda, \\gamma) = \\gamma + \\lambda
            (-\\ln(1 - p))^{1/k}

        Parameters
        ----------
        P : ArrayLike
            The probability values (between 0 and 1) at which to evaluate the PPF.

        Returns
        -------
        NDArray[np.float64]
            The PPF values corresponding to each probability in :attr:`P`.
        """

        P = np.asarray(P, dtype=float64)
        ppf_vals = self.loc + self.scale * np.power(-np.log(1 - P), 1.0 / self.shape)
        return np.where((P >= 0) & (P <= 1), ppf_vals, np.nan)

    def lpdf(self, X):
        """Log of the Probability Density Function (LPDF).

        The log-PDF for the three-parameter Weibull distribution is:

        .. math::

            \\ln f(x) = \\ln(k) - \\ln(\\lambda) + (k - 1)
            (\\ln(x - \\gamma) - \\ln(\\lambda)) -
            \\left(\\frac{x - \\gamma}{\\lambda}\\right)^k

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the LPDF.

        Returns
        -------
        NDArray[np.float64]
            The log-PDF values corresponding to each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=float64)
        z = (X - self.loc) / self.scale
        with np.errstate(divide="ignore"):
            lpdf_vals = np.log(self.shape) - np.log(self.scale) + (self.shape - 1) * np.log(z) - np.power(z, self.shape)
        return np.where(self.loc < X, lpdf_vals, -np.inf)

    def _dlog_shape(self, X):
        """Partial derivative of the lpdf w.r.t. the shape parameter."""

        X = np.asarray(X, dtype=float64)
        z = (X - self.loc) / self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = 1.0 / self.shape + np.log(z) - np.power(z, self.shape) * np.log(z)
        return np.where(self.loc < X, np.nan_to_num(grad), 0.0)

    def _dlog_loc(self, X):
        """Partial derivative of the lpdf w.r.t. the loc parameter."""

        X = np.asarray(X, dtype=float64)
        z = (X - self.loc) / self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = -(self.shape - 1) / (X - self.loc) + (self.shape / self.scale) * np.power(z, self.shape - 1)
        return np.where(self.loc < X, np.nan_to_num(grad), 0.0)

    def _dlog_scale(self, X):
        """Partial derivative of the lpdf w.r.t. the scale parameter."""

        X = np.asarray(X, dtype=float64)
        z = (X - self.loc) / self.scale
        grad = -self.shape / self.scale + (self.shape / self.scale) * np.power(z, self.shape)
        return np.where(self.loc < X, grad, 0.0)

    def log_gradients(self, X):
        """Calculates the gradients of the log-PDF w.r.t. its parameters.

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
        X = np.asarray(X, dtype=float64)

        gradient_calculators = {
            self.PARAM_SHAPE: self._dlog_shape,
            self.PARAM_LOC: self._dlog_loc,
            self.PARAM_SCALE: self._dlog_scale,
        }

        optimizable_params = sorted(list(self.params_to_optimize))

        if not optimizable_params:
            return np.empty((len(X), 0))

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

        return np.asarray(weibull_min.rvs(c=self.shape, loc=self.loc, scale=self.scale, size=size), dtype=float64)

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Weibull(shape=2.0, loc=0.0, scale=1.0)".
        """

        return f"{self.__class__.__name__}(shape={self.shape}, loc={self.loc}, scale={self.scale})"
