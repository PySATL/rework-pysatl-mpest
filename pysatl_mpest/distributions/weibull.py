"""Module providing three-parametric weibull distribution class"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from scipy.stats import weibull_min

from ..core import Parameter
from ..typings import DType
from .continuous_dist import ContinuousDistribution


class Weibull(ContinuousDistribution[DType]):
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

    def __init__(self, shape: float, loc: float, scale: float, dtype: type[DType] = np.float64):  # type: ignore[assignment]
        super().__init__(dtype=dtype)
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
        DType | NDArray[DType]
            The PDF values corresponding to each point in :attr:`X`.
            Return a scalar when given a scalar, and to return an array when given an array.
        """

        X = np.asarray(X, dtype=self.dtype)
        return np.exp(self.lpdf(X))

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
        DType | NDArray[DType]
            The PPF values corresponding to each probability in :attr:`P`.
            Return a scalar when given a scalar, and to return an array when given an array.
        """

        is_scalar = np.isscalar(P)
        P = np.asarray(P, dtype=self.dtype)
        dtype = self.dtype

        ppf_vals = self.loc + self.scale * np.power(-np.log(dtype(1) - P), dtype(1.0) / self.shape)
        result = np.where((P >= 0) & (P <= 1), ppf_vals, dtype(np.nan))

        if is_scalar:
            return result[()]
        return result

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
        DType | NDArray[DType]
            The log-PDF values corresponding to each point in :attr:`X`.
            Return a scalar when given a scalar, and to return an array when given an array.
        """

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        z = (X - self.loc) / self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            # Handle potential NaN from 0 * -inf when shape=1 and z=0
            # This term's limit is 0, so we replace NaN with 0.
            lpdf_vals = (
                np.log(self.shape)
                - np.log(self.scale)
                + np.nan_to_num((self.shape - dtype(1)) * np.log(z), nan=dtype(0.0))
                - np.power(z, self.shape)
            )
        result = np.where(self.loc < X, lpdf_vals, dtype(-np.inf))

        if is_scalar:
            return result[()]
        return result

    def _dlog_shape(self, X):
        """Partial derivative of the lpdf w.r.t. the shape parameter."""

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        z = (X - self.loc) / self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            # Handle z^k * ln(z), which -> 0 as z -> 0.
            # This prevents NaN from 0 * -inf.
            grad = (
                dtype(1.0) / self.shape + np.log(z) - np.nan_to_num(np.power(z, self.shape) * np.log(z), nan=dtype(0.0))
            )
        result = np.where(self.loc < X, np.nan_to_num(grad), dtype(0.0))

        if is_scalar:
            return result[()]
        return result

    def _dlog_loc(self, X):
        """Partial derivative of the lpdf w.r.t. the loc parameter."""

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        z = (X - self.loc) / self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = -(self.shape - dtype(1)) / (X - self.loc) + (self.shape / self.scale) * np.power(
                z, self.shape - dtype(1)
            )
        result = np.where(self.loc < X, np.nan_to_num(grad), dtype(0.0))

        if is_scalar:
            return result[()]
        return result

    def _dlog_scale(self, X):
        """Partial derivative of the lpdf w.r.t. the scale parameter."""

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        z = (X - self.loc) / self.scale
        grad = -self.shape / self.scale + (self.shape / self.scale) * np.power(z, self.shape)
        result = np.where(self.loc < X, grad, dtype(0.0))

        if is_scalar:
            return result[()]
        return result

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
            Returns a 1D array if X is a scalar.
        """

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=self.dtype)

        gradient_calculators = {
            self.PARAM_SHAPE: self._dlog_shape,
            self.PARAM_LOC: self._dlog_loc,
            self.PARAM_SCALE: self._dlog_scale,
        }

        optimizable_params = sorted(list(self.params_to_optimize))

        if not optimizable_params:
            return np.empty((len(X), 0), dtype=self.dtype)

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
        DType | NDArray[DType]
            A scalar or NumPy array containing the generated samples.
        """

        samples = weibull_min.rvs(c=self.shape, loc=self.loc, scale=self.scale, size=size)

        if size is None:
            return self.dtype(samples)
        return np.asarray(samples, dtype=self.dtype)

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Weibull(shape=2.0, loc=0.0, scale=1.0, dtype=np.float64)".
        """

        return (
            f"{self.__class__.__name__}(shape={self.shape}, "
            f"loc={self.loc}, scale={self.scale}, dtype=np.{self.dtype.__name__})"
        )
