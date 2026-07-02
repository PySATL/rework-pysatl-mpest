"""Module providing pareto type 1 distribution class"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from scipy.stats import pareto

from ..core.parameter import Parameter
from ..distributions.continuous_dist import ContinuousDistribution


class Pareto(ContinuousDistribution):
    """Class for the two-parameter Pareto distribution.

    The Pareto distribution is a power-law probability distribution commonly used
    to model phenomena with heavy-tailed behavior, such as income distribution,
    city population sizes, or file sizes.

    Parameters
    ----------
    shape : float
        Shape parameter. Must be positive.
    scale : float
        Scale parameter. Must be positive.

    Attributes
    ----------
    shape : float
        Shape parameter of the distribution.
    scale : float
        Scale (minimum) parameter of the distribution.

    Methods
    -------

    .. autosummary::
        :toctree: generated/

        pdf
        ppf
        lpdf
        log_gradients
        generate
    """

    PARAM_SHAPE = "shape"
    PARAM_SCALE = "scale"

    shape = Parameter(lambda x: x > 0, "Shape parameter must be a positive")
    scale = Parameter(lambda x: x > 0, "Scale parameter must be a positive")

    def __init__(self, shape: float, scale: float):
        super().__init__()
        self.shape = shape
        self.scale = scale

    @property
    def name(self) -> str:
        return "Pareto"

    @property
    def params(self) -> set[str]:
        return {self.PARAM_SHAPE, self.PARAM_SCALE}

    def pdf(self, X):
        """Probability density function (PDF).

        The PDF for the two-parameter Pareto distribution is:

        .. math::

            f(x | \\alpha, \\beta) = \\frac{\\alpha \\cdot \\beta^\\alpha}{x^{\\alpha + 1}}

        where :math:`\\alpha` is the :attr:`shape` parameter and :math:`\\beta` is the
        :attr:`scale` parameter. The function is zero for :math:`x < \\beta`.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the PDF.

        Returns
        -------
        np.float64 | FloatArray
            The log-PDF values corresponding to each point in :attr:`X`.
            Return a scalar when given a scalar, and to return an array when given an array.
        """

        X = np.asarray(X, dtype=np.float64)
        return np.exp(self.lpdf(X))

    def ppf(self, P):
        """Percent Point Function (PPF) or quantile function.

        The PPF for the two-parameter Pareto distribution is:

        .. math::

            Q(p | \\alpha, \\beta) = \\beta \\cdot (1 - p)^{-1/\\alpha}

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

        result = np.where(
            (P >= 0) & (P <= 1), self.scale * (np.float64(1) - P) ** (np.float64(-1.0) / self.shape), np.float64(np.nan)
        )

        if is_scalar:
            return result[()]
        return result

    def lpdf(self, X):
        """Log of the Probability Density Function (LPDF).

        The log-PDF for the two-parameter Pareto distribution is:

        .. math::

            \\ln f(x | \\alpha, \\beta) = \\alpha \\ln \\beta + \\ln \\alpha - (\\alpha + 1) \\ln x

        where :math:`\\alpha` is the :attr:`shape` parameter and :math:`\\beta` is the
        :attr:`scale` parameter. The log-density is :math:`-\\infty` for :math:`x < \\beta`.

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

        result = np.where(
            self.scale <= X,
            np.log(self.shape) + self.shape * np.log(self.scale) - (np.float64(1) + self.shape) * np.log(X),
            np.float64(-np.inf),
        )

        if is_scalar:
            return result[()]
        return result

    def _dlog_shape(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`shape` parameter.

        The derivative is non-zero only for :math:`X \\geq \\text{scale}`.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta)}{\\partial \\alpha} =
            \\frac{1}{\\alpha} + \\ln \\beta - \\ln x

        where :math:`\\alpha` is the :attr:`shape` parameter and :math:`\\beta` is the
        :attr:`scale` parameter.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        np.float64 | FloatArray
            The gradient of the lpdf with respect to :attr:`shape` for each point in :attr:`X`.
            Return a scalar when given a scalar, and to return an array when given an array.
        """

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=np.float64)

        result = np.where(
            self.scale <= X, np.float64(1.0) / self.shape + np.log(self.scale) - np.log(X), np.float64(0.0)
        )

        if is_scalar:
            return result[()]
        return result

    def _dlog_scale(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`scale` parameter.

        The derivative is non-zero only for :math:`X \\geq \\text{scale}`.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta)}{\\partial \\beta} = \\frac{\\alpha}{\\beta}

        where :math:`\\alpha` is the :attr:`shape` parameter and :math:`\\beta` is the
        :attr:`scale` parameter.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        np.float64 | FloatArray
            The gradient of the lpdf with respect to :attr:`scale` for each point in :attr:`X`.
            Return a scalar when given a scalar, and to return an array when given an array.
        """

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=np.float64)

        result = np.where(self.scale <= X, self.shape / self.scale, np.float64(0.0))
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
            self.PARAM_SHAPE: self._dlog_shape,
            self.PARAM_SCALE: self._dlog_scale,
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

        samples = pareto.rvs(scale=self.scale, b=self.shape, size=size)

        if size is None:
            return np.float64(samples)
        return np.asarray(samples, dtype=np.float64)

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Pareto(shape=0.0, scale=2.0)".
        """

        return f"{self.__class__.__name__}(shape={self.shape}, scale={self.scale})"
