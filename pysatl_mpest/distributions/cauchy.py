"""Module providing Cauchy distribution class"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from scipy.stats import cauchy

from ..core import Parameter
from .continuous_dist import ContinuousDistribution


class Cauchy(ContinuousDistribution):
    """Class for the two-parameter cauchy distribution.

    Parameters
    ----------
    loc : float
        Location parameter. Can be any real number.
    scale : float
        Scale parameter. Must be positive.

    Attributes
    ----------
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

    PARAM_LOC = "loc"
    PARAM_SCALE = "scale"

    loc = Parameter()
    scale = Parameter(lambda x: x > 0.0, "Scale parameter should be positive")

    def __init__(self, loc: float, scale: float):
        super().__init__()
        self.loc = loc
        self.scale = scale

    @property
    def name(self) -> str:
        return "Cauchy"

    @property
    def params(self) -> set[str]:
        return {self.PARAM_LOC, self.PARAM_SCALE}

    def pdf(self, X):
        """Probability density function (PDF).

        The PDF for the two-parameter cauchy distribution is:

        .. math::

            f(x | \\alpha, \\beta) = \\frac{1}{\\pi \\cdot \\beta \\cdot (1 +(\\frac{(x - \\alpha)}{\\beta})^2)}

        where :math:`\\beta` is the scale parameter and :math:`\\alpha` is the
        location parameter.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the PDF.

        Returns
        -------
        np.float64 | FloatArray
            The PDF values corresponding to each point in :attr:`X`.
            Return a scalar when given a scalar, and to return an array when given an array.
        """

        X = np.asarray(X, dtype=np.float64)
        return np.exp(self.lpdf(X))

    def ppf(self, P):
        """Percent Point Function (PPF) or quantile function.

        The PPF for the two-parameter cauchy distribution is:

        .. math::

            Q(p | \\alpha, \\beta) = \\alpha + \\beta \\cdot \\tan(\\pi \\cdot (p - 0.5))

        where :math:`\\alpha` is the location parameter and :math:`\\beta` is the
        scale parameter.

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
            (P >= 0) & (P <= 1),
            np.where(
                (P == 0) | (P == 1),
                np.where(P == 1, np.float64(np.inf), np.float64(-np.inf)),
                self.loc + self.scale * np.tan(np.float64(np.pi) * (P - np.float64(0.5))),
            ),
            np.float64(np.nan),
        )
        if is_scalar:
            return result[()]
        return result

    def lpdf(self, X):
        """Log of the Probability Density Function (LPDF).

        The log-PDF for the two-parameter cauchy distribution is:

        .. math::

            \\ln f(x | \\alpha, \\beta) = \\ln(\\gamma) - \\ln(\\pi \\cdot ((x - \\alpha)^2 + \\beta^2))

        where :math:`\\alpha` is the location parameter and :math:`\\beta` is the
        scale parameter.

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

        result = (
            np.log(np.float64(1.0))
            - np.log(np.float64(np.pi))
            - np.log(self.scale)
            - np.log(np.float64(1.0) + ((X - self.loc) / self.scale) ** 2)
        )

        if is_scalar:
            return result[()]
        return result

    def _dlog_loc(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`loc` parameter.

        The derivative is defined over all real numbers.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta)}{\\partial \\alpha} =
            \\frac{2 \\cdot x - 2 \\cdot \\alpha}{\\beta^2 + x^2 - 2 \\cdot \\alpha \\cdot x + \\alpha^2}

        where :math:`\\alpha` is the location parameter and :math:`\\beta` is the
        scale parameter.

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

        result = (np.float64(2) * X - np.float64(2) * self.loc) / (
            self.scale**2 + X**2 - np.float64(2) * self.loc * X + self.loc**2
        )

        if is_scalar:
            return result[()]
        return result

    def _dlog_scale(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`scale` parameter.

        The derivative is defined over all real numbers.

        .. math::

             \\frac{\\partial \\ln f(x | \\alpha, \\beta)}{\\partial \\alpha} =
             \\frac{-\\beta^2 + x^2 - 2 \\cdot \\alpha \\cdot x + \\alpha^2}{\\beta^3 + \\beta \\cdot (x^2)
             - 2 \\cdot \\alpha \\cdot \\beta \\cdot x + \\beta \\cdot \\alpha^2}

        where :math:`\\alpha` is the location and :math:`\\beta` is the scale.

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

        result = (-(self.scale**2) + X**2 - np.float64(2) * self.loc * X + self.loc**2) / (
            self.scale**3 + self.scale * (X**2) - np.float64(2) * self.loc * self.scale * X + self.scale * self.loc**2
        )

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

        samples = cauchy.rvs(loc=self.loc, scale=self.scale, size=size)

        if size is None:
            return np.float64(samples)
        return np.asarray(samples, dtype=np.float64)

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Cauchy(loc=0.0, scale=2.0)".
        """

        return f"{self.__class__.__name__}(loc={self.loc}, scale={self.scale})"
