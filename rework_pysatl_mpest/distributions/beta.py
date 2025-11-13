"""Module providing four parametric beta distribution distribution class"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from scipy.special import digamma
from scipy.stats import beta as beta_dist

from ..core import Parameter
from ..typings import DType
from .continuous_dist import ContinuousDistribution


class Beta(ContinuousDistribution):
    """Class for the four-parameteric beta distribution.
       Parameters
       ----------
       alpha : float
           The first shape parameter. Must be positive or zero.
       beta : float
           The second shape parameter. Must be positive or zero.
       left_border : float
           Left border of section [a, b]. Can be any real number.
       right_border : float
           Right border of section [a, b]. Can be any real number.

       Attributes
       ----------
       loc : float
           Location parameter.
       scale : float
           Scale parameter.
       scale : float
           Scale parameter (gamma). Must be positive.
       scale : float
           Scale parameter (gamma). Must be positive.

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

    PARAM_ALPHA = "alpha"
    PARAM_BETA = "beta"
    PARAM_LEFT_BORDER = "left_border"
    PARAM_RIGHT_BORDER = "right_border"

    alpha = Parameter(lambda x: x >= 0.0, "Alpha parameter should be positive or zero")
    beta = Parameter(lambda x: x >= 0.0, "Beta parameter should be positive or zero")
    left_border = Parameter()
    right_border = Parameter()

    def __init__(
        self,
        alpha: float,
        beta: float,
        left_border: float,
        right_border: float,
        dtype: type[DType] = np.float64,  # type: ignore[assignment]
    ):
        super().__init__(dtype=dtype)
        if left_border >= right_border:
            raise ValueError("Left border must be less than right border")
        self.alpha = alpha
        self.beta = beta
        self.left_border = left_border
        self.right_border = right_border

    @property
    def name(self) -> str:
        return "Beta"

    @property
    def params(self) -> set[str]:
        return {self.PARAM_ALPHA, self.PARAM_BETA, self.PARAM_LEFT_BORDER, self.PARAM_RIGHT_BORDER}

    def pdf(self, X):
        """Probability density function (PDF).

        The PDF for the four-parameter beta distribution is:

        .. math::

            f(x | \\alpha, \\beta, a, b) = \\frac{(x - a)^(\\alpha - 1)
            \\cdot (b - x)^(\\beta - 1)}
            { (b - a)^(\\alpha + \\beta - 1) \\cdot B(\\alpha, \\beta)}

        where :math:`a` is the left_border parameter, :math:`b` is the
        right_border parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter,:math:`B(\\alpha, \\beta) =
        \frac{\\Gamma(\\alpha)\\Gamma(\\beta)}{\\Gamma(\\alpha + \\beta)}`
        is the Beta function.

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
        return np.exp(self.lpdf(X))

    def ppf(self, P):
        """Percent Point Function (PPF) or quantile function.

        The PPF for the four-parameter beta distribution is:

        .. math::

            Q(p | \\alpha, \\beta, a, b) = a + (b - a)
            \\cdot ppf(p, \\alpha, \\beta)

        where :math:`a` is the left_border parameter, :math:`b` is the
        right_border parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter.

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
        dtype = self.dtype

        return np.where(
            (P >= 0) & (P <= 1),
            (
                self.left_border
                + (self.right_border - self.left_border) * beta_dist.ppf(P, self.alpha, self.beta).astype(dtype)
            ),
            dtype(np.nan),
        )

    def lpdf(self, X):
        """Log of the Probability Density Function (LPDF).

        The log-PDF for the four-parameter beta distribution is:

        .. math::

            \\ln f(x | \\alpha, \\beta, a, b) &=
            (\\alpha - 1) \\cdot \\ln(x - a) +
            (\\beta - 1) \\cdot \\ln(b - x) \\
            &\\quad - (\\alpha + \\beta - 1) \\cdot \\ln(b - a)
            - \\ln B(\\alpha, \\beta)

        where :math:`a` is the left_border parameter, :math:`b` is the
        right_border parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter, :math:`B(\\alpha, \\beta) =
        \frac{\\Gamma(\\alpha)\\Gamma(\\beta)}{\\Gamma(\\alpha + \\beta)}`

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

        Z = (X - self.left_border) / (self.right_border - self.left_border)

        log_pdf_standard = beta_dist.logpdf(Z, self.alpha, self.beta).astype(dtype)
        result = log_pdf_standard - np.log(self.right_border - self.left_border)

        return np.atleast_1d(result)

    def _dlog_alpha(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`alpha` parameter.

        The derivative is non-zero only for :math:`a < X \\leq b`.

        .. math::

            \frac{\\partial \\ln f(x | \\alpha, \\beta, a, b)}{\\partial \\alpha} =
            \\ln(x - a) - \\ln(b - a)
            - \\psi(\\alpha) + \\psi(\\alpha + \\beta)

        where :math:`a` is the left_border parameter, :math:`b` is the
        right_border parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter, :math:`\\psi(\\cdot)`
        is the digamma function.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[DType]
            The gradient of the lpdf with respect to :attr:`alpha` for each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        in_bounds = (self.left_border < X) & (self.right_border >= X)
        return np.where(
            in_bounds,
            np.log(X - self.left_border)
            - np.log(self.right_border - self.left_border)
            - (dtype(digamma(self.alpha)) - dtype(digamma(self.alpha + self.beta))),
            dtype(0.0),
        )

    def _dlog_beta(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`beta` parameter.

        The derivative is non-zero only for :math:`a < X \\leq b`.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta, a, b)}{\\partial \\beta} =
            \\ln(b - x) - \\ln(b - a)
            - \\psi(\\beta) + \\psi(\\alpha + \\beta)

        where :math:`a` is the left_border parameter, :math:`b` is the
        right_border parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter, :math:`\\psi(\\cdot)`
        is the digamma function..

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[DType]
            The gradient of the lpdf with respect to :attr:`beta` for each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        in_bounds = (self.left_border < X) & (self.right_border >= X)
        return np.where(
            in_bounds,
            np.log(self.right_border - X)
            - np.log(self.right_border - self.left_border)
            - (dtype(digamma(self.beta)) - dtype(digamma(self.alpha + self.beta))),
            dtype(0.0),
        )

    def _dlog_left_border(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`left_border` parameter.

        The derivative is non-zero only for :math:`a < X \\leq b`.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta, a, b)}{\\partial a} =
            \\frac{-\\alpha - \\beta + 1}{a - b} - \\frac{\\alpha - 1}{x - a}

        where :math:`a` is the left_border parameter, :math:`b` is the
        right_border parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[DType]
            The gradient of the lpdf with respect to :attr:`left_border` for each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        in_bounds = (self.left_border < X) & (self.right_border >= X)
        return np.where(
            in_bounds,
            (
                ((self.alpha + self.beta - dtype(1)) / (self.right_border - self.left_border))
                - ((self.alpha - dtype(1)) / (X - self.left_border))
            ),
            dtype(0.0),
        )

    def _dlog_right_border(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`right_border` parameter.

        The derivative is non-zero only for :math:`\\theta_1 < X \\leq \\theta_2`.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta, a, b)}{\\partial b} =
            \\frac{\\alpa_2 - 1}{\\theta_2 - x} - \\frac{\\alpha_1 + \\alpha_2 - 1}{\\theta_2 - \\theta_1}

        where :math:`a` is the left_border parameter, :math:`b` is the
        right_border parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[DType]
            The gradient of the lpdf with respect to :attr:`right_border` for each point in :attr:`X`.
        """
        X = np.asarray(X, dtype=self.dtype)
        dtype = self.dtype

        in_bounds = (self.left_border < X) & (self.right_border >= X)
        return np.where(
            in_bounds,
            (
                ((self.beta - dtype(1)) / (self.right_border - X))
                - ((self.alpha + self.beta - dtype(1)) / (self.right_border - self.left_border))
            ),
            dtype(0.0),
        )

    def log_gradients(self, X):
        """Calculates the gradients of the log-PDF w.r.t. its parameters.

        The gradients are computed for the parameters that are not fixed.

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
            self.PARAM_ALPHA: self._dlog_alpha,
            self.PARAM_BETA: self._dlog_beta,
            self.PARAM_LEFT_BORDER: self._dlog_left_border,
            self.PARAM_RIGHT_BORDER: self._dlog_right_border,
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
        NDArray[DType]
            A NumPy array containing the generated samples.
        """

        return np.asarray(
            beta_dist.rvs(
                self.alpha, self.beta, loc=self.left_border, scale=self.right_border - self.left_border, size=size
            ),
            dtype=self.dtype,
        )

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Beta(alpha=1.0, beta=2.0, left_border=0.0, right_border=1.0, dtype=np.float64)".
        """

        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"left_border={self.left_border}, "
            f"right_border={self.right_border}, "
            f"dtype=np.{self.dtype.__name__})"
        )
