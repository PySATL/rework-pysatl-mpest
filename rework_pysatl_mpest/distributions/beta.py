"""Module providing four parametric beta distribution distribution class"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy import float64
from scipy.special import digamma
from scipy.stats import beta as beta_dist

from ..core import Parameter
from .continuous_dist import ContinuousDistribution


class Beta(ContinuousDistribution):
    """Class for the four-parameteric beta distribution."""

    PARAM_ALPHA = "alpha"
    PARAM_BETA = "beta"
    PARAM_LOWER_BOUND = "lower_bound"
    PARAM_UPPER_BOUND = "upper_bound"

    alpha = Parameter(lambda x: x >= 0.0, "Alpha parameter should be positive or zero")
    beta = Parameter(lambda x: x >= 0.0, "Beta parameter should be positive or zero")
    lower_bound = Parameter()
    upper_bound = Parameter()

    def __init__(self, alpha: float, beta: float, lower_bound: float, upper_bound: float):
        super().__init__()
        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be less than upper bound")
        self.alpha = alpha
        self.beta = beta
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    def name(self) -> str:
        return "Beta"

    @property
    def params(self) -> set[str]:
        return {self.PARAM_ALPHA, self.PARAM_BETA, self.PARAM_LOWER_BOUND, self.PARAM_UPPER_BOUND}

    def pdf(self, X):
        """Probability density function (PDF).

        The PDF for the four-parameter beta distribution is:

        .. math::

            f(x | \\alpha, \\beta, a, b) = \\frac{(x - a)^(\\alpha - 1)
            \\cdot (b - x)^(\\beta - 1)}
            { (b - a)^(\\alpha + \\beta - 1) \\cdot B(\\alpha, \\beta)}

        where :math:`a` is the lower_bound parameter, :math:`b` is the
        upper_bound parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter,:math:`B(\\alpha, \\beta) =
        \frac{\\Gamma(\\alpha)\\Gamma(\\beta)}{\\Gamma(\\alpha + \\beta)}`
        is the Beta function.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the PDF.

        Returns
        -------
        NDArray[np.float64]
            The PDF values corresponding to each point in :attr:`X`.

        """
        return np.exp(self.lpdf(X))

    def ppf(self, P):
        """Percent Point Function (PPF) or quantile function.

        The PPF for the four-parameter beta distribution is:

        .. math::

            Q(p | \\alpha, \\beta, a, b) = a + (b - a)
            \\cdot ppf(p, \\alpha, \\beta)

        where :math:`a` is the lower_bound parameter, :math:`b` is the
        upper_bound parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter.

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
        return np.where(
            (P >= 0) & (P <= 1),
            (self.lower_bound + (self.upper_bound - self.lower_bound) * beta_dist.ppf(P, self.alpha, self.beta)),
            np.nan,
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

        where :math:`a` is the lower_bound parameter, :math:`b` is the
        upper_bound parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter, :math:`B(\\alpha, \\beta) =
        \frac{\\Gamma(\\alpha)\\Gamma(\\beta)}{\\Gamma(\\alpha + \\beta)}`

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

        Z = (X - self.lower_bound) / (self.upper_bound - self.lower_bound)

        log_pdf_standard = beta_dist.logpdf(Z, self.alpha, self.beta)

        return log_pdf_standard - np.log(self.upper_bound - self.lower_bound)

    def _dlog_alpha(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`alpha` parameter.

        The derivative is non-zero only for :math:`a < X \\leq b`.

        .. math::

            \frac{\\partial \\ln f(x | \\alpha, \\beta, a, b)}{\\partial \\alpha} =
            \\ln(x - a) - \\ln(b - a)
            - \\psi(\\alpha) + \\psi(\\alpha + \\beta)

        where :math:`a` is the lower_bound parameter, :math:`b` is the
        upper_bound parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter, :math:`\\psi(\\cdot)`
        is the digamma function.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[np.float64]
            The gradient of the lpdf with respect to :attr:`alpha` for each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=float64)
        in_bounds = (self.lower_bound < X) & (self.upper_bound >= X)
        return np.where(
            in_bounds,
            np.log(X - self.lower_bound)
            - np.log(self.upper_bound - self.lower_bound)
            - (digamma(self.alpha) - digamma(self.alpha + self.beta)),
            0.0,
        )

    def _dlog_beta(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`beta` parameter.

        The derivative is non-zero only for :math:`a < X \\leq b`.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta, a, b)}{\\partial \\beta} =
            \\ln(b - x) - \\ln(b - a)
            - \\psi(\\beta) + \\psi(\\alpha + \\beta)

        where :math:`a` is the lower_bound parameter, :math:`b` is the
        upper_bound parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter, :math:`\\psi(\\cdot)`
        is the digamma function..

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[np.float64]
            The gradient of the lpdf with respect to :attr:`beta` for each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=float64)
        in_bounds = (self.lower_bound < X) & (self.upper_bound >= X)
        return np.where(
            in_bounds,
            np.log(self.upper_bound - X)
            - np.log(self.upper_bound - self.lower_bound)
            - (digamma(self.beta) - digamma(self.alpha + self.beta)),
            0.0,
        )

    def _dlog_lower_bound(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`lower_bound` parameter.

        The derivative is non-zero only for :math:`a < X \\leq b`.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta, a, b)}{\\partial a} =
            \\frac{-\\alpha - \\beta + 1}{a - b} - \\frac{\\alpha - 1}{x - a}

        where :math:`a` is the lower_bound parameter, :math:`b` is the
        upper_bound parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[np.float64]
            The gradient of the lpdf with respect to :attr:`lower_bound` for each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=float64)
        in_bounds = (self.lower_bound < X) & (self.upper_bound >= X)
        return np.where(
            in_bounds,
            (
                ((self.alpha + self.beta - 1) / (self.upper_bound - self.lower_bound))
                - ((self.alpha - 1) / (X - self.lower_bound))
            ),
            0.0,
        )

    def _dlog_upper_bound(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`upper_bound` parameter.

        The derivative is non-zero only for :math:`\\theta_1 < X \\leq \\theta_2`.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha, \\beta, a, b)}{\\partial b} =
            \\frac{\\alpa_2 - 1}{\\theta_2 - x} - \\frac{\\alpha_1 + \\alpha_2 - 1}{\\theta_2 - \\theta_1}

        where :math:`a` is the lower_bound parameter, :math:`b` is the
        upper_bound parameter, :math:`\\alpha` is the first shape parameter and
        :math:`\\beta` is the second shape parameter.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[np.float64]
            The gradient of the lpdf with respect to :attr:`upper_bound` for each point in :attr:`X`.
        """
        X = np.asarray(X, dtype=float64)
        in_bounds = (self.lower_bound < X) & (self.upper_bound >= X)
        return np.where(
            in_bounds,
            (
                ((self.beta - 1) / (self.upper_bound - X))
                - ((self.alpha + self.beta - 1) / (self.upper_bound - self.lower_bound))
            ),
            0.0,
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
        NDArray[np.float64]
            An array where each row corresponds to a data point in :attr:`X`
            and each column corresponds to the gradient with respect to a
            specific optimizable parameter. The order of columns corresponds
            to the sorted order of :attr:`self.params_to_optimize`.
        """
        X = np.asarray(X, dtype=float64)

        gradient_calculators = {
            self.PARAM_ALPHA: self._dlog_alpha,
            self.PARAM_BETA: self._dlog_beta,
            self.PARAM_LOWER_BOUND: self._dlog_lower_bound,
            self.PARAM_UPPER_BOUND: self._dlog_upper_bound,
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

        return np.asarray(
            beta_dist.rvs(
                self.alpha, self.beta, loc=self.lower_bound, scale=self.upper_bound - self.lower_bound, size=size
            ),
            dtype=float64,
        )

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Beta(alpha=1.0, beta=2.0, lower_bound=0.0, upper_bound=1.0)".
        """

        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"lower_bound={self.lower_bound}, "
            f"upper_bound={self.upper_bound})"
        )
