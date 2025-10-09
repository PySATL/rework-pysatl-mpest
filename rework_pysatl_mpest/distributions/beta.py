"""Module providing four parametric beta distribution distribution class"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy import float64
from scipy.special import digamma
from scipy.stats import beta as beta_dist

from rework_pysatl_mpest.core.parameter import Parameter
from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution


class Beta(ContinuousDistribution):
    """Class for the four-parameteric beta distribution."""

    PARAM_SHAPE1 = "shape1"
    PARAM_SHAPE2 = "shape2"
    PARAM_LOWER_BOUND = "lower_bound"
    PARAM_UPPER_BOUND = "upper_bound"

    shape1 = Parameter(lambda x: x >= 0.0, "Shape1 parameter should be positive or zero")
    shape2 = Parameter(lambda x: x >= 0.0, "Shape2 parameter should be positive or zero")
    lower_bound = Parameter()
    upper_bound = Parameter()

    def __init__(self, shape1: float, shape2: float, lower_bound: float, upper_bound: float):
        super().__init__()
        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be smaller Upper bound")
        self.shape1 = shape1
        self.shape2 = shape2
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    def name(self) -> str:
        return "Beta"

    @property
    def params(self) -> set[str]:
        return {self.PARAM_SHAPE1, self.PARAM_SHAPE2, self.PARAM_LOWER_BOUND, self.PARAM_UPPER_BOUND}

    def pdf(self, X):
        """Probability density function (PDF).

        The PDF for the four-parameter beta distribution is:

        .. math::

            f(x | \\alpha_1, \\alpha_2, \\theta_1, \\theta_2) = \\frac{(x - \\theta_1)^(\\alpha_1 - 1)
            \\cdot (\\theta_2 - x)^(\\alpha_2 - 1)}
            { (\\theta_2 - \\theta_1)^(\\alpha_1 + \\alpha_2 - 1) \\cdot B(\\alpha_1, \\alpha_2)}

        where :math:`\\theta_1` is the lower_bound parameter, :math:`\\theta_2` is the
        upper_bound parameter, :math:`\\alpha_1` is the shape1 parameter and
        :math:`\\alpha_2` is the shape2 parameter,:math:`B(\\alpha_1, \\alpha_2) =
        \frac{\\Gamma(\\alpha_1)\\Gamma(\\alpha_2)}{\\Gamma(\\alpha_1 + \\alpha_2)}`
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

            Q(p | \\alpha_1, \\alpha_2, \\theta_1, \\theta_2) = \\theta_1 + (\\theta_2 - \\theta_1)
            \\cdot ppf(p, \\alpha_1, \\alpha_2)

        where :math:`\\theta_1` is the lower_bound parameter, :math:`\\theta_2`
        is the upper_bound parameter, :math:`\\alpha_1` is the shape1 parameter
        and :math:`\\alpha_2` is the shape2 parameter.
        """
        P = np.asarray(P, dtype=float64)
        return np.where(
            (P >= 0) & (P <= 1),
            (self.lower_bound + (self.upper_bound - self.lower_bound) * beta_dist.ppf(P, self.shape1, self.shape2)),
            np.nan,
        )

    def lpdf(self, X):
        """Log of the Probability Density Function (LPDF).

        The log-PDF for the four-parameter beta distribution is:

        .. math::

            \\ln f(x | \\alpha_1, \\alpha_2, \\theta_1, \\theta_2) &=
            (\\alpha_1 - 1) \\cdot \\ln(x - \\theta_1) +
            (\\alpha_2 - 1) \\cdot \\ln(\\theta_2 - x) \\
            &\\quad - (\\alpha_1 + \\alpha_2 - 1) \\cdot \\ln(\\theta_2 - \\theta_1)
            - \\ln B(\\alpha_1, \\alpha_2)

        where :math:`\\theta_1` is the lower_bound parameter, :math:`\\theta_2` is the
        upper_bound parameter, :math:`\\alpha_1` is the shape1 parameter and
        :math:`\\alpha_2` is the shape2 parameter, :math:`B(\alpha_1, \alpha_2) =
        \frac{\\Gamma(\\alpha_1)\\Gamma(\\alpha_2)}{\\Gamma(\\alpha_1 + \\alpha_2)}`

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

        log_pdf_standard = beta_dist.logpdf(Z, self.shape1, self.shape2)

        return log_pdf_standard - np.log(self.upper_bound - self.lower_bound)

    def _dlog_shape1(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`shape1` parameter.

        The derivative is non-zero only for :math:`\\theta_1 < X \\leq \\theta_2`.

        .. math::

            \frac{\\partial \\ln f}{\\partial \\alpha_1} =
            \\ln(x - \\theta_1) - \\ln(\\theta_2 - \\theta_1)
            - \\psi(\\alpha_1) + \\psi(\\alpha_1 + \\alpha_2)

        where :math:`\\theta_1` is the :attr:`lower_bound` parameter, :math:`\\theta_2` is the
        :attr:`upper_bound` parameter, :math:`\\alpha_1` is the :attr:`shape1` parameter and
        :math:`\\alpha_2` is the :attr:`shape2` parameter, :math:`\\psi(\\cdot)`
        is the digamma function.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[np.float64]
            The gradient of the lpdf with respect to :attr:`shape1` for each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=float64)
        in_bounds = (self.lower_bound < X) & (self.upper_bound >= X)
        return np.where(
            in_bounds,
            np.log(X - self.lower_bound)
            - np.log(self.upper_bound - self.lower_bound)
            - (digamma(self.shape1) - digamma(self.shape1 + self.shape2)),
            0.0,
        )

    def _dlog_shape2(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`shape2` parameter.

        The derivative is non-zero only for :math:`\\theta_1 < X \\leq \\theta_2`.

        .. math::

            \\frac{\\partial \\ln f}{\\partial \\alpha_2} =
            \\ln(\\theta_2 - x) - \\ln(\\theta_2 - \\theta_1)
            - \\psi(\\alpha_2) + \\psi(\\alpha_1 + \\alpha_2)

        where :math:`\\theta_1` is the :attr:`lower_bound` parameter, :math:`\\theta_2` is the
        :attr:`upper_bound` parameter, :math:`\\alpha_1` is the :attr:`shape1` parameter and
        :math:`\\alpha_2` is the :attr:`shape2` parameter, :math:`\\psi(\\cdot)`
        is the digamma function..

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        NDArray[np.float64]
            The gradient of the lpdf with respect to :attr:`shape2` for each point in :attr:`X`.
        """

        X = np.asarray(X, dtype=float64)
        in_bounds = (self.lower_bound < X) & (self.upper_bound >= X)
        return np.where(
            in_bounds,
            np.log(self.upper_bound - X)
            - np.log(self.upper_bound - self.lower_bound)
            - (digamma(self.shape2) - digamma(self.shape1 + self.shape2)),
            0.0,
        )

    def _dlog_lower_bound(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`lower_bound` parameter.

        The derivative is non-zero only for :math:`\\theta_1 < X \\leq \\theta_2`.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha_1, \\alpha_2, \\theta_1, \\theta_2)}{\\partial \\theta_1} =
            \\frac{-\\alpha_1 - \\alpha_2 + 1}{\\theta_1 - \\theta_2} - \\frac{\\alpha_1 - 1}{x - \\theta_1}

        where :math:`\\theta_1` is the :attr:`lower_bound` parameter, :math:`\\theta_2` is the
        :attr:`upper_bound` parameter, :math:`\\alpha_1` is the :attr:`shape1` parameter and
        :math:`\\alpha_2` is the :attr:`shape2` parameter.

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
                ((self.shape1 + self.shape2 - 1) / (self.upper_bound - self.lower_bound))
                - ((self.shape1 - 1) / (X - self.lower_bound))
            ),
            0.0,
        )

    def _dlog_upper_bound(self, X):
        """Partial derivative of the lpdf w.r.t. the :attr:`upper_bound` parameter.

        The derivative is non-zero only for :math:`\\theta_1 < X \\leq \\theta_2`.

        .. math::

            \\frac{\\partial \\ln f(x | \\alpha_1, \\alpha_2, \\theta_1, \\theta_2)}{\\partial \\theta_2} =
            \\frac{\\alpa_2 - 1}{\\theta_2 - x} - \\frac{\\alpha_1 + \\alpha_2 - 1}{\\theta_2 - \\theta_1}

        where :math:`\\theta_1` is the :attr:`lower_bound` parameter, :math:`\\theta_2` is the
        :attr:`upper_bound` parameter, :math:`\\alpha_1` is the :attr:`shape1` parameter and
        :math:`\\alpha_2` is the :attr:`shape2` parameter.

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
                ((self.shape2 - 1) / (self.upper_bound - X))
                - ((self.shape1 + self.shape2 - 1) / (self.upper_bound - self.lower_bound))
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
            self.PARAM_SHAPE1: self._dlog_shape1,
            self.PARAM_SHAPE2: self._dlog_shape2,
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
                self.shape1, self.shape2, loc=self.lower_bound, scale=self.upper_bound - self.lower_bound, size=size
            ),
            dtype=float64,
        )

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Beta(shape1=1.0, shape2=2.0, lower_bound=0.0, upper_bound=1.0)".
        """

        return (
            f"{self.__class__.__name__}("
            f"shape1={self.shape1}, "
            f"shape2={self.shape2}, "
            f"lower_bound={self.lower_bound}, "
            f"upper_bound={self.upper_bound})"
        )
