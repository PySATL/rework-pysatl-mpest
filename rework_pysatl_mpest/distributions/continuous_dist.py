"""
A module providing an abstract class for continuous distributions.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy import float64
from numpy.typing import ArrayLike, NDArray


class ContinuousDistribution(ABC):
    """Abstract base class for continuous distributions.

    This class defines the basic mathematical functions of distributions
    that must be implemented by specific distributions.
    This class also provides some functions that are common to all distributions.

    .. rubric:: Key functionality

    - Parameter management: fixing and releasing parameters for optimization.
    - Function calculation: standard implementation of `q_function`.
    - Parameter vectorization: getting and setting parameters from a numpy vector.

    .. rubric:: Instance attributes

    :ivar: name
    :ivar: params
    :ivar: fixed_params
    :ivar: params_to_optimize

    .. rubric:: Implementation Requirements

    Subclasses must:
    1. Implement the :attr:`~name` property to identify the distribution.
    2. Implement the :attr:`~params` property to return all parameter names.
    3. Implement the abstract methods: :meth:`~pdf`, :meth:`~ppf`, :meth:`~lpdf`,
       :meth:`~log_gradients`, and :meth:`~generate`.
    4. Define their parameters as instance attributes (e.g. `self.loc`, `self.scale`) with descriptor `Parameter`.

    Attributes:
        fixed_params (set[str]): Set of parameter names that are
            fixed and not subject to optimization.

    """

    def __init__(self):
        """The constructor must be called by all descendants for the `fixed_params` attribute to appear."""

        self._fixed_params: set[str] = set()

    def q_function(self, X: ArrayLike, H: ArrayLike) -> float:
        """Calculates the Q-function (expectation of the complete log-likelihood).

        .. rubric:: Mathematical definition

        The formula for the Q-function for one component of a mixture is:

        .. math::

            `Q(\\theta | \\theta^{(t)}) = \\sum_{i=1}^{n} h_i \\ln f(x_i | \\theta)`

        Where:
            - :math:`X = \\{x_1, x_2, \\ldots, x_n\\}` is the input data sample.
            - :math:`n` — number of observations in the sample.
            - :math:`h_i` — responsibility (posterior probability) of this
                component for observation :math:`x_i`, calculated with **old** parameters :math:`\\theta^{(t)}`.
                Corresponds to the argument `H`.
            - :math:`f(x_i | \\theta)` — probability density function for
                observation :math:`x_i` with **new** parameters :math:`\theta`.
            - :math:`\\theta` — new set of parameters for optimization.

        Args:
            X (ArrayLike): Input data array or scalar (sample).
            H (ArrayLike): Array or scalar of posterior probabilities (responsibilities)
                corresponding to each element of `X`.

        Returns:
            float: Q-function value.

        Raises:
            ValueError: If the shapes of X and H are not equal.
        """

        X = np.asarray(X, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)

        if X.shape != H.shape:
            raise ValueError(f"X and H shapes must be equal, while got {X.shape} and {H.shape}")

        lpdf_values = self.lpdf(X)
        return np.dot(H, lpdf_values).item()

    def fix_param(self, name: str):
        """Fixes a parameter, excluding it from the optimization process and further changes.

        Args:
            name (str): The name of the parameter to freeze.

        Raises:
            ValueError: If the parameter with the specified name does not exist.
        """

        if name not in self.params:
            raise ValueError(f"Parameter '{name}' does not exist in this distribution.")

        self._fixed_params.add(name)

    def unfix_param(self, name: str):
        """Unfix a parameter, allowing it to be changed again.

        If the parameter was not fixed, the method does nothing.

        Args:
            name (str): The name of the parameter to unfix.
        """

        self._fixed_params.discard(name)

    def get_params_vector(self, param_names: Sequence[str]) -> NDArray[float64]:
        """Retrieves specified parameter values as a NumPy array.

        Args:
            param_names (Sequence[str]): A sequence of strings with the names
                of the parameters to retrieve.

        Returns:
            NDArray[np.float64]: A 1D NumPy array containing the values of the requested
                parameters in the order they were specified.

        Raises:
            ValueError: If any of the requested parameter names do not exist
                in the distribution's `params`.
        """

        if not set(param_names).issubset(self.params):
            invalid_params = set(param_names) - self.params
            raise ValueError(f"Invalid parameter names provided: {invalid_params}")

        return np.array([getattr(self, name) for name in param_names], dtype=float64)

    def set_params_from_vector(self, param_names: Sequence[str], vector: ArrayLike):
        """Sets parameter values from a NumPy array.

        Updates the distribution's parameters using the values from the provided
        vector. The order of values in the vector must correspond to the order
        of names in `param_names`.

        Args:
            param_names (Sequence[str]): A sequence of parameter names to update.
            vector (ArrayLike): A 1D array-like object (e.g., list, NumPy array) of new values for the parameters.

        Raises:
            ValueError: If any of the requested parameter names do not exist,
                or if the length of `param_names` does not match the length of `vector`.
        """

        vector = np.asarray(vector, dtype=float64)

        if len(param_names) != len(vector):
            raise ValueError("The number of parameter names must match the number of values in the vector.")

        if not set(param_names).issubset(self.params):
            invalid_params = set(param_names) - self.params
            raise ValueError(f"Invalid parameter names provided: {invalid_params}")

        for name, value in zip(param_names, vector):
            setattr(self, name, value.item())

    @property
    def params_to_optimize(self) -> set[str]:
        """set[str]: Gets the set of parameter names that are not fixed."""

        return self.params - self._fixed_params

    @property
    @abstractmethod
    def name(self) -> str:
        """str: The name of the distribution (e.g., 'Normal', 'Gamma')."""

    @property
    @abstractmethod
    def params(self) -> set[str]:
        """set[str]: A set containing the names of all parameters of the distribution."""

    @abstractmethod
    def pdf(self, X: ArrayLike) -> NDArray[float64]:
        """Calculates the Probability Density Function (PDF).

        Args:
            X (ArrayLike): The input data points at which to evaluate the PDF.

        Returns:
            NDArray[np.float64]: The PDF values corresponding to each point in `X`.
        """

    @abstractmethod
    def ppf(self, P: ArrayLike) -> NDArray[float64]:
        """Calculates the Percent Point Function (PPF) or quantile function.

        This is the inverse of the Cumulative Distribution Function (CDF).

        Args:
            P (ArrayLike): The probability values (between 0 and 1) at which
                to evaluate the PPF.

        Returns:
            NDArray[np.float64]: The PPF values corresponding to each probability in `P`.
        """

    @abstractmethod
    def lpdf(self, X: ArrayLike) -> NDArray[float64]:
        """Calculates the Log of the Probability Density Function (LPDF).

        Evaluating the log-PDF is often more numerically stable than
        evaluating the PDF directly, especially for very small probability values.

        Args:
            X (ArrayLike): The input data points at which to evaluate the LPDF.

        Returns:
            NDArray[np.float64]: The log-PDF values corresponding to each point in `X`.
        """

    @abstractmethod
    def log_gradients(self, X: ArrayLike) -> NDArray[float64]:
        """Calculates the gradients of the log-PDF with respect to its parameters.

        The gradients are computed for the parameters that are not fixed.

        Args:
            X (ArrayLike): The input data points at which to calculate the gradients.

        Returns:
            NDArray[np.float64]: An array where each row corresponds to a data point in `X`
                and each column corresponds to the gradient with respect to a
                specific optimizable parameter. The order of columns corresponds
                to the sorted order of `self.params_to_optimize`.
        """

    @abstractmethod
    def generate(self, size: int) -> NDArray[float64]:
        """Generates random samples from the distribution.

        Args:
            size (int): The number of random samples to generate.

        Returns:
            NDArray[np.float64]: A NumPy array containing the generated samples.
        """
