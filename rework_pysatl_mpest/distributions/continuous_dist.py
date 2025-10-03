"""
A module providing an abstract class for continuous distributions.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Sequence

from numpy import float64
from numpy.typing import ArrayLike, NDArray


class ContinuousDistribution(ABC):
    """Abstract base class for continuous distributions.

    This class defines the basic mathematical functions of distributions
    that must be implemented by specific distributions. This class also
    provides some functions that are common to all distributions.

    Attributes
    ----------
    name: str
        The name of the distribution (e.g., 'Normal', 'Gamma').
    params: set[str]
        The names of all parameters of the distribution
    params_to_optimize: set[str]
        Parameters names that are not fixed and can be optimized.
    fixed_params : set[str]
        A set of parameter names that are fixed and not subject to
        optimization.

    Methods
    -------

    **Implemented methods**

    .. autosummary::
        :toctree: generated/

        fix_param
        unfix_param
        get_params_vector
        set_params_from_vector

    **Abstract methods**

    .. autosummary::
        :toctree: generated/

        ppf
        pdf
        lpdf
        log_gradients
        generate

    Notes
    -----
    **Key Functionality**

    - Parameter management: fixing and releasing parameters for optimization.
    - Parameter vectorization: getting and setting parameters from a numpy
      vector.

    **Implementation Requirements**

    Subclasses must:

    1. Implement the :attr:`name` property to identify the distribution.

    2. Implement the :attr:`params` property to return all parameter names.

    3. Implement the abstract methods: :meth:`pdf`, :meth:`ppf`,
       :meth:`lpdf`, :meth:`log_gradients`, and :meth:`generate`.

    4. Define their parameters as instance attributes (e.g., :attr:`self.loc`,
       :attr:`self.scale`) with a :class:`rework_pysatl_mpest.core.Parameter` descriptor.

    """

    def __init__(self):
        """The constructor must be called by all descendants for the
        `fixed_params` attribute to be initialized.
        """

        self._fixed_params: set[str] = set()

    def fix_param(self, name: str):
        """Fixes a parameter, excluding it from optimization and further changes.

        Parameters
        ----------
        name : str
            The name of the parameter to freeze.

        Raises
        ------
        ValueError
            If a parameter with the specified name does not exist.
        """

        if name not in self.params:
            raise ValueError(f"Parameter '{name}' does not exist in this distribution.")

        self._fixed_params.add(name)

    def unfix_param(self, name: str):
        """Unfixes a parameter, allowing it to be changed again.

        If the parameter was not fixed, the method does nothing.

        Parameters
        ----------
        name : str
            The name of the parameter to unfix.
        """

        self._fixed_params.discard(name)

    def get_params_vector(self, param_names: Sequence[str]) -> list[float]:
        """Retrieves specified parameter values as a list.

        Parameters
        ----------
        param_names : Sequence[str]
            A sequence of strings with the names of the parameters to retrieve.

        Returns
        -------
        list[float]
            A list containing the values of the requested parameters
            in the specified order.

        Raises
        ------
        ValueError
            If any of the requested parameter names do not exist in the
            distribution's :attr:`params`.
        """

        if not set(param_names).issubset(self.params):
            invalid_params = set(param_names) - self.params
            raise ValueError(f"Invalid parameter names provided: {invalid_params}")

        return [getattr(self, name) for name in param_names]

    def set_params_from_vector(self, param_names: Sequence[str], vector: Sequence[float]):
        """Sets parameter values from a sequence of floats.

        Updates the distribution's parameters using values from the provided
        sequence. The order of values in the :attr:`vector` must correspond to the order
        of names in :attr:`param_names`.

        Parameters
        ----------
        param_names : Sequence[str]
            A sequence of parameter names to update.
        vector : Sequence[float]
            A sequence of new values for the parameters.

        Raises
        ------
        ValueError
            If any parameter names do not exist, or if the length of
            :attr:`param_names` does not match the length of :attr:`vector`.
        """

        if len(param_names) != len(vector):
            raise ValueError("The number of parameter names must match the number of values in the vector.")

        if not set(param_names).issubset(self.params):
            invalid_params = set(param_names) - self.params
            raise ValueError(f"Invalid parameter names provided: {invalid_params}")

        for name, value in zip(param_names, vector):
            setattr(self, name, value)

    @property
    @abstractmethod
    def name(self) -> str:
        """str: The name of the distribution (e.g., 'Normal', 'Gamma')."""

    @property
    @abstractmethod
    def params(self) -> set[str]:
        """set[str]: A set containing the names of all parameters of the distribution."""

    @property
    def params_to_optimize(self) -> set[str]:
        """set[str]: Gets the set of parameter names that are not fixed."""

        return self.params - self._fixed_params

    @abstractmethod
    def pdf(self, X: ArrayLike) -> NDArray[float64]:
        """Probability Density Function.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the PDF.

        Returns
        -------
        NDArray[np.float64]
            The PDF values corresponding to each point in :attr:`X`.
        """

    @abstractmethod
    def ppf(self, P: ArrayLike) -> NDArray[float64]:
        """Percent Point Function (PPF) or quantile function.

        This is the inverse of the Cumulative Distribution Function (CDF).

        Parameters
        ----------
        P : ArrayLike
            The probability values (between 0 and 1) at which to evaluate
            the PPF.

        Returns
        -------
        NDArray[np.float64]
            The PPF values corresponding to each probability in :attr:`P`.
        """

    @abstractmethod
    def lpdf(self, X: ArrayLike) -> NDArray[float64]:
        """Logarithm of the Probability Density Function.

        Evaluating the log-PDF is often more numerically stable than
        evaluating the PDF directly, especially for very small probability
        values.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the LPDF.

        Returns
        -------
        NDArray[np.float64]
            The log-PDF values corresponding to each point in :attr:`X`.
        """

    @abstractmethod
    def log_gradients(self, X: ArrayLike) -> NDArray[float64]:
        """Calculates the gradients of the log-PDF with respect to its parameters.

        The gradients are computed for the parameters that are not fixed.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to calculate the gradients.

        Returns
        -------
        NDArray[np.float64]
            An array where each row corresponds to a data point in :attr:`X` and
            each column corresponds to the gradient with respect to a specific
            optimizable parameter. The order of columns corresponds to the
            sorted order of :attr:`params_to_optimize`.
        """

    @abstractmethod
    def generate(self, size: int) -> NDArray[float64]:
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
