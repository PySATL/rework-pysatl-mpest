"""A module that provides the `MixtureModel` class, which allows for the creation,
manipulation, and analysis of a finite mixture of continuous probability
distributions."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from typing import Optional

import numpy as np
from numpy import float64
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp, softmax

from rework_pysatl_mpest.distributions.continuous_dist import ContinuousDistribution


class MixtureModel:
    """Represents a finite mixture of continuous probability distributions.

    This class encapsulates a collection of distribution components and their
    corresponding weights.

    Parameters
    ----------
    components : Sequence[ContinuousDistribution]
        A sequence of distribution objects that will form the mixture.
    weights : Optional[ArrayLike], optional
        An array of initial weights for the components. The weights must be
        positive and sum to 1. If None, components are assigned equal
        weights. Defaults to None.

    Attributes
    ----------
    components : tuple[ContinuousDistribution]
        A tuple of the distribution objects that form the mixture.
    n_components : int
        The number of components in the mixture.
    weights : NDArray[np.float64]
        A NumPy array of the normalized weights for each component. The sum
        of weights is always 1.
    log_weights : NDArray[np.float64]
        A NumPy array of the natural logarithm of the component weights.

    Raises
    ------
    ValueError
        If the list of components is empty, or if the provided weights are
        invalid (e.g., negative, wrong number of elements, or do not sum to 1).

    Methods
    -------

    .. autosummary::
        :toctree: generated/

        add_component
        remove_component
        pdf
        lpdf
        loglikelihood
        generate
    """

    def __init__(self, components: Sequence[ContinuousDistribution], weights: Optional[ArrayLike] = None):
        n_components = len(components)
        if n_components == 0:
            raise ValueError("List of components cannot be an empty")

        if weights is None:
            weights = np.full(n_components, 1.0 / n_components)
        else:
            weights = np.asarray(weights, dtype=float64)
            self._validate_weights(n_components, weights)

        self._components = list(components)
        self._log_weights = np.log(weights + 1e-30)
        self._cached_weights: Optional[NDArray[float64]] = None

    def _validate_weights(self, n_components: int, weights: NDArray[float64]):
        """Validates the component weights.

        Parameters
        ----------
        n_components : int
            The expected number of components.
        weights : NDArray[np.float64]
            The array of weights to validate.

        Raises
        ------
        ValueError
            If the number of weights does not match the number of components,
            if any weight is negative, or if the weights do not sum to 1.
        """

        if len(weights) != n_components:
            raise ValueError(f"Components number ({n_components}) must be equal to weights number ({len(weights)}).")

        if np.any(weights < 0):
            raise ValueError("Weights must be positive.")

        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError(f"Sum of the weights must be equal 1, but it equal {np.sum(weights)}.")

    @property
    def n_components(self):
        """int: The number of components in the mixture model."""

        return len(self.components)

    @property
    def components(self):
        """tuple[ContinuousDistribution, ...]: The components of the mixture."""

        return tuple(self._components)

    @property
    def weights(self) -> NDArray[float64]:
        """NDArray[np.float64]: The normalized weights of the components.

        The weights are computed from the log-weights using the softmax
        function and cached for efficiency.
        """

        if self._cached_weights is None:
            self._cached_weights = softmax(self._log_weights)

        return self._cached_weights  # type: ignore

    @property
    def log_weights(self) -> NDArray[float64]:
        """NDArray[np.float64]: The logarithm of the component weights."""

        return self._log_weights

    @log_weights.setter
    def log_weights(self, new_log_weights: ArrayLike):
        """Sets the log-weights for the components.

        Parameters
        ----------
        new_log_weights : ArrayLike
            A 1D NumPy array of new log-weights.

        Raises
        ------
        ValueError
            If the length of the new log-weights vector does not match the
            number of components.
        """

        new_log_weights = np.asarray(new_log_weights, dtype=float64)

        if len(new_log_weights) != self.n_components:
            raise ValueError("The length of the new logit vector does not match the number of components.")
        self._log_weights = np.asarray(new_log_weights, dtype=float)
        self._cached_weights = None

    def add_component(self, component: ContinuousDistribution, weight: float):
        """Adds a new component to the mix, preserving the proportions of the existing component weights.

        If :attr:`weight` is specified for the new component, the old component
        weights are multiplied by `(1 - weight)`.

        Parameters
        ----------
        component : ContinuousDistribution
            The distribution component to add.
        weight : float
            The weight for the new component, a number in the range (0, 1).

        Raises
        ------
        ValueError
            If the specified :attr:`weight` is outside the range (0, 1).
        """

        if not (0 < weight < 1):
            raise ValueError("The weight of the new component must be in the range (0, 1).")

        self._log_weights += np.log(1 - weight)
        new_log_weight = np.log(weight)
        self._log_weights = np.append(self._log_weights, new_log_weight)

        self._components.append(component)
        self._cached_weights = None

    def remove_component(self, component_idx: int):
        """Removes a component from the mixture by its index.

        The weights of the remaining components are renormalized to sum to 1.

        Parameters
        ----------
        component_idx : int
            The index of the component to remove.

        Raises
        ------
        IndexError
            If the component index is out of range.
        ValueError
            If an attempt is made to remove the last component.
        """

        n_components = self.n_components
        if not 0 <= component_idx < n_components:
            raise IndexError(f"Index {component_idx} out of range [0, {n_components - 1}]")

        if n_components <= 1:
            raise ValueError("The last component cannot be removed from the model.")

        del self._components[component_idx]
        self._log_weights = np.delete(self._log_weights, component_idx)
        self._log_weights -= logsumexp(self._log_weights)
        self._cached_weights = None

    def pdf(self, X: ArrayLike) -> NDArray[float64]:
        """Probability Density Function of the mixture.

        The PDF is computed as the weighted sum of the PDFs of its
        components.

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
        component_pdfs = np.array([comp.pdf(X) for comp in self.components])
        return np.asarray(np.dot(self.weights, component_pdfs))

    def lpdf(self, X: ArrayLike) -> NDArray[float64]:
        """Logarithms of the Probability Density Function.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the LPDF.

        Returns
        -------
        NDArray[np.float64]
            The log-PDF values corresponding to each point in :attr:`X`.
        """

        X = np.atleast_1d(X)
        component_lpdfs = np.array([comp.lpdf(X) for comp in self.components])
        log_weights = self.log_weights
        log_terms = log_weights[:, np.newaxis] + component_lpdfs
        return logsumexp(log_terms, axis=0)  # type: ignore

    def loglikelihood(self, X: ArrayLike) -> float:
        """Log-likelihood of the complete data :attr:`X`.

        The log-likelihood is the sum of the log-PDF values for all data
        points.

        Parameters
        ----------
        X : ArrayLike
            The input data sample.

        Returns
        -------
        float
            The total log-likelihood value.
        """

        X = np.asarray(X, dtype=float64)
        return np.sum(self.lpdf(X))

    def generate(self, size: int) -> NDArray[float64]:
        """Generates random samples from the mixture model.

        First, a component is chosen based on the mixture weights. Then, a
        sample is drawn from the chosen component's distribution. This
        process is repeated `size` times.

        Parameters
        ----------
        size : int
            The number of random samples to generate.

        Returns
        -------
        NDArray[np.float64]
            A NumPy array containing the generated samples. Returns an
            empty array if :attr:`size` is not positive.
        """

        if size == 0:
            return np.array([])

        component_choices = np.random.choice(self.n_components, size=size, p=self.weights)

        counts = np.bincount(component_choices, minlength=self.n_components)

        samples_list = [self.components[i].generate(size=count) for i, count in enumerate(counts) if count > 0]

        samples = np.concatenate(samples_list)
        np.random.shuffle(samples)
        return samples

    def __getitem__(self, key: int):
        """Retrieves components by index.

        Parameters
        ----------
        key : int
            The index of the component.

        Returns
        -------
        ContinuousDistribution
            A single component of the mixture
        """

        return self.components[key]

    def __iter__(self):
        """Returns an iterator over the mixture components.

        This allows the `MixtureModel` instance to be used directly in
        loops, such as a `for` loop, to iterate over its components.

        Yields
        ------
        Iterator[ContinuousDistribution]
            An iterator that yields the components of the mixture model.
        """

        return iter(self.components)
