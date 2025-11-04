"""A module that provides the `MixtureModel` class, which allows for the creation,
manipulation, and analysis of a finite mixture of continuous probability
distributions."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Iterator, Sequence
from copy import copy
from typing import TYPE_CHECKING, Generic, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp, softmax

from ..typings import DType

if TYPE_CHECKING:
    from ..distributions import ContinuousDistribution


class MixtureModel(Generic[DType]):
    """Represents a finite mixture of continuous probability distributions.

    This class encapsulates a collection of distribution components and their
    corresponding weights. All components within the mixture are automatically
    converted to the specified `dtype` of the MixtureModel, ensuring
    computational consistency.

    Instances of this class can be compared for equality (``==``) and
    inequality (``!=``). Two models are considered equal if they have the
    same set of components and weights, regardless of the order in which
    components were added.

    Parameters
    ----------
    components : Sequence[ContinuousDistribution]
        A sequence of distribution objects that will form the mixture.
    weights : Optional[ArrayLike], optional
        An array of initial weights for the components. The weights must be
        positive and sum to 1. If None, components are assigned equal
        weights. Defaults to None.
    dtype : type[DType], optional
        The numpy data type used for internal calculations and
        output arrays (e.g., `np.float32` or `np.float64`).
        Defaults to `np.float64`.

    Attributes
    ----------
    components : tuple[ContinuousDistribution[DType], ...]
        A tuple of the distribution objects that form the mixture.
    n_components : int
        The number of components in the mixture.
    weights : NDArray[DType]
        A NumPy array of the normalized weights for each component. The sum
        of weights is always 1.
    log_weights : NDArray[DType]
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

    _dtype: type[DType]

    def __init__(
        self,
        components: Sequence["ContinuousDistribution"],
        weights: Optional[ArrayLike] = None,
        dtype: type[DType] = np.float64,  # type: ignore[assignment]
    ):
        n_components = len(components)
        if n_components == 0:
            raise ValueError("List of components cannot be an empty")

        self._dtype = dtype

        if weights is None:
            weights = np.full(n_components, 1.0 / n_components, dtype=self.dtype)
        else:
            weights = np.asarray(weights, dtype=self.dtype)
            self._validate_weights(n_components, weights)

        self._components = [comp.astype(self.dtype) for comp in components]
        self._log_weights = np.log(weights + self.dtype(1e-30))
        self._cached_weights: Optional[NDArray[DType]] = None

        self._sorted_pairs_cache: Optional[list[tuple[ContinuousDistribution[DType], DType]]] = None

    def _validate_weights(self, n_components: int, weights: NDArray[DType]):
        """Validates the component weights.

        Parameters
        ----------
        n_components : int
            The expected number of components.
        weights : NDArray[DType]
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

        if not np.isclose(np.sum(weights), self.dtype(1.0)):
            raise ValueError(f"Sum of the weights must be equal 1, but it equal {np.sum(weights)}.")

    @property
    def dtype(self) -> type[DType]:
        """type[DType]: The numpy data type of the mixture's outputs."""
        return self._dtype

    @property
    def n_components(self):
        """int: The number of components in the mixture model."""

        return len(self.components)

    @property
    def components(self):
        """tuple[ContinuousDistribution[DType], ...]: The components of the mixture."""

        return tuple(self._components)

    @property
    def weights(self) -> NDArray[DType]:
        """NDArray[DType]: The normalized weights of the components.

        The weights are computed from the log-weights using the softmax
        function and cached for efficiency.
        """

        if self._cached_weights is None:
            self._cached_weights = softmax(self._log_weights)

        return self._cached_weights  # type: ignore

    @property
    def log_weights(self) -> NDArray[DType]:
        """NDArray[DType]: The logarithm of the component weights."""

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

        new_log_weights = np.asarray(new_log_weights, dtype=self.dtype)

        if len(new_log_weights) != self.n_components:
            raise ValueError("The length of the new logit vector does not match the number of components.")
        self._log_weights = new_log_weights
        self._cached_weights = None
        self._sorted_pairs_cache = None

    def add_component(self, component: "ContinuousDistribution", weight: float):
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

        d_weight = self.dtype(weight)
        self._log_weights += np.log(self.dtype(1.0) - d_weight)
        new_log_weight = np.log(d_weight)
        self._log_weights = np.append(self._log_weights, new_log_weight)

        new_component = component.astype(self.dtype)
        self._components.append(new_component)
        self._cached_weights = None
        self._sorted_pairs_cache = None

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
        self._sorted_pairs_cache = None

    def pdf(self, X: ArrayLike) -> NDArray[DType]:
        """Probability Density Function of the mixture.

        The PDF is computed as the weighted sum of the PDFs of its
        components.

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
        component_pdfs = np.array([comp.pdf(X) for comp in self.components])
        return np.asarray(np.dot(self.weights, component_pdfs))

    def lpdf(self, X: ArrayLike) -> NDArray[DType]:
        """Logarithms of the Probability Density Function.

        Parameters
        ----------
        X : ArrayLike
            The input data points at which to evaluate the LPDF.

        Returns
        -------
        NDArray[DType]
            The log-PDF values corresponding to each point in :attr:`X`.
        """

        X = np.atleast_1d(X).astype(self.dtype)
        component_lpdfs = np.array([comp.lpdf(X) for comp in self.components])
        log_weights = self.log_weights
        log_terms = log_weights[:, np.newaxis] + component_lpdfs
        return logsumexp(log_terms, axis=0)  # type: ignore

    def loglikelihood(self, X: ArrayLike) -> DType:
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

        X = np.asarray(X, dtype=self.dtype)
        return np.sum(self.lpdf(X))

    def generate(self, size: int) -> NDArray[DType]:
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
        NDArray[DType]
            A NumPy array containing the generated samples. Returns an
            empty array if :attr:`size` is not positive.
        """

        if size == 0:
            return np.array([], dtype=self.dtype)

        component_choices = np.random.choice(self.n_components, size=size, p=self.weights)

        counts = np.bincount(component_choices, minlength=self.n_components)

        samples_list = [self.components[i].generate(size=count) for i, count in enumerate(counts) if count > 0]

        samples = np.concatenate(samples_list)
        np.random.shuffle(samples)
        return samples

    def __getitem__(self, key: int) -> "ContinuousDistribution[DType]":
        """Retrieves components by index.

        Parameters
        ----------
        key : int
            The index of the component.

        Returns
        -------
        ContinuousDistribution[DType]
            A single component of the mixture
        """

        return self.components[key]

    def __iter__(self) -> Iterator["ContinuousDistribution[DType]"]:
        """Returns an iterator over the mixture components.

        This allows the `MixtureModel` instance to be used directly in
        loops, such as a `for` loop, to iterate over its components.

        Yields
        ------
        Iterator[ContinuousDistribution[DType]
            An iterator that yields the components of the mixture model.
        """

        return iter(self.components)

    def __copy__(self) -> "MixtureModel[DType]":
        """Creates a copy of the mixture model instance.

        Returns
        -------
        MixtureModel[DType]
            A new instance of the distribution, identical to the original.
        """

        copied_components = [copy(component) for component in self._components]
        new_mixture = MixtureModel(components=copied_components, weights=self.weights.copy(), dtype=self.dtype)
        return new_mixture

    def _get_sorted_pairs(self, for_hashing: bool = False) -> list[tuple["ContinuousDistribution[DType]", DType]]:
        """Internal helper to get component-weight pairs, sorted by component hash."""

        if self._sorted_pairs_cache is None or for_hashing:
            weights_to_use = self.weights
            if for_hashing:
                weights_to_use = np.round(weights_to_use, 8)

            pairs = sorted(zip(self.components, weights_to_use), key=lambda p: hash(p[0]))
            if not for_hashing:
                self._sorted_pairs_cache = pairs
            return pairs
        return self._sorted_pairs_cache

    def __eq__(self, other: object) -> bool:
        """Checks if two mixture models are equal.

        Two mixture models are considered equal if they have the same number of
        components and the same set of (component, weight) pairs.

        Parameters
        ----------
        other : object
            The object to compare against.

        Returns
        -------
        bool
            True if the mixture models are equal, False otherwise.
        """

        if not isinstance(other, MixtureModel):
            return NotImplemented

        if self.dtype != other.dtype or self.n_components != other.n_components:
            return False

        self_pairs = self._get_sorted_pairs()
        other_pairs = other._get_sorted_pairs()

        for (self_comp, self_weight), (other_comp, other_weight) in zip(self_pairs, other_pairs):
            if self_comp != other_comp or not np.isclose(self_weight, other_weight):
                return False

        return True

    def __hash__(self) -> int:
        """Computes a hash for the mixture model.

        The hash is computed based on the set of (component, weight) pairs.

        Returns
        -------
        int
            The hash value of the mixture model.
        """

        sorted_pairs_for_hash = self._get_sorted_pairs(for_hashing=True)
        return hash((self.dtype, tuple(sorted_pairs_for_hash)))
