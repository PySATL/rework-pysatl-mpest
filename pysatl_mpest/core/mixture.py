"""A module that provides the `MixtureModel` class, which allows for the creation,
manipulation, and analysis of a finite mixture of continuous probability
distributions."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.transformations.operations.distributions.mixture import FiniteMixtureDistribution, finite_mixture
from pysatl_core.types import CharacteristicName
from scipy.special import logsumexp, softmax

from ..typings import FloatArray, Scalar, UnivariateFloatArray
from . import _components


class MixtureModel:
    """Represents a finite mixture of continuous probability distributions.

    This class encapsulates a collection of distribution components and their
    corresponding weights.

    Instances of this class can be compared for equality (``==``) and
    inequality (``!=``). Two models are considered equal if they have the
    same set of components and weights, regardless of the order in which
    components were added.

    Parameters
    ----------
    components : Sequence[ParametricFamilyDistribution]
        A sequence of distribution objects that will form the mixture.
    weights : ArrayLike | None, optional
        An array of initial weights for the components. The weights must be
        positive and sum to 1. If None, components are assigned equal
        weights. Defaults to None.

    Attributes
    ----------
    components : tuple[ParametricFamilyDistribution, ...]
        A tuple of the distribution objects that form the mixture.
    n_components : int
        The number of components in the mixture.
    weights : UnivariateFloatArray
        A NumPy array of the normalized weights for each component. The sum
        of weights is always 1.
    log_weights : UnivariateFloatArray
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
        component_lpdf_matrix
        loglikelihood
        generate
    """

    def __init__(
        self,
        components: Sequence[ParametricFamilyDistribution],
        weights: ArrayLike | None = None,
    ):
        n_components = len(components)
        if n_components == 0:
            raise ValueError("List of components cannot be empty")

        if weights is None:
            weights = np.full(n_components, 1.0 / n_components, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            self._validate_weights(n_components, weights)

        self._components = list(components)
        self._log_weights = np.log(weights + np.finfo(np.float64).tiny)
        self._cached_weights: UnivariateFloatArray | None = None

        self._sorted_pairs_cache: list[tuple[ParametricFamilyDistribution, float]] | None = None
        self._core_mixture_cache: FiniteMixtureDistribution | None = None

    @property
    def n_components(self):
        """int: The number of components in the mixture model."""

        return len(self.components)

    @property
    def components(self):
        """tuple[ParametricFamilyDistribution, ...]: The components of the mixture."""

        return tuple(self._components)

    @property
    def weights(self) -> UnivariateFloatArray:
        """UnivariateFloatArray: The normalized weights of the components.

        The weights are computed from the log-weights using the softmax
        function and cached for efficiency.
        """

        if self._cached_weights is None:
            self._cached_weights = softmax(self._log_weights)

        return self._cached_weights  # type: ignore

    @property
    def log_weights(self) -> UnivariateFloatArray:
        """UnivariateFloatArray: The logarithm of the component weights."""

        return self._log_weights

    @log_weights.setter
    def log_weights(self, new_log_weights: ArrayLike) -> None:
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

        new_log_weights = np.asarray(new_log_weights, dtype=np.float64)

        if len(new_log_weights) != self.n_components:
            raise ValueError("The length of the new logit vector does not match the number of components.")
        self._log_weights = new_log_weights
        self._invalidate_cache()

    def add_component(self, component: ParametricFamilyDistribution, weight: Scalar) -> None:
        """Adds a new component to the mix, preserving the proportions of the existing component weights.

        If :attr:`weight` is specified for the new component, the old component
        weights are multiplied by `(1 - weight)`.

        Parameters
        ----------
        component : ParametricFamilyDistribution
            The distribution component to add.
        weight : Scalar
            The weight for the new component, a number in the range (0, 1).

        Raises
        ------
        ValueError
            If the specified :attr:`weight` is outside the range (0, 1).
        """

        if not (0 < weight < 1):
            raise ValueError("The weight of the new component must be in the range (0, 1).")

        d_weight = float(weight)
        self._log_weights += np.log(1.0 - d_weight)
        new_log_weight = np.log(d_weight)
        self._log_weights = np.append(self._log_weights, new_log_weight)

        self._components.append(component)
        self._invalidate_cache()

    def remove_component(self, component_idx: int) -> None:
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
        self._invalidate_cache()

    def to_core_mixture(self) -> FiniteMixtureDistribution:
        """Converts the model to a pysatl_core FiniteMixtureDistribution.

        Returns
        -------
        FiniteMixtureDistribution
            The corresponding finite mixture distribution object.
        """

        weighted_components = [(float(w), comp) for w, comp in zip(self.weights, self.components)]
        return finite_mixture(weighted_components)

    @property
    def _core_mixture(self) -> FiniteMixtureDistribution:
        """Lazy instantiation and caching of the pysatl-core FiniteMixtureDistribution."""

        if self._core_mixture_cache is None:
            self._core_mixture_cache = self.to_core_mixture()
        return self._core_mixture_cache

    def pdf(self, X: ArrayLike) -> np.float64 | FloatArray:
        """Probability Density Function of the mixture.

        The PDF is computed as the weighted sum of the PDFs of its
        components.

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

        is_scalar = np.isscalar(X)
        X = np.asarray(X, dtype=np.float64)
        result = self._core_mixture.calculate_characteristic(CharacteristicName.PDF, X)
        if is_scalar:
            return np.float64(result[()])
        return np.asarray(result, dtype=np.float64)

    def lpdf(self, X: ArrayLike) -> np.float64 | FloatArray:
        """Logarithms of the Probability Density Function.

        The log-density is computed as ``logsumexp(log_weights + component log-densities)``,
        which stays finite where the density itself underflows to zero.

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
        result = logsumexp(self.component_lpdf_matrix(X) + self.log_weights, axis=-1)

        if is_scalar:
            return np.float64(result[()])
        return np.asarray(result, dtype=np.float64)

    def component_lpdf_matrix(self, X: ArrayLike) -> FloatArray:
        """Log-densities of every component, without the mixture weights.

        Parameters
        ----------
        X : ArrayLike
            The input data points.

        Returns
        -------
        FloatArray
            An array of shape ``(*X.shape, n_components)`` whose last axis indexes
            the components. For a sample of ``n`` points this is an ``(n, n_components)`` matrix.
        """

        X = np.asarray(X, dtype=np.float64)
        return np.stack([_components.lpdf(component, X) for component in self.components], axis=-1)

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

        X = np.asarray(X, dtype=np.float64)
        return np.sum(self.lpdf(X))

    def generate(
        self,
        size: int | tuple[int, ...] | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> np.float64 | FloatArray:
        """Generates random samples from the mixture model.

        First, a component is chosen based on the mixture weights. Then, a
        sample is drawn from the chosen component's distribution. This
        process is repeated `size` times.

        Parameters
        ----------
        size : int | tuple[int, ...] | None, optional
            Defining number of random variates.
            - If None (default), returns a single scalar.
            - If int, returns a 1D array of that length.
            - If tuple, returns an array of that shape.
        random_state : int | np.random.Generator | None, optional
            A seed or random number generator to use for reproducible output.

        Returns
        -------
        np.float64 | FloatArray
            A NumPy array containing the generated samples. Returns an
            empty array if :attr:`size` is not positive.
        """

        target_shape: tuple[int, ...]
        if size is None:
            n_samples = 1
            target_shape = ()
        else:
            target_shape = (size,) if isinstance(size, int) else size
            n_samples = int(np.prod(target_shape))
            if n_samples == 0:
                return np.empty(target_shape, dtype=np.float64)

        rng = np.random.default_rng(random_state)
        component_choices = rng.choice(self.n_components, size=n_samples, p=self.weights)
        counts = np.bincount(component_choices, minlength=self.n_components)

        samples_list = [
            np.asarray(self.components[i].sample(n=int(count), random_state=rng), np.float64).ravel()
            for i, count in enumerate(counts)
            if count > 0
        ]

        samples = np.concatenate(samples_list)
        rng.shuffle(samples)

        if size is None:
            return samples[0]
        return samples.reshape(target_shape)

    def __getitem__(self, key: int) -> ParametricFamilyDistribution:
        """Retrieves components by index.

        Parameters
        ----------
        key : int
            The index of the component.

        Returns
        -------
        ParametricFamilyDistribution
            A single component of the mixture
        """

        return self.components[key]

    def __setitem__(self, key: int, value: ParametricFamilyDistribution) -> None:
        """Sets or replaces a component at a specific index.

        Parameters
        ----------
        key : int
            The index of the component to replace.
        value : ParametricFamilyDistribution
            The new distribution component to set.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if not (0 <= key <= self.n_components - 1):
            raise IndexError(f"Index of component must be between 0 and {self.n_components - 1}")

        self._components[key] = value
        self._invalidate_cache()

    def __iter__(self) -> Iterator[ParametricFamilyDistribution]:
        """Returns an iterator over the mixture components.

        This allows the `MixtureModel` instance to be used directly in
        loops, such as a `for` loop, to iterate over its components.

        Returns
        -------
        Iterator[ParametricFamilyDistribution]
            An iterator that yields the components of the mixture model.
        """

        return iter(self.components)

    def __copy__(self) -> "MixtureModel":
        """Creates a copy of the mixture model instance.

        Returns
        -------
        MixtureModel
            A new instance of the distribution, identical to the original.
        """
        return MixtureModel(components=list(self._components), weights=self.weights.copy())

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

        self_pairs = self._get_sorted_pairs()
        other_pairs = other._get_sorted_pairs()

        for (self_comp, self_weight), (other_comp, other_weight) in zip(self_pairs, other_pairs):
            if not self._comps_are_equal(self_comp, other_comp) or not np.isclose(self_weight, other_weight):
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

        decimals = np.finfo(np.float64).precision
        sorted_pairs = self._get_sorted_pairs()
        hashable_pairs = tuple((self._get_comp_hash(comp), np.round(weight, decimals)) for comp, weight in sorted_pairs)
        return hash(hashable_pairs)

    @staticmethod
    def _get_comp_hash(comp: ParametricFamilyDistribution) -> int:
        return hash((comp.family_name, comp.parametrization_name, comp.parametrization))

    @staticmethod
    def _comps_are_equal(comp1: ParametricFamilyDistribution, comp2: ParametricFamilyDistribution) -> bool:
        return (
            comp1.family_name == comp2.family_name
            and comp1.parametrization_name == comp2.parametrization_name
            and comp1.parametrization == comp2.parametrization
        )

    def _invalidate_cache(self) -> None:
        self._cached_weights = None
        self._sorted_pairs_cache = None
        self._core_mixture_cache = None

    def _validate_weights(self, n_components: int, weights: UnivariateFloatArray) -> None:
        """Validates the component weights.

        Parameters
        ----------
        n_components : int
            The expected number of components.
        weights : UnivariateFloatArray
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

    def _get_sorted_pairs(self) -> list[tuple[ParametricFamilyDistribution, float]]:
        """Internal helper to get component-weight pairs, sorted by component hash."""
        if self._sorted_pairs_cache is None:
            pairs = [(comp, float(w)) for comp, w in zip(self.components, self.weights)]
            self._sorted_pairs_cache = sorted(pairs, key=lambda p: self._get_comp_hash(p[0]))
        return self._sorted_pairs_cache
