"""
Adapter for using pysatl-core distributions in pysatl-mpest.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike
from pysatl_core.families import ParametricFamily
from pysatl_core.types import CharacteristicName

from pysatl_mpest.distributions.continuous_dist import ContinuousDistribution
from pysatl_mpest.typings import FloatArray


class CoreDistributionAdapter(ContinuousDistribution):
    """
    Adapter bridging `pysatl-core` ParametricFamily to `pysatl-mpest` distributions.

    This class encapsulates a mathematical family and dynamically binds its parameters
    to provide the `ContinuousDistribution` interface. It inherently supports freezing
    parameters by projecting the gradient evaluations onto the subspace of free parameters.

    Parameters
    ----------
    core_family : ParametricFamily
        The core mathematical family to adapt (e.g., Normal).
    parametrization_name : str
        The name of the parametrization to use (e.g., "meanStd").
    **kwargs : Any
        Initial values for the parameters defined by the parametrization.
    """

    def __init__(self, core_family: ParametricFamily, parametrization_name: str, **kwargs: Any) -> None:
        super().__init__()
        self.core_family = core_family
        self.parametrization_name = parametrization_name
        self.core_dist = core_family.distribution(parametrization_name, **kwargs)

    @property
    def name(self) -> str:
        """str: The string representation of the distribution family name."""
        return str(self.core_family.name)

    @property
    def params(self) -> set[str]:
        """set[str]: The set of parameter names available in this distribution."""
        return set(self.core_dist.parametrization.parameters.keys())

    def __getattr__(self, name: str) -> Any:
        if name in self.params:
            return self.core_dist.parametrization.parameters[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __copy__(self) -> "CoreDistributionAdapter":
        """
        Create a shallow copy of the adapter instance.

        Returns
        -------
        CoreDistributionAdapter
            A new adapter instance with identical parameters and freeze states.
        """
        new_instance = CoreDistributionAdapter(
            self.core_family,
            self.parametrization_name,
            **self.core_dist.parametrization.parameters,
        )
        new_instance._fixed_params = self._fixed_params.copy()
        return new_instance

    def set_params_from_vector(self, param_names: Sequence[str], vector: Sequence[np.float64 | float]) -> None:
        """
        Update the free parameters from a flat vector sequence.

        Since the core structures are immutable, this recreates the internal
        core distribution instance.

        Parameters
        ----------
        param_names : Sequence[str]
            The names of the parameters to update.
        vector : Sequence[np.float64 | float]
            The new values for the corresponding parameters.
        """
        current_params = self.core_dist.parametrization.parameters.copy()
        for name, value in zip(param_names, vector):
            current_params[name] = float(value)

        self.core_dist = self.core_family.distribution(self.parametrization_name, **current_params)

    def _calc_characteristic(self, char_name: CharacteristicName, array: ArrayLike) -> np.float64 | FloatArray:
        """Helper to calculate a characteristic and cast the output."""
        arr = np.asarray(array)
        is_scalar = arr.ndim == 0
        res = np.asarray(
            self.core_dist.calculate_characteristic(char_name, arr),
            dtype=np.float64,
        )
        if is_scalar:
            return np.float64(res.item())
        return cast(FloatArray, res)

    def pdf(self, X: ArrayLike) -> np.float64 | FloatArray:
        """
        Evaluate the probability density function.

        Parameters
        ----------
        X : ArrayLike
            Points at which to evaluate the PDF.

        Returns
        -------
        np.float64 | FloatArray
            The computed PDF values.
        """

        return self._calc_characteristic(CharacteristicName.PDF, X)

    def lpdf(self, X: ArrayLike) -> np.float64 | FloatArray:
        """
        Evaluate the log probability density function.

        Parameters
        ----------
        X : ArrayLike
            Points at which to evaluate the log-PDF.

        Returns
        -------
        np.float64 | FloatArray
            The computed log-PDF values.
        """
        return self._calc_characteristic(CharacteristicName.LPDF, X)

    def ppf(self, P: ArrayLike) -> np.float64 | FloatArray:
        """
        Evaluate the percent point function (inverse CDF).

        Parameters
        ----------
        P : ArrayLike
            Probabilities at which to evaluate the PPF.

        Returns
        -------
        np.float64 | FloatArray
            The computed PPF values.
        """
        return self._calc_characteristic(CharacteristicName.PPF, P)

    def log_gradients(self, X: ArrayLike) -> FloatArray:
        """
        Calculate the gradient of the log-pdf with respect to free parameters.

        This projects the underlying parametric family into a partial family using
        the fixed parameters, and computes the score only for the parameters that
        are currently being optimized.

        Parameters
        ----------
        X : ArrayLike
            Points at which to evaluate the log-pdf gradients.

        Returns
        -------
        FloatArray
            Gradients of the log-pdf with respect to the free parameters.
        """

        X = np.asarray(X, dtype=np.float64)
        is_scalar = X.ndim == 0

        if len(self.params_to_optimize) == 0:
            if is_scalar:
                return np.empty(shape=(0,))
            else:
                return np.empty(shape=(len(X), 0))

        fixed_kwargs = {name: getattr(self, name) for name in self._fixed_params}

        viewed_family = self.core_family.view(**fixed_kwargs)
        free_kwargs = {name: getattr(self, name) for name in self.params_to_optimize}

        p_viewed = viewed_family.parametrizations[self.parametrization_name](**free_kwargs)

        score_val = viewed_family.score(p_viewed, X)

        if is_scalar:
            return score_val[0]

        return cast(FloatArray, np.asarray(score_val, dtype=np.float64))

    def generate(self, size: int | tuple[int, ...] | None = None) -> np.float64 | FloatArray:
        """
        Generate random samples from the distribution.

        Parameters
        ----------
        size : int | tuple[int, ...] | None, default=None
            The shape of the generated samples. If None, returns a scalar.

        Returns
        -------
        np.float64 | FloatArray
            The generated random samples.
        """
        if size is None:
            return np.float64(self.core_dist.sample(n=1)[0])

        if isinstance(size, int):
            return cast(FloatArray, np.asarray(self.core_dist.sample(n=size), dtype=np.float64))

        total_elements = int(np.prod(size))
        samples = self.core_dist.sample(n=total_elements)
        return cast(FloatArray, np.asarray(samples.reshape(size), dtype=np.float64))
