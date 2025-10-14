"""Tests for MixtureModel class"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from rework_pysatl_mpest.core import MixtureModel
from rework_pysatl_mpest.distributions import ContinuousDistribution, Exponential


@pytest.fixture
def exp_components() -> tuple[Exponential, Exponential]:
    """Provides a tuple of two distinct Exponential distribution components."""

    return Exponential(loc=0.0, rate=1.0), Exponential(loc=5.0, rate=2.0)


@pytest.fixture
def mixture_model(exp_components: tuple[Exponential, Exponential]) -> MixtureModel:
    """Provides a standard MixtureModel instance with two components and equal weights."""

    return MixtureModel(components=exp_components, weights=[0.5, 0.5])


class TestMixtureModelInitialization:
    """Tests for the __init__ method of MixtureModel."""

    @pytest.mark.parametrize(
        "components_seq",
        [
            (Exponential(loc=0, rate=1), Exponential(loc=1, rate=2)),
            [Exponential(loc=0, rate=1), Exponential(loc=1, rate=2)],
        ],
    )
    def test_init_with_valid_component_sequences(self, components_seq: Sequence[ContinuousDistribution]):
        """Tests that MixtureModel can be initialized with different sequence types."""

        model = MixtureModel(components=components_seq)
        expected_n_components = 2
        assert model.n_components == expected_n_components
        assert isinstance(model.components, tuple)

    def test_init_with_equal_weights_by_default(self, exp_components: tuple[Exponential, Exponential]):
        """Tests that weights are distributed equally when not provided."""

        model = MixtureModel(components=exp_components)
        expected_n_components = 2
        assert model.n_components == expected_n_components
        np.testing.assert_allclose(model.weights, [0.5, 0.5])

    def test_init_with_specified_weights(self, exp_components: tuple[Exponential, Exponential]):
        """Tests initialization with correctly specified weights."""

        weights = [0.3, 0.7]
        model = MixtureModel(components=exp_components, weights=weights)
        expected_n_components = 2
        assert model.n_components == expected_n_components
        np.testing.assert_allclose(model.weights, weights)
        np.testing.assert_allclose(np.exp(model.log_weights), weights, atol=1e-9)

    @pytest.mark.parametrize(
        "invalid_weights, error_msg",
        [
            ([0.5, 0.5, 0.5], "Components number \\(2\\) must be equal to weights number \\(3\\)."),
            ([-0.2, 1.2], "Weights must be positive."),
            ([0.4, 0.4], "Sum of the weights must be equal 1, but it equal 0.8."),
        ],
    )
    def test_init_with_invalid_weights_raises_value_error(
        self, exp_components: tuple[Exponential, Exponential], invalid_weights: list[float], error_msg: str
    ):
        """Tests that initialization with invalid weights raises a ValueError."""

        with pytest.raises(ValueError, match=error_msg):
            MixtureModel(components=exp_components, weights=invalid_weights)

    def test_init_with_empty_components_raises_value_error(self):
        """Tests that initialization with an empty component list raises a ValueError."""

        with pytest.raises(ValueError, match="List of components cannot be an empty"):
            MixtureModel(components=[])


class TestMixtureModelProperties:
    """Tests for the properties of the MixtureModel class."""

    def test_n_components_property(self, mixture_model: MixtureModel):
        """Tests that n_components returns the correct number of components."""

        expected_n_components = 2
        assert mixture_model.n_components == expected_n_components

    def test_components_property_is_immutable_tuple(self, mixture_model: MixtureModel):
        """Tests that the 'components' property returns a tuple and is immutable."""

        components = mixture_model.components
        assert isinstance(components, tuple)
        with pytest.raises(TypeError):
            components[0] = Exponential(0, 1)  # type: ignore

    def test_weights_caching(self, exp_components: tuple[Exponential, Exponential]):
        """Tests the caching mechanism of the 'weights' property."""

        model = MixtureModel(components=exp_components)
        assert model._cached_weights is None

        first_access_weights = model.weights
        assert model._cached_weights is not None
        np.testing.assert_array_equal(first_access_weights, model._cached_weights)

        second_access_weights = model.weights
        assert id(first_access_weights) == id(second_access_weights)

    def test_log_weights_setter_and_cache_invalidation(self, mixture_model: MixtureModel):
        """Tests setting log_weights and verifies that it invalidates the weights cache."""

        old_weights = mixture_model.weights
        assert mixture_model._cached_weights is not None

        new_log_weights = np.log([0.3, 0.7])
        mixture_model.log_weights = new_log_weights

        assert mixture_model._cached_weights is None
        np.testing.assert_allclose(mixture_model.log_weights, new_log_weights)

        new_weights = mixture_model.weights
        np.testing.assert_allclose(new_weights, [0.3, 0.7])
        assert id(old_weights) != id(new_weights)
        assert np.isclose(np.sum(new_weights), 1.0)

    def test_log_weights_setter_with_invalid_shape_raises_error(self, mixture_model: MixtureModel):
        """Tests that setting log_weights with an incorrect shape raises a ValueError."""

        with pytest.raises(ValueError, match="The length of the new logit vector does not match"):
            mixture_model.log_weights = np.log([0.1, 0.2, 0.7])


class TestMixtureModelModification:
    """Tests for methods that modify the MixtureModel instance."""

    def test_add_component(self, mixture_model: MixtureModel):
        """Tests adding a new component and verifies weight preservation and renormalization."""

        old_weights = mixture_model.weights.copy()
        new_component = Exponential(loc=10, rate=3)
        new_weight = 0.4

        mixture_model.add_component(new_component, weight=new_weight)

        expected_n_components = 3

        assert mixture_model.n_components == expected_n_components
        assert mixture_model.components[-1] == new_component

        assert np.isclose(mixture_model.weights[-1], new_weight)

        expected_old_weights_scaled = old_weights * (1 - new_weight)
        np.testing.assert_allclose(mixture_model.weights[:-1], expected_old_weights_scaled)

        assert np.isclose(np.sum(mixture_model.weights), 1.0)

    @pytest.mark.parametrize("invalid_weight", [-0.1, 0, 1, 1.1])
    def test_add_component_with_invalid_weight_raises_error(self, mixture_model: MixtureModel, invalid_weight: float):
        """Tests that adding a component with a weight outside (0, 1) raises ValueError."""

        with pytest.raises(ValueError, match="The weight of the new component must be in the range"):
            mixture_model.add_component(Exponential(10, 3), weight=invalid_weight)

    def test_remove_component(self):
        """Tests removing a component and verifies weight renormalization."""

        components = [Exponential(0, 1), Exponential(5, 2), Exponential(10, 3)]
        model = MixtureModel(components=components, weights=[0.2, 0.5, 0.3])

        model.remove_component(1)

        expected_n_components = 2

        assert model.n_components == expected_n_components

        comp1, comp2 = model.components
        assert isinstance(comp1, Exponential)
        assert isinstance(comp2, Exponential)

        assert model.components == (components[0], components[2])

        expected_comp1_loc = 0
        expected_comp2_loc = 10

        assert comp1.loc == expected_comp1_loc
        assert comp2.loc == expected_comp2_loc

        expected_weights = np.array([0.2, 0.3]) / (0.2 + 0.3)
        np.testing.assert_allclose(model.weights, expected_weights)
        assert np.isclose(np.sum(model.weights), 1.0)

    def test_remove_last_component_raises_error(self):
        """Tests that attempting to remove the last component raises a ValueError."""

        model = MixtureModel(components=[Exponential(0, 1)])
        with pytest.raises(ValueError, match="The last component cannot be removed"):
            model.remove_component(0)

    def test_remove_component_with_invalid_index_raises_error(self, mixture_model: MixtureModel):
        """Tests that removing a component with an out-of-bounds index raises IndexError."""

        with pytest.raises(IndexError, match="Index 2 out of range"):
            mixture_model.remove_component(2)


class TestMixtureModelCalculations:
    """Tests for calculation methods like pdf, lpdf, etc."""

    @pytest.mark.parametrize("X", [1.5, [1.5], np.array([1.0, 1.5, 6.0])])
    def test_pdf_calculation(self, mixture_model: MixtureModel, X):
        """Tests the PDF calculation against the definition."""

        c1, c2 = mixture_model.components
        w1, w2 = mixture_model.weights

        expected_pdf = w1 * c1.pdf(X) + w2 * c2.pdf(X)
        calculated_pdf = mixture_model.pdf(X)

        assert isinstance(calculated_pdf, np.ndarray)
        np.testing.assert_allclose(calculated_pdf, expected_pdf)

    @pytest.mark.parametrize("X", [1.5, [1.5], np.array([1.0, 1.5, 6.0])])
    def test_lpdf_calculation(self, mixture_model: MixtureModel, X):
        """Tests the LPDF calculation against the definition."""

        expected_lpdf = np.log(mixture_model.pdf(X))
        calculated_lpdf = mixture_model.lpdf(X)

        assert isinstance(calculated_lpdf, np.ndarray)
        np.testing.assert_allclose(calculated_lpdf, expected_lpdf)

    def test_loglikelihood_calculation(self, mixture_model: MixtureModel):
        """Tests that loglikelihood is the sum of LPDF values."""

        X = np.array([1.0, 1.5, 6.0])
        expected_loglikelihood = np.sum(mixture_model.lpdf(X))
        calculated_loglikelihood = mixture_model.loglikelihood(X)

        assert isinstance(calculated_loglikelihood, float)
        assert np.isclose(calculated_loglikelihood, expected_loglikelihood)


class TestMixtureModelGenerate:
    """Statistical tests for the `generate` method."""

    def test_generate_returns_correct_size(self, mixture_model: MixtureModel):
        """Tests that generate returns an array of the requested size."""

        size = 100

        assert len(mixture_model.generate(size=size)) == size
        assert isinstance(mixture_model.generate(size=size), np.ndarray)

    def test_generate_with_size_zero(self, mixture_model):
        """Tests that generating with size = 0 returns an empty array."""

        assert len(mixture_model.generate(0)) == 0

    @pytest.mark.parametrize("size", [-1, -10])
    def test_generate_with_negative_size(self, mixture_model: MixtureModel, size: int):
        """Tests that generating with size < raises ValueError."""

        with pytest.raises(ValueError):
            mixture_model.generate(size=size)

    @given(seed=st.integers(0, 2**32 - 1))
    def test_generate_statistical_properties(self, seed):
        """
        Performs a statistical test on the generated samples.
        1. Checks if the proportion of samples from each component matches the weights.
        2. Checks if the sample mean matches the theoretical mean of the mixture.
        """

        np.random.seed(seed)

        c1 = Exponential(loc=0, rate=1.0)
        c2 = Exponential(loc=10, rate=0.5)
        components = [c1, c2]
        weights = np.array([0.3, 0.7])
        model = MixtureModel(components=components, weights=weights)

        e_x1 = c1.loc + 1 / c1.rate
        e_x2 = c2.loc + 1 / c2.rate
        theoretical_mean = weights[0] * e_x1 + weights[1] * e_x2

        n_samples = 20000
        samples = model.generate(size=n_samples)

        assert len(samples) == n_samples
        assert np.mean(samples) == pytest.approx(theoretical_mean, rel=0.1)

        midpoint_between_means = (e_x1 + e_x2) / 2
        samples_from_c1 = samples[samples < midpoint_between_means]
        proportion_c1 = len(samples_from_c1) / n_samples
        assert proportion_c1 == pytest.approx(weights[0], abs=0.05)


class TestMixtureModelDunderMethods:
    """Tests for special (dunder) methods of MixtureModel."""

    @pytest.mark.parametrize(
        "index, expected_component_index",
        [
            (0, 0),
            (1, 1),
            (-1, 1),  # Test negative indexing
            (-2, 0),
        ],
    )
    def test_getitem_retrieves_correct_component(
        self,
        mixture_model: MixtureModel,
        exp_components: tuple[Exponential, ...],
        index: int,
        expected_component_index: int,
    ):
        """Tests that __getitem__ retrieves the correct component by index."""

        assert mixture_model[index] is exp_components[expected_component_index]
        assert mixture_model[index] == exp_components[expected_component_index]

    def test_getitem_out_of_bounds_raises_index_error(self, mixture_model: MixtureModel):
        """Tests that accessing an out-of-bounds index raises an IndexError."""

        with pytest.raises(IndexError):
            _ = mixture_model[2]
        with pytest.raises(IndexError):
            _ = mixture_model[-3]

    @pytest.mark.parametrize("invalid_key", ["a_string", 1.5, (0, 1)])
    def test_getitem_with_invalid_key_type_raises_type_error(self, mixture_model: MixtureModel, invalid_key: Any):
        """Tests that using a non-integer key (that is not a slice) raises a TypeError."""

        with pytest.raises(TypeError):
            _ = mixture_model[invalid_key]

    def test_iter_yields_correct_components_in_order(
        self, mixture_model: MixtureModel, exp_components: tuple[Exponential, ...]
    ):
        """Tests that iterating over the model yields all components in the correct order."""

        iterated_components = list(mixture_model)
        assert iterated_components == list(exp_components)
        assert all(comp_iter is comp_orig for comp_iter, comp_orig in zip(iterated_components, exp_components))
        assert all(comp_iter == comp_orig for comp_iter, comp_orig in zip(iterated_components, exp_components))

    def test_iter_is_reusable(self, mixture_model: MixtureModel):
        """Tests that the model can be iterated over multiple times."""

        first_pass = list(mixture_model)
        second_pass = list(mixture_model)

        assert len(first_pass) == mixture_model.n_components
        assert first_pass is not second_pass  # The lists are different objects
        assert first_pass == second_pass  # But their contents are identical


class TestMixtureModelComparison:
    """Tests the __eq__ and __hash__ methods for MixtureModel."""

    def test_eq_identical_models(self):
        """Tests that two identical models are equal."""

        c = [Exponential(0, 1), Exponential(10, 2)]
        m1 = MixtureModel(components=c, weights=[0.4, 0.6])
        m2 = MixtureModel(components=c, weights=[0.4, 0.6])
        assert m1 == m2

    def test_eq_order_insensitivity(self):
        """Tests that models with the same components/weights in a different order are equal."""

        c1 = [Exponential(0, 1), Exponential(10, 2)]
        w1 = [0.4, 0.6]
        m1 = MixtureModel(components=c1, weights=w1)

        c2 = [Exponential(10, 2), Exponential(0, 1)]
        w2 = [0.6, 0.4]
        m2 = MixtureModel(components=c2, weights=w2)

        assert m1 == m2

    def test_neq_different_weights(self):
        """Tests that models with different weights are not equal."""

        c = [Exponential(0, 1), Exponential(10, 2)]
        m1 = MixtureModel(components=c, weights=[0.4, 0.6])
        m2 = MixtureModel(components=c, weights=[0.41, 0.59])
        assert m1 != m2

    def test_neq_different_components(self):
        """Tests that models with different components are not equal."""

        c1 = [Exponential(0, 1), Exponential(10, 2)]
        c2 = [Exponential(0, 1), Exponential(99, 2)]
        m1 = MixtureModel(components=c1, weights=[0.4, 0.6])
        m2 = MixtureModel(components=c2, weights=[0.4, 0.6])
        assert m1 != m2

    def test_neq_different_n_components(self):
        """Tests that models with a different number of components are not equal."""

        c1 = [Exponential(0, 1), Exponential(10, 2)]
        c2 = [Exponential(0, 1)]
        m1 = MixtureModel(components=c1, weights=[0.4, 0.6])
        m2 = MixtureModel(components=c2, weights=[1.0])
        assert m1 != m2

    def test_hash_consistency(self):
        """Tests that equal models produce the same hash."""

        m1 = MixtureModel(
            components=[Exponential(0, 1), Exponential(10, 2)],
            weights=[0.4, 0.6],
        )
        m2 = MixtureModel(
            components=[Exponential(10, 2), Exponential(0, 1)],
            weights=[0.6, 0.4],
        )
        assert m1 == m2
        assert hash(m1) == hash(m2)

    def test_hash_difference(self):
        """Tests that non-equal models are likely to have different hashes."""

        m1 = MixtureModel(
            components=[Exponential(0, 1), Exponential(10, 2)],
            weights=[0.4, 0.6],
        )
        m2 = MixtureModel(
            components=[Exponential(0, 1), Exponential(10, 2)],
            weights=[0.5, 0.5],
        )
        assert m1 != m2
        assert hash(m1) != hash(m2)
