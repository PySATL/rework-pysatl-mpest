"""Tests for MixtureModel class"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from copy import copy
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from rework_pysatl_mpest.core import MixtureModel
from rework_pysatl_mpest.distributions import ContinuousDistribution, Exponential

DTYPES_TO_TEST = [np.float16, np.float32, np.float64]


@pytest.fixture
def exp_components() -> tuple[Exponential, Exponential]:
    """Provides a tuple of two distinct Exponential distribution components."""

    return Exponential(loc=0.0, rate=1.0), Exponential(loc=5.0, rate=2.0)


@pytest.fixture(params=DTYPES_TO_TEST)
def mixture_model(exp_components: tuple[Exponential, Exponential], request) -> MixtureModel:
    """Provides a standard MixtureModel instance with two components and equal weights."""

    dtype = request.param
    return MixtureModel(components=exp_components, weights=[0.5, 0.5], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestMixtureModelInitialization:
    """Tests for the __init__ method of MixtureModel."""

    @pytest.mark.parametrize(
        "components_seq",
        [
            (Exponential(loc=0, rate=1), Exponential(loc=1, rate=2)),
            [Exponential(loc=0, rate=1), Exponential(loc=1, rate=2)],
        ],
    )
    def test_init_with_valid_component_sequences(self, components_seq: Sequence[ContinuousDistribution], dtype):
        """Tests that MixtureModel can be initialized with different sequence types."""

        model = MixtureModel(components=components_seq, dtype=dtype)
        expected_n_components = 2
        assert model.n_components == expected_n_components
        assert isinstance(model.components, tuple)

    def test_init_with_equal_weights_by_default(self, exp_components: tuple[Exponential, Exponential], dtype):
        """Tests that weights are distributed equally when not provided."""

        model = MixtureModel(components=exp_components, dtype=dtype)
        expected_n_components = 2
        assert model.n_components == expected_n_components
        np.testing.assert_allclose(model.weights, [0.5, 0.5])

    def test_init_with_specified_weights(self, exp_components: tuple[Exponential, Exponential], dtype):
        """Tests initialization with correctly specified weights."""

        weights = [0.3, 0.7]
        model = MixtureModel(components=exp_components, weights=weights, dtype=dtype)
        expected_n_components = 2
        assert model.n_components == expected_n_components

        atol = np.finfo(dtype).eps
        np.testing.assert_allclose(model.weights, weights, rtol=atol)
        np.testing.assert_allclose(np.exp(model.log_weights), weights, rtol=atol)

    @pytest.mark.parametrize(
        "invalid_weights, error_msg",
        [
            ([0.5, 0.5, 0.5], "Components number \\(2\\) must be equal to weights number \\(3\\)."),
            ([-0.2, 1.2], "Weights must be positive."),
        ],
    )
    def test_init_with_invalid_weights_raises_value_error(
        self, exp_components: tuple[Exponential, Exponential], invalid_weights: list[float], error_msg: str, dtype
    ):
        """Tests that initialization with invalid weights raises a ValueError."""

        with pytest.raises(ValueError, match=error_msg):
            MixtureModel(components=exp_components, weights=invalid_weights, dtype=dtype)

    @pytest.mark.parametrize(
        "invalid_weights, error_msg",
        [
            ([0.4, 0.4], "Sum of the weights must be equal 1, but it equal "),
        ],
    )
    def test_init_with_invalid_sum_of_weights_raises_value_error(
        self, exp_components: tuple[Exponential, Exponential], invalid_weights: list[float], error_msg: str, dtype
    ):
        """Tests that initialization with invalid sum of weights raises a ValueError."""
        with pytest.raises(ValueError) as excinfo:
            MixtureModel(components=exp_components, weights=invalid_weights, dtype=dtype)

        actual_error_msg = str(excinfo.value)
        assert actual_error_msg.startswith(error_msg)

    def test_init_with_empty_components_raises_value_error(self, dtype):
        """Tests that initialization with an empty component list raises a ValueError."""

        with pytest.raises(ValueError, match="List of components cannot be an empty"):
            MixtureModel(components=[], dtype=dtype)

    def test_init_casts_component_dtypes(self, dtype):
        """Tests that the MixtureModel casts all components to its own dtype during initialization."""

        comp1_f64 = Exponential(loc=0.0, rate=1.0)
        comp2_f64 = Exponential(loc=5.0, rate=2.0)
        assert comp1_f64.dtype == np.float64

        mixture = MixtureModel(components=[comp1_f64, comp2_f64], dtype=dtype)

        assert mixture.dtype == dtype
        assert mixture.weights.dtype == dtype
        for component in mixture.components:
            assert component.dtype == dtype
            for param in component.params:
                assert isinstance(getattr(component, param), dtype)

        # Original components have not changed
        for component in [comp1_f64, comp2_f64]:
            assert component.dtype == np.float64
            for param in component.params:
                assert isinstance(getattr(component, param), np.float64)

    def test_init_does_not_recreate_components_with_correct_dtype(self, dtype):
        """Tests that components with the correct dtype are not recreated."""

        comp = Exponential(loc=0.0, rate=1.0, dtype=dtype)

        original_id = id(comp)

        mixture = MixtureModel(components=[comp], dtype=dtype)

        assert id(mixture.components[0]) == original_id


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

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    def test_weights_caching(self, exp_components: tuple[Exponential, Exponential], dtype):
        """Tests the caching mechanism of the 'weights' property."""

        model = MixtureModel(components=exp_components, dtype=dtype)
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
        np.testing.assert_allclose(mixture_model.log_weights, new_log_weights, atol=1e-3)

        new_weights = mixture_model.weights
        np.testing.assert_allclose(new_weights, [0.3, 0.7], atol=1e-3)
        assert id(old_weights) != id(new_weights)
        assert np.isclose(np.sum(new_weights), 1.0, atol=1e-3)

    def test_log_weights_setter_with_invalid_shape_raises_error(self, mixture_model: MixtureModel):
        """Tests that setting log_weights with an incorrect shape raises a ValueError."""

        with pytest.raises(ValueError, match="The length of the new logit vector does not match"):
            mixture_model.log_weights = np.log([0.1, 0.2, 0.7])

    def test_properties_have_correct_dtype(self, mixture_model: MixtureModel):
        """Tests that checks the dtype of the weights and log_weights properties."""

        assert mixture_model.weights.dtype == mixture_model.dtype
        assert mixture_model.log_weights.dtype == mixture_model.dtype


class TestMixtureModelModification:
    """Tests for methods that modify the MixtureModel instance."""

    def test_add_component(self, mixture_model: MixtureModel):
        """Tests adding a new component and verifies weight preservation and renormalization."""

        dtype = mixture_model.dtype

        old_weights = mixture_model.weights.copy()
        new_component = Exponential(loc=10, rate=3, dtype=dtype)
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

    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    def test_add_component_casts_dtype(self, dtype):
        """Tests that the add_component method casts the type of the new component to the dtype of the mixture."""

        comp = Exponential(loc=0.0, rate=1.0)
        mixture = MixtureModel(components=[comp], dtype=dtype)

        new_comp_f64 = Exponential(loc=10.0, rate=2.0)
        assert new_comp_f64.dtype == np.float64

        mixture.add_component(new_comp_f64, weight=0.3)

        added_component_in_mixture = mixture.components[-1]
        assert added_component_in_mixture.dtype == dtype
        assert isinstance(added_component_in_mixture.loc, dtype)

        # Original component have not changed
        assert comp.dtype == np.float64
        for param in comp.params:
            assert isinstance(getattr(comp, param), np.float64)

    def test_remove_component(self):
        """Tests removing a component and verifies weight renormalization."""

        components = [Exponential(0, 1), Exponential(5, 2), Exponential(10, 3)]
        model = MixtureModel(components=components, weights=[0.2, 0.5, 0.3])

        model.remove_component(1)

        expected_n_components = 2

        assert model.n_components == expected_n_components

        assert model.components == (components[0], components[2])

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


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestMixtureModelCalculations:
    """Tests for calculation methods like pdf, lpdf, etc."""

    @given(x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_pdf_calculation_for_array(self, x, dtype):
        """Tests the PDF calculation against the definition."""

        mixture_model = MixtureModel(
            components=[Exponential(loc=0.0, rate=1.0), Exponential(loc=5.0, rate=2.0)], weights=[0.5, 0.5], dtype=dtype
        )
        c1, c2 = mixture_model.components
        w1, w2 = mixture_model.weights

        expected_pdf = w1 * c1.pdf(x) + w2 * c2.pdf(x)
        calculated_pdf = mixture_model.pdf(x)

        assert calculated_pdf.dtype == dtype
        if not np.isscalar(x):
            assert isinstance(calculated_pdf, np.ndarray)

        np.testing.assert_allclose(calculated_pdf, expected_pdf, atol=np.finfo(dtype).eps)

    @given(x=st.floats(-1e6, 1e6))
    def test_pdf_calculation_for_scalar(self, x, dtype):
        """Tests the PDF calculation against the definition."""

        mixture_model = MixtureModel(
            components=[Exponential(loc=0.0, rate=1.0), Exponential(loc=5.0, rate=2.0)], weights=[0.5, 0.5], dtype=dtype
        )
        c1, c2 = mixture_model.components
        w1, w2 = mixture_model.weights

        expected_pdf = w1 * c1.pdf(x) + w2 * c2.pdf(x)
        calculated_pdf = mixture_model.pdf(x)

        assert np.isscalar(calculated_pdf)
        assert isinstance(calculated_pdf, dtype)
        np.testing.assert_allclose(calculated_pdf, expected_pdf, atol=np.finfo(dtype).eps)

    @given(x=arrays(np.float64, st.integers(0, 10), elements=st.floats(-1e6, 1e6)))
    def test_lpdf_calculation_for_array(self, x, dtype):
        """Tests the LPDF calculation against the definition."""

        mixture_model = MixtureModel(
            components=[Exponential(loc=0.0, rate=1.0), Exponential(loc=5.0, rate=2.0)], weights=[0.5, 0.5], dtype=dtype
        )

        expected_pdf = mixture_model.pdf(x)
        calculated_lpdf = mixture_model.lpdf(x)

        if not np.isscalar(x):
            assert isinstance(calculated_lpdf, np.ndarray)

        assert calculated_lpdf.dtype == dtype
        np.testing.assert_allclose(np.exp(calculated_lpdf), expected_pdf, atol=np.finfo(dtype).eps)

    @given(x=st.floats(-1e6, 1e6))
    def test_lpdf_calculation_for_scalar(self, x, dtype):
        """Tests the LPDF calculation against the definition."""

        mixture_model = MixtureModel(
            components=[Exponential(loc=0.0, rate=1.0), Exponential(loc=5.0, rate=2.0)], weights=[0.5, 0.5], dtype=dtype
        )

        expected_pdf = mixture_model.pdf(x)
        calculated_lpdf = mixture_model.lpdf(x)

        assert np.isscalar(calculated_lpdf)
        assert calculated_lpdf.dtype == dtype
        np.testing.assert_allclose(np.exp(calculated_lpdf), expected_pdf, atol=np.finfo(dtype).eps)

    def test_loglikelihood_calculation(self, dtype):
        """Tests that loglikelihood is the sum of LPDF values."""

        mixture_model = MixtureModel(
            components=[Exponential(loc=0.0, rate=1.0), Exponential(loc=5.0, rate=2.0)], weights=[0.5, 0.5], dtype=dtype
        )
        X = np.array([1.0, 1.5, 6.0])
        expected_loglikelihood = np.sum(mixture_model.lpdf(X))
        calculated_loglikelihood = mixture_model.loglikelihood(X)

        assert isinstance(calculated_loglikelihood, dtype)
        assert np.isclose(calculated_loglikelihood, expected_loglikelihood)


class TestMixtureModelGenerate:
    """Statistical tests for the `generate` method."""

    def test_generate_returns_correct_size(self, mixture_model: MixtureModel):
        """Tests that generate returns an array of the requested size."""

        size = 100
        dtype = mixture_model.dtype

        samples = mixture_model.generate(size=size)

        assert len(samples) == size
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == dtype

    def test_generate_with_size_zero(self, mixture_model):
        """Tests that generating with size = 0 returns an empty array."""

        dtype = mixture_model.dtype

        samples = mixture_model.generate(0)
        assert len(samples) == 0
        assert samples.dtype == dtype

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


class TestMixtureModelAstype:
    """Tests for astype method of MixtureModel"""

    def test_astype_successful_conversion(self, exp_components: tuple[Exponential, Exponential]):
        """
        Tests that astype creates a new instance with the correct new dtype
        and that the original instance remains unchanged.
        """
        mixture_model = MixtureModel(components=exp_components, weights=[0.5, 0.5], dtype=np.float64)

        assert mixture_model.dtype == np.float64
        assert mixture_model.log_weights.dtype == np.float64
        for component in mixture_model.components:
            assert component.dtype == np.float64

        target_dtype = np.float32
        new_mixture = mixture_model.astype(target_dtype)

        assert new_mixture is not mixture_model
        assert new_mixture != mixture_model

        assert new_mixture.dtype == np.float32
        assert new_mixture.log_weights.dtype == np.float32
        for component in new_mixture.components:
            assert component.dtype == np.float32

        # original instance remains unchanged
        assert mixture_model.dtype == np.float64
        assert mixture_model.log_weights.dtype == np.float64
        for component in mixture_model.components:
            assert component.dtype == np.float64

    def test_astype_returns_self_if_same_dtype(self, exp_components: tuple[Exponential, Exponential]):
        """
        Tests that astype returns the same instance if the target dtype
        is identical to the current one, avoiding unnecessary copying.
        """
        mixture_model = MixtureModel(components=exp_components, weights=[0.5, 0.5], dtype=np.float64)

        assert mixture_model.dtype == np.float64

        same_dtype_dist = mixture_model.astype(np.float64)

        assert same_dtype_dist is mixture_model


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

        dtype = mixture_model.dtype
        assert mixture_model[index] == exp_components[expected_component_index].astype(dtype)

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

        dtype = mixture_model.dtype
        exp_components = [component.astype(dtype) for component in exp_components]

        iterated_components = list(mixture_model)
        assert iterated_components == list(exp_components)
        assert all(comp_iter == comp_orig for comp_iter, comp_orig in zip(iterated_components, exp_components))

    def test_iter_is_reusable(self, mixture_model: MixtureModel):
        """Tests that the model can be iterated over multiple times."""

        first_pass = list(mixture_model)
        second_pass = list(mixture_model)

        assert len(first_pass) == mixture_model.n_components
        assert first_pass is not second_pass  # The lists are different objects
        assert first_pass == second_pass  # But their contents are identical


class TestMixtureModelCopying:
    """Tests the __copy__ method for MixtureModel."""

    def test_copy_creates_new_equal_instance(self, mixture_model: MixtureModel):
        """Tests that copy.copy() creates a new instance that is equal to the original."""

        copied_model = copy(mixture_model)

        assert copied_model is not mixture_model
        assert copied_model == mixture_model

    def test_copy_is_independent_weights(self, mixture_model: MixtureModel):
        """Tests that modifying the copied model's weights does not affect the original."""

        copied_model = copy(mixture_model)
        copied_model.log_weights = np.log([0.1, 0.9])

        np.testing.assert_allclose(mixture_model.weights, [0.5, 0.5], atol=1e-4)
        np.testing.assert_allclose(copied_model.weights, [0.1, 0.9], atol=1e-4)

    def test_copy_is_independent_components(self, mixture_model: MixtureModel):
        """Tests that the components in the copied model are also independent copies."""

        copied_model = copy(mixture_model)

        # Check that component objects are new instances
        assert copied_model.components[0] is not mixture_model.components[0]
        assert copied_model.components[1] is not mixture_model.components[1]

        # Modify a parameter in a component of the copied model
        copied_model.components[0].rate = 999.0

        # Verify the original model's component is unchanged
        assert mixture_model.components[0].rate != copied_model.components[0].rate


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
class TestMixtureModelComparison:
    """Tests the __eq__ and __hash__ methods for MixtureModel."""

    def test_eq_identical_models(self, dtype):
        """Tests that two identical models are equal."""

        c = [Exponential(0, 1), Exponential(10, 2)]
        m1 = MixtureModel(components=c, weights=[0.4, 0.6], dtype=dtype)
        m2 = MixtureModel(components=c, weights=[0.4, 0.6], dtype=dtype)
        assert m1 == m2

    def test_eq_order_insensitivity(self, dtype):
        """Tests that models with the same components/weights in a different order are equal."""

        c1 = [Exponential(0, 1), Exponential(10, 2)]
        w1 = [0.4, 0.6]
        m1 = MixtureModel(components=c1, weights=w1, dtype=dtype)

        c2 = [Exponential(10, 2), Exponential(0, 1)]
        w2 = [0.6, 0.4]
        m2 = MixtureModel(components=c2, weights=w2, dtype=dtype)

        assert m1 == m2

    def test_neq_different_weights(self, dtype):
        """Tests that models with different weights are not equal."""

        c = [Exponential(0, 1), Exponential(10, 2)]
        m1 = MixtureModel(components=c, weights=[0.4, 0.6], dtype=dtype)
        m2 = MixtureModel(components=c, weights=[0.41, 0.59], dtype=dtype)
        assert m1 != m2

    def test_neq_different_components(self, dtype):
        """Tests that models with different components are not equal."""

        c1 = [Exponential(0, 1), Exponential(10, 2)]
        c2 = [Exponential(0, 1), Exponential(99, 2)]
        m1 = MixtureModel(components=c1, weights=[0.4, 0.6], dtype=dtype)
        m2 = MixtureModel(components=c2, weights=[0.4, 0.6], dtype=dtype)
        assert m1 != m2

    def test_neq_different_n_components(self, dtype):
        """Tests that models with a different number of components are not equal."""

        c1 = [Exponential(0, 1), Exponential(10, 2)]
        c2 = [Exponential(0, 1)]
        m1 = MixtureModel(components=c1, weights=[0.4, 0.6], dtype=dtype)
        m2 = MixtureModel(components=c2, weights=[1.0], dtype=dtype)
        assert m1 != m2

    def test_neq_different_dtype(self, dtype):
        """Tests that models with different dtype are not equal."""

        c1 = [Exponential(0, 1), Exponential(10, 2)]
        c2 = [Exponential(0, 1), Exponential(99, 2)]
        m1 = MixtureModel(components=c1, weights=[0.4, 0.6], dtype=dtype)
        m2 = MixtureModel(components=c2, weights=[0.4, 0.6], dtype=np.float128)
        assert m1 != m2

    def test_neq_other_object(self, mixture_model, exp_components, dtype):
        """Tests that a model instance is not equal to an object of a different class."""
        model = mixture_model

        other = "not_a_mixture_model"
        assert model != other

    def test_hash_consistency(self, dtype):
        """Tests that equal models produce the same hash."""

        m1 = MixtureModel(components=[Exponential(0, 1), Exponential(10, 2)], weights=[0.4, 0.6], dtype=dtype)
        m2 = MixtureModel(components=[Exponential(10, 2), Exponential(0, 1)], weights=[0.6, 0.4], dtype=dtype)
        assert m1 == m2
        assert hash(m1) == hash(m2)

    def test_hash_difference(self, dtype):
        """Tests that non-equal models are likely to have different hashes."""

        m1 = MixtureModel(components=[Exponential(0, 1), Exponential(10, 2)], weights=[0.4, 0.6], dtype=dtype)
        m2 = MixtureModel(components=[Exponential(0, 1), Exponential(10, 2)], weights=[0.5, 0.5], dtype=dtype)
        assert m1 != m2
        assert hash(m1) != hash(m2)

    def test_hash_changes_after_modifying_weights(self, dtype):
        """Tests that the hash of the model updates after its weights are changed."""
        model = MixtureModel(components=[Exponential(0, 1), Exponential(10, 2)], weights=[0.4, 0.6], dtype=dtype)
        old_hash = hash(model)

        model.log_weights = np.log([0.5, 0.5])
        new_hash = hash(model)

        assert old_hash != new_hash

    def test_hash_changes_after_adding_component(self, dtype):
        """Tests that the hash of the model updates after a new component is added."""
        model = MixtureModel(components=[Exponential(0, 1), Exponential(10, 2)], weights=[0.4, 0.6], dtype=dtype)
        old_hash = hash(model)

        model.add_component(Exponential(99, 3), weight=0.1)
        new_hash = hash(model)
        expected_n_components = 3

        assert model.n_components == expected_n_components
        assert old_hash != new_hash

    def test_hash_changes_after_removing_component(self, dtype):
        """Tests that the hash of the model updates after a component is removed."""
        model = MixtureModel(
            components=[Exponential(0, 1), Exponential(10, 2), Exponential(99, 3)], weights=[0.4, 0.5, 0.1], dtype=dtype
        )
        old_hash = hash(model)

        model.remove_component(1)
        new_hash = hash(model)
        expected_n_components = 2

        assert model.n_components == expected_n_components
        assert old_hash != new_hash
