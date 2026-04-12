"""Tests for L-moments optimization strategy for Normal distribution"""

__author__ = "Maksim Pastukhov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pysatl_mpest.distributions import Normal
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState
from pysatl_mpest.estimators.iterative._strategies import lmoments_strategy

DTYPES_TO_TEST = [np.float32, np.float64]


@pytest.fixture(params=DTYPES_TO_TEST)
def parametrized_normal_setup(request):
    dtype = request.param
    component = Normal(loc=0.0, scale=1.0, dtype=dtype)

    state = PipelineState(
        X=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype),
        H=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]], dtype=dtype),
        prev_mixture=None,
        curr_mixture=None,
        error=None,
    )
    return component, state, dtype


def test_lmoments_normal_raises_value_error_if_h_is_none(parametrized_normal_setup):
    normal_component, state, _ = parametrized_normal_setup
    state.H = None
    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)

    with pytest.raises(ValueError, match="Responsibility matrix H is not computed."):
        lmoments_strategy(normal_component, state, block, optimizer=None)


def test_lmoments_normal_respects_fixed_params(parametrized_normal_setup):
    normal_component, state, _ = parametrized_normal_setup
    normal_component.fix_param("loc")  # Фиксируем mu

    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)
    _, new_params = lmoments_strategy(normal_component, state, block, optimizer=None)

    assert "loc" not in new_params
    assert "scale" in new_params


def test_lmoments_normal_calculation_correctness(parametrized_normal_setup):
    """
    Проверка конкретного примера.
    Для данных [10, 20] с весами [0.5, 0.5]:
    L1 = (10*0.5 + 20*0.5) / 1.0 = 15.0
    rank_weights:
      w1 = (0.5 - 0.5*0.5)/1 = 0.25
      w2 = (1.0 - 0.5*0.5)/1 = 0.75
    b1 = (10*0.5*0.25 + 20*0.5*0.75) / 1.0 = 1.25 + 7.5 = 8.75
    L2 = 2*8.75 - 15.0 = 2.5
    Sigma = L2 * sqrt(pi) = 2.5 * 1.77245... = 4.4311...
    """
    component, state, dtype = parametrized_normal_setup
    state.X = np.array([10.0, 20.0], dtype=dtype)
    state.H = np.array([[0.5], [0.5]], dtype=dtype)

    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)
    _, new_params = lmoments_strategy(component, state, block, optimizer=None)

    expected_mu = 15.0
    expected_sigma = 2.5 * np.sqrt(np.pi)

    assert new_params[Normal.PARAM_LOC] == pytest.approx(expected_mu, rel=1e-4)
    assert new_params[Normal.PARAM_SCALE] == pytest.approx(expected_sigma, rel=1e-4)


def test_lmoments_normal_handles_zero_variance(parametrized_normal_setup):
    """Если все точки одинаковые, L2 будет 0, sigma должна быть зажата eps."""
    component, state, dtype = parametrized_normal_setup
    val = dtype(10.0)
    state.X = np.array([val, val, val], dtype=dtype)
    state.H = np.ones((3, 1), dtype=dtype)

    block = OptimizationBlock(0, {"scale"}, MaximizationStrategy.LMOMENTS)
    _, new_params = lmoments_strategy(component, state, block, optimizer=None)

    assert new_params[Normal.PARAM_SCALE] >= np.finfo(dtype).eps


@settings(max_examples=20, deadline=None)
@given(mu=st.floats(min_value=-10, max_value=10), sigma=st.floats(min_value=1.0, max_value=5.0))
def test_lmoments_normal_recovery_property(mu, sigma):
    """
    Свойство: на большой выборке взвешенные L-моменты должны
    восстанавливать параметры mu и sigma.
    """
    dtype = np.float64
    sample_size = 10000
    rng = np.random.default_rng(42)

    X = rng.normal(loc=mu, scale=sigma, size=sample_size).astype(dtype)
    H = np.ones((sample_size, 1), dtype=dtype)  # Полная ответственность

    component = Normal(loc=0.0, scale=1.0, dtype=dtype)
    state = PipelineState(X=X, H=H, prev_mixture=None, curr_mixture=None, error=None)
    block = OptimizationBlock(0, {"loc", "scale"}, MaximizationStrategy.LMOMENTS)

    _, new_params = lmoments_strategy(component, state, block, optimizer=None)

    assert new_params[Normal.PARAM_LOC] == pytest.approx(mu, abs=0.1)
    assert new_params[Normal.PARAM_SCALE] == pytest.approx(sigma, rel=0.05)
