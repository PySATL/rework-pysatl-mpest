"""
Unit tests for the CoreDistributionAdapter class.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import copy

import numpy as np
import pytest
from pysatl_core.families import configure_families_register
from pysatl_mpest.distributions.core_dist_adapter import CoreDistributionAdapter


@pytest.fixture
def core_adapter() -> CoreDistributionAdapter:
    """Fixture providing a CoreDistributionAdapter for the Normal family."""
    registry = configure_families_register()
    normal_family = registry.get("Normal")

    return CoreDistributionAdapter(
        core_family=normal_family,
        parametrization_name="meanStd",
        mu=0.0,
        sigma=1.0,
    )


def test_properties(core_adapter: CoreDistributionAdapter) -> None:
    """Test name and params properties."""
    assert core_adapter.name == "Normal"
    assert core_adapter.params == {"mu", "sigma"}


def test_getattr(core_adapter: CoreDistributionAdapter) -> None:
    """Test dynamic attribute access for parameters."""
    assert core_adapter.mu == 0.0
    assert core_adapter.sigma == 1.0

    with pytest.raises(AttributeError):
        _ = core_adapter.unknown_param  # type: ignore


def test_copy(core_adapter: CoreDistributionAdapter) -> None:
    """Test that __copy__ preserves parameters and fixed state."""
    core_adapter.fix_param("mu")

    adapter_copy = copy.copy(core_adapter)

    assert adapter_copy is not core_adapter
    assert adapter_copy.mu == core_adapter.mu
    assert adapter_copy.sigma == core_adapter.sigma
    assert adapter_copy.params_to_optimize == core_adapter.params_to_optimize
    assert adapter_copy._fixed_params == core_adapter._fixed_params


def test_pdf_lpdf_ppf(core_adapter: CoreDistributionAdapter) -> None:
    """Test PDF, LPDF, and PPF methods."""
    x = np.array([0.0, 1.0])

    pdf_vals = core_adapter.pdf(x)
    assert isinstance(pdf_vals, np.ndarray)
    assert pdf_vals.shape == x.shape

    lpdf_vals = core_adapter.lpdf(x)
    assert isinstance(lpdf_vals, np.ndarray)
    assert lpdf_vals.shape == x.shape

    q = np.array([0.5, 0.9])
    ppf_vals = core_adapter.ppf(q)
    assert isinstance(ppf_vals, np.ndarray)
    assert ppf_vals.shape == q.shape


def test_log_gradients(core_adapter: CoreDistributionAdapter) -> None:
    """Test log gradients computation."""
    x = np.array([0.0, 1.0])
    grads = core_adapter.log_gradients(x)

    assert isinstance(grads, np.ndarray)
    assert grads.shape == (2, 2)  # 2 points, 2 free parameters

    core_adapter.fix_param("sigma")
    grads_fixed = core_adapter.log_gradients(x)
    assert isinstance(grads_fixed, np.ndarray)
    assert grads_fixed.shape == (2, 1)  # 2 points, 1 free parameter


def test_generate(core_adapter: CoreDistributionAdapter) -> None:
    """Test sample generation."""
    # Test int size
    samples_int = core_adapter.generate(size=10)
    assert isinstance(samples_int, np.ndarray)
    assert samples_int.shape == (10,)

    # Test None size
    sample_none = core_adapter.generate()
    assert isinstance(sample_none, np.float64)

    # Test tuple size
    samples_tuple = core_adapter.generate(size=(2, 5))
    assert isinstance(samples_tuple, np.ndarray)
    assert samples_tuple.shape == (2, 5)
