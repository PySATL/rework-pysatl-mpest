"""Pytest configuration and shared fixtures for the PySATL project tests."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import random

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from hypothesis import settings
from tests.helpers.golden import GoldenDataComparator

# Зафиксируем сид для Hypothesis, чтобы убрать рандом в генерации параметров
settings.register_profile("deterministic", derandomize=True)
settings.load_profile("deterministic")


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Sets random seeds for reproducible tests across the suite."""
    random.seed(42)
    np.random.seed(42)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom flags for pytest."""

    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Update or create golden data files",
    )


@pytest.fixture
def golden(request: FixtureRequest) -> GoldenDataComparator:
    """Fixture to use Golden Tests."""

    return GoldenDataComparator(request)
