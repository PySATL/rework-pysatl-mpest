"""Pytest configuration and shared fixtures for the PySATL project tests."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest
from _pytest.fixtures import FixtureRequest
from tests.helpers.golden import GoldenDataComparator


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
