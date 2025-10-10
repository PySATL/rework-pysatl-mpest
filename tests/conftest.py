"""Pytest configuration file to set a safe multiprocessing start method."""

import multiprocessing
import sys
from contextlib import suppress


def pytest_sessionstart(session):
    """
    Pytest hook called before test collection and execution begins.
    """
    if sys.platform.startswith("linux"):
        with suppress(RuntimeError):
            multiprocessing.set_start_method("spawn")
