"""Golden data comparator utilities for testing."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from _pytest.fixtures import FixtureRequest


class GoldenDataComparator:
    """
    A class to compare test results against golden data (snapshots).
    Supports NumPy arrays (.npz) and scalar values (.json).
    """

    def __init__(self, request: FixtureRequest):
        self.request = request
        self.update_mode: bool = request.config.getoption("--update-golden")

        root_dir = Path(request.config.rootdir)
        self.golden_dir = root_dir / "tests" / "testdata" / "golden"
        self.golden_dir.mkdir(parents=True, exist_ok=True)

        # Build file name from module and test name
        test_module = Path(request.node.path).stem
        safe_test_name = re.sub(r"[/\\\[\]]", "_", request.node.name)
        self.golden_file_base = self.golden_dir / f"{test_module}_{safe_test_name}"

    def _ensure_file_exists(self, file_path: Path) -> None:
        """Fails the test if the golden file does not exist."""

        if not file_path.exists():
            pytest.fail(f"Golden data file not found: {file_path}. Run with --update-golden to create it.")

    def compare_tensors(self, data: dict[str, npt.NDArray[Any]], rtol: float = 1e-5, atol: float = 1e-8) -> None:
        """Compares dictionaries of NumPy arrays."""

        file_path = self.golden_file_base.with_suffix(".npz")

        if self.update_mode:
            np.savez(file_path, **data)
            pytest.skip(f"Golden data updated at {file_path}")
            return

        self._ensure_file_exists(file_path)

        with np.load(file_path) as loaded:
            for key, value in data.items():
                if key not in loaded:
                    pytest.fail(f"Key '{key}' not found in golden data.")
                np.testing.assert_allclose(
                    value, loaded[key], rtol=rtol, atol=atol, err_msg=f"Mismatch in golden key '{key}'"
                )

    def compare_scalars(self, data: dict[str, Any]) -> None:
        """Compares scalar values, saving them to a JSON file."""

        file_path = self.golden_file_base.with_suffix(".json")

        if self.update_mode:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            pytest.skip(f"Golden scalar data updated at {file_path}")
            return

        self._ensure_file_exists(file_path)

        with open(file_path, encoding="utf-8") as f:
            loaded = json.load(f)

        for key, value in data.items():
            if key not in loaded:
                pytest.fail(f"Key '{key}' not found in golden data.")
            assert value == loaded[key], f"Mismatch in golden key '{key}': expected {loaded[key]}, got {value}"
