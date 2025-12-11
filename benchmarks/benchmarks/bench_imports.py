"""
Benchmarks for import time latency.
"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class ImportLib:
    """
    Benchmarks checking the import time of the library and its submodules.
    Crucial for ensuring CLI responsiveness and avoiding heavy imports at top-level.
    """

    def timeraw_import_top_level(self):
        return "import rework_pysatl_mpest"

    def timeraw_import_core(self):
        return "from rework_pysatl_mpest import core"

    def timeraw_import_distributions(self):
        return "from rework_pysatl_mpest import distributions"

    def timeraw_import_estimators(self):
        return "from rework_pysatl_mpest import estimators"
