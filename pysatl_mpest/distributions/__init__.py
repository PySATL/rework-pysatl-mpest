"""
The `distribution` package provides an abstract class of continuous distributions and concrete implementations.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_core.families import configure_families_register

configure_families_register()


def Exponential():
    pass


def Normal():
    pass


def Uniform():
    pass
