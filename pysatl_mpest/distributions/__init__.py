"""
The `distribution` package provides an abstract class of continuous distributions and concrete implementations.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_core.families import ParametricFamilyDistribution, ParametricFamilyRegister, configure_families_register

configure_families_register()


def Exponential(parametrization_name: str = "rate", **kwargs) -> ParametricFamilyDistribution:
    family = ParametricFamilyRegister.get("Exponential")
    return family.distribution(parametrization_name=parametrization_name, **kwargs)


def Normal(parametrization_name: str = "meanStd", **kwargs) -> ParametricFamilyDistribution:
    family = ParametricFamilyRegister.get("Normal")
    return family.distribution(parametrization_name=parametrization_name, **kwargs)


def Uniform(parametrization_name: str = "standard", **kwargs) -> ParametricFamilyDistribution:
    family = ParametricFamilyRegister.get("Uniform")
    return family.distribution(parametrization_name=parametrization_name, **kwargs)
