"""Module which contains all available distributions for preprocessing module"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Union

from rework_pysatl_mpest.distributions.beta import Beta
from rework_pysatl_mpest.distributions.cauchy import Cauchy
from rework_pysatl_mpest.distributions.exponential import Exponential
from rework_pysatl_mpest.distributions.normal import Normal
from rework_pysatl_mpest.distributions.uniform import Uniform
from rework_pysatl_mpest.distributions.weibull import Weibull

Distribution = type[Union[Normal, Weibull, Exponential, Cauchy, Uniform, Beta]]
