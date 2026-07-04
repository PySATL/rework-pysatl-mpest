"""Module providing uniform distribution class"""

__author__ = "Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_core.families import ParametricFamilyRegister
from pysatl_core.types import FamilyName

from pysatl_mpest.distributions.core_dist_adapter import CoreDistributionAdapter


class Uniform(CoreDistributionAdapter):
    """
    The Uniform continuous probability distribution.

    The uniform distribution describes an experiment where there is an arbitrary
    outcome that lies between certain bounds. The probability is constant between
    these bounds and zero elsewhere.

    Parameters
    ----------
    left_border : float
        Left border of section [a, b]. Can be any real number.
    right_border : float
        Right border of section [a, b]. Can be any real number.

    Attributes
    ----------
    left_border : float
        Left border of section [a, b].
    right_border : float
        Right border of section [a, b].

    Raises
    ------
    ValueError
        If left_border is greater than or equal to right_border, or if either
        parameter is not finite.

    Methods
    -------

    .. autosummary::
        :toctree: generated/

        ppf
        pdf
        lpdf
        log_gradients
        generate
    """

    def __init__(self, lower_bound: float, upper_bound: float):
        super().__init__(
            ParametricFamilyRegister.get(FamilyName.CONTINUOUS_UNIFORM),
            "standard",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Uniform(lower_bound=0.0, upper_bound=2.0)".
        """

        return f"{self.__class__.__name__}(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
