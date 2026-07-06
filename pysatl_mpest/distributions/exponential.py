"""Module providing exponential distribution class"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_core.families import ParametricFamilyRegister
from pysatl_core.types import FamilyName

from pysatl_mpest.distributions.core_dist_adapter import CoreDistributionAdapter


class Exponential(CoreDistributionAdapter):
    """Class for the two-parameter exponential distribution.

    Parameters
    ----------
    loc : float
        Location parameter. Can be any real number.
    rate : float
        Rate parameter (lambda). Must be positive.

    Attributes
    ----------
    loc : float
        Location parameter.
    rate : float
        Rate parameter.


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

    def __init__(self, lambda_: float):
        super().__init__(ParametricFamilyRegister.get(FamilyName.EXPONENTIAL), "rate", lambda_=lambda_)

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Exponential(lambda_=2.0)".
        """

        return f"{self.__class__.__name__}(lambda_={self.lambda_})"
