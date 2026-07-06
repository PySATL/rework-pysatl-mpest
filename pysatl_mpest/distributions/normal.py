"""Module providing normal (Gaussian) distribution class"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_core.families import ParametricFamilyRegister

from pysatl_mpest.distributions.core_dist_adapter import CoreDistributionAdapter


class Normal(CoreDistributionAdapter):
    """Class for the Normal (Gaussian) distribution.

    Parameters
    ----------
    loc : float
        Mean of the distribution (mu). Can be any real number.
    scale : float
        Standard deviation of the distribution (sigma). Must be positive.

    Attributes
    ----------
    loc : float
        Mean of the distribution.
    scale : float
        Standard deviation of the distribution.


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

    def __init__(self, mu: float, sigma: float):
        super().__init__(ParametricFamilyRegister.get("Normal"), "meanStd", mu=mu, sigma=sigma)

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        Returns
        -------
        str
            A string that can be used to recreate the object, e.g.,
            "Normal(mu=0.0, sigma=1.0)".
        """

        return f"{self.__class__.__name__}(mu={self.mu}, sigma={self.sigma})"
