"""
components_family module for evaluating the family of components of a mixture

This module provides a ready-made model for determining the most likely families of mixture components.
The choice of a mixture component family is important when working with mixtures of distributions

**Purpose**

components_family module helps to speed up the search for the most suitable mixture component configuration
by narrowing down the search to a few options

**Usage Example**

.. code-block:: python
    >>> import numpy as np
    >>> from pysatl_mpest.preprocessing.components_family import ComponentsFamily
    >>> from pysatl_mpest.preprocessing.components_family import XGBBaseModel

    >>> # Create random sample
    >>> X = np.linspace(-10, 10, 200)

    >>> # Determine 5 possible configurations using XGBaseModel
    >>> model = ComponentsFamily(XGBBaseModel, top_k=5)
    >>> configurations = model.predict(X)

    >>> print(f"Best 5 configurations: {configurations}")
    >>> print(f"Best configuration: {configurations[0]}")
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_mpest.preprocessing.components_family.components_family import ComponentsFamily
from pysatl_mpest.preprocessing.components_family.mixture_classifiers import XGBBaseModel

__all__ = ["ComponentsFamily", "XGBBaseModel"]
