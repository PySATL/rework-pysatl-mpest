.. _api.estimators:

###############
mpest.estimators
###############

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   direct/index
   iterative/index

.. automodule:: pysatl_mpest.estimators
   :no-members:

------------------

Facades (High-Level API)
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: em.rst

    ~EM

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

    ~ECM
    ~ECME

.. note::

   ``EM`` is the standalone Expectation-Maximization estimator; the current ``ECM`` implementation performs the same component-wise Q-function updates.

Abstract Classes
================

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~BaseEstimator
