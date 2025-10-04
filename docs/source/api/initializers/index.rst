.. _api.initializers:

##################
mpest.initializers
##################

.. automodule:: rework_pysatl_mpest.Initializers
   :no-members:

Initialization Strategies
=========================

Abstract Classes
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~Initializer

Concrete Implementations
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~ClusterizeInitializer

Cluster Matching Strategies
---------------------------

Functions for matching clusters to distribution models based on statistical criteria.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: function.rst

   ~match_clusters_for_models_akaike
   ~match_clusters_for_models_log_likelihood

Parameter Estimation Strategies
-------------------------------

Q-function based parameter estimation with multiple implementations.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: function.rst

   ~q_function_strategy

Q-function Strategy Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: function.rst

   ~q_function_strategy_exponential

Utility Functions
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: function.rst

   ~get_available_q_function_implementations

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Main Q-function Strategy
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rework_pysatl_mpest.Initializers.q_function_strategy

Exponential Distribution Specialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rework_pysatl_mpest.Initializers.q_function_strategy_exponential

Implementation Registry
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rework_pysatl_mpest.Initializers.get_available_q_function_implementations

Strategy Enumerations
---------------------

Enumeration types that define available strategies for initialization.

Cluster Matching Strategies (`ClusterMatchStrategy`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~ClusterMatchStrategy

Parameter Estimation Strategies (`EstimationStrategy`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   ~EstimationStrategy
