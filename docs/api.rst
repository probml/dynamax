
API
===

State Space Model (Base class)
===============================

.. autoclass:: dynamax.ssm.SSM
  :members:

Parameters
----------

Parameters and their associated properties are stored as :class:`jax.DeviceArray`
and :class:`dynamax.parameters.ParameterProperties`, respectively. They are bundled together into a
:class:`dynamax.parameters.ParameterSet` and a :class:`dynamax.parameters.PropertySet`, which are simply
aliases for immutable datastructures (in our case,  :class:`NamedTuple`).

.. autoclass:: dynamax.parameters.ParameterSet
.. autoclass:: dynamax.parameters.PropertySet
.. autoclass:: dynamax.parameters.ParameterProperties

Hidden Markov Model
===================

High-level class
----------------

.. autoclass:: dynamax.hidden_markov_model.HMM
  :members:

Low-level inference
-------------------

.. autofunction:: dynamax.hidden_markov_model.hmm_filter
.. autofunction:: dynamax.hidden_markov_model.hmm_smoother
.. autofunction:: dynamax.hidden_markov_model.hmm_two_filter_smoother
.. autofunction:: dynamax.hidden_markov_model.hmm_fixed_lag_smoother
.. autofunction:: dynamax.hidden_markov_model.hmm_posterior_mode
.. autofunction:: dynamax.hidden_markov_model.hmm_posterior_sample

Linear Gaussian SSM
====================

High-level class
----------------

.. autoclass:: dynamax.linear_gaussian_ssm.LinearGaussianSSM
  :members:

Low-level inference
-------------------

.. autofunction:: dynamax.linear_gaussian_ssm.lgssm_filter
.. autofunction:: dynamax.linear_gaussian_ssm.lgssm_smoother
.. autofunction:: dynamax.linear_gaussian_ssm.lgssm_posterior_sample

Types
-----

.. autoclass:: dynamax.linear_gaussian_ssm.ParamsLGSSM
.. autoclass:: dynamax.linear_gaussian_ssm.ParamsLGSSMInitial
.. autoclass:: dynamax.linear_gaussian_ssm.ParamsLGSSMDynamics
.. autoclass:: dynamax.linear_gaussian_ssm.ParamsLGSSMEmissions
.. autoclass:: dynamax.linear_gaussian_ssm.PosteriorLGSSMFiltered
.. autoclass:: dynamax.linear_gaussian_ssm.PosteriorLGSSMSmoothed

Nonlinear Gaussian GSSM
========================


High-level class
----------------

.. autoclass:: dynamax.nonlinear_gaussian_ssm.NonlinearGaussianSSM
  :members:

Low-level inference
-------------------

.. autofunction:: dynamax.nonlinear_gaussian_ssm.extended_kalman_filter
.. autofunction:: dynamax.nonlinear_gaussian_ssm.iterated_extended_kalman_filter
.. autofunction:: dynamax.nonlinear_gaussian_ssm.extended_kalman_smoother
.. autofunction:: dynamax.nonlinear_gaussian_ssm.iterated_extended_kalman_smoother

.. autofunction:: dynamax.nonlinear_gaussian_ssm.unscented_kalman_filter
.. autofunction:: dynamax.nonlinear_gaussian_ssm.unscented_kalman_smoother

Generalized Gaussian GSSM
==========================

High-level class
----------------

.. autoclass:: dynamax.generalized_gaussian_ssm.GeneralizedGaussianSSM
  :members:

Low-level inference
-------------------

.. autofunction:: dynamax.generalized_gaussian_ssm.conditional_moments_gaussian_filter
.. autofunction:: dynamax.generalized_gaussian_ssm.iterated_conditional_moments_gaussian_filter
.. autofunction:: dynamax.generalized_gaussian_ssm.conditional_moments_gaussian_smoother
.. autofunction:: dynamax.generalized_gaussian_ssm.iterated_conditional_moments_gaussian_smoother