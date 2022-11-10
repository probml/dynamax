
API
===

State Space Model (Base class)
===============================

.. autoclass:: dynamax.abstractions.SSM
  :members:

Hidden Markov Model
===================

High-level class
----------------

.. autoclass:: dynamax.hmm.models.abstractions.HMM
  :members:

Low-level inference
-------------------

.. autofunction:: dynamax.hmm.inference.hmm_filter
.. autofunction:: dynamax.hmm.inference.hmm_smoother
.. autofunction:: dynamax.hmm.inference.hmm_two_filter_smoother
.. autofunction:: dynamax.hmm.inference.hmm_fixed_lag_smoother
.. autofunction:: dynamax.hmm.inference.hmm_posterior_mode
.. autofunction:: dynamax.hmm.inference.hmm_posterior_sample

Linear Gaussian SSM
====================

High-level class
----------------

.. autoclass:: dynamax.linear_gaussian_ssm.linear_gaussian_ssm.LinearGaussianSSM
  :members:

Low-level inference
-------------------

.. autofunction:: dynamax.linear_gaussian_ssm.inference.lgssm_filter
.. autofunction:: dynamax.linear_gaussian_ssm.inference.lgssm_smoother
.. autofunction:: dynamax.linear_gaussian_ssm.inference.lgssm_posterior_sample

Nonlinear Gaussian GSSM
========================


High-level class
----------------

.. autoclass:: dynamax.nonlinear_gaussian_ssm.nonlinear_gaussian_ssm.NonlinearGaussianSSM
  :members:

Low-level inference
-------------------

.. autofunction:: dynamax.nonlinear_gaussian_ssm.extended_kalman_filter.extended_kalman_filter
.. autofunction:: dynamax.nonlinear_gaussian_ssm.extended_kalman_filter.iterated_extended_kalman_filter
.. autofunction:: dynamax.nonlinear_gaussian_ssm.extended_kalman_filter.extended_kalman_smoother
.. autofunction:: dynamax.nonlinear_gaussian_ssm.extended_kalman_filter.iterated_extended_kalman_smoother

.. autofunction:: dynamax.nonlinear_gaussian_ssm.unscented_kalman_filter.unscented_kalman_filter
.. autofunction:: dynamax.nonlinear_gaussian_ssm.unscented_kalman_filter.unscented_kalman_smoother

Generalized Gaussian GSSM
==========================

High-level class
----------------

.. autoclass:: dynamax.generalized_gaussian_ssm.generalized_gaussian_ssm.GeneralizedGaussianSSM
  :members:

Low-level inference
-------------------

.. autofunction:: dynamax.generalized_gaussian_ssm.conditional_moments_gaussian_filter.conditional_moments_gaussian_filter
.. autofunction:: dynamax.generalized_gaussian_ssm.conditional_moments_gaussian_filter.iterated_conditional_moments_gaussian_filter
.. autofunction:: dynamax.generalized_gaussian_ssm.conditional_moments_gaussian_filter.conditional_moments_gaussian_smoother
.. autofunction:: dynamax.generalized_gaussian_ssm.conditional_moments_gaussian_filter.iterated_conditional_moments_gaussian_smoother