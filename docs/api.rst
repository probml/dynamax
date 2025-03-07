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

Abstract classes
------------------

.. autoclass:: dynamax.hidden_markov_model.HMM
  :show-inheritance:
  :members:

.. autoclass:: dynamax.hidden_markov_model.HMMInitialState
  :members:

.. autoclass:: dynamax.hidden_markov_model.HMMTransitions
  :members:

.. autoclass:: dynamax.hidden_markov_model.HMMEmissions
  :members:

High-level models
-----------------

The HMM implementations below cover common emission distributions and,
if the emissions are exponential family distributions, the models implement
closed form EM updates. For HMMs with emissions outside the non-exponential family,
these models default to a generic M-step implemented in :class:`HMMEmissions`.

Unless otherwise specified, these models have standard initial distributions and
transition distributions with conjugate, Bayesian priors on their parameters.

**Initial distribution:**

$$p(z_1 \mid \pi_1) = \mathrm{Cat}(z_1 \mid \pi_1)$$
$$p(\pi_1) = \mathrm{Dir}(\pi_1 \mid \alpha 1_K)$$

where $\alpha$ is the prior concentration on the initial distribution $\pi_1$.

**Transition distribution:**

$$p(z_t \mid z_{t-1}, \theta) = \mathrm{Cat}(z_t \mid A_{z_{t-1}})$$
$$p(A) = \prod_{k=1}^K \mathrm{Dir}(A_k \mid \beta 1_K + \kappa e_k)$$

where $\beta$ is the prior concentration on the rows of the transition matrix $A$
and $\kappa$ is the `stickiness`, which biases the prior toward transition matrices
with larger values along the diagonal.

These hyperparameters can be specified in the HMM constructors, and they
default to weak priors without any stickiness.


.. autoclass:: dynamax.hidden_markov_model.BernoulliHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.CategoricalHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.GammaHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.GaussianHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.DiagonalGaussianHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.SphericalGaussianHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.SharedCovarianceGaussianHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.LowRankGaussianHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.MultinomialHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.PoissonHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.GaussianMixtureHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.DiagonalGaussianMixtureHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.LinearRegressionHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.LogisticRegressionHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.CategoricalRegressionHMM
  :show-inheritance:
  :members: initialize

.. autoclass:: dynamax.hidden_markov_model.LinearAutoregressiveHMM
  :show-inheritance:
  :members: initialize, sample, compute_inputs

Low-level inference
-------------------

.. autoclass:: dynamax.hidden_markov_model.HMMPosterior
.. autoclass:: dynamax.hidden_markov_model.HMMPosteriorFiltered

.. autofunction:: dynamax.hidden_markov_model.hmm_filter
.. autofunction:: dynamax.hidden_markov_model.hmm_smoother
.. autofunction:: dynamax.hidden_markov_model.hmm_two_filter_smoother
.. autofunction:: dynamax.hidden_markov_model.hmm_fixed_lag_smoother
.. autofunction:: dynamax.hidden_markov_model.hmm_posterior_mode
.. autofunction:: dynamax.hidden_markov_model.hmm_posterior_sample
.. autofunction:: dynamax.hidden_markov_model.parallel_hmm_filter
.. autofunction:: dynamax.hidden_markov_model.parallel_hmm_smoother

Types
-----

.. autoclass:: dynamax.hidden_markov_model.HMMParameterSet
.. autoclass:: dynamax.hidden_markov_model.HMMPropertySet


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

.. autoclass:: dynamax.linear_gaussian_ssm.PosteriorGSSMFiltered
.. autoclass:: dynamax.linear_gaussian_ssm.PosteriorGSSMSmoothed

Nonlinear Gaussian GSSM
========================


High-level class
----------------

.. autoclass:: dynamax.nonlinear_gaussian_ssm.NonlinearGaussianSSM
  :members:

Low-level inference
-------------------

.. autofunction:: dynamax.nonlinear_gaussian_ssm.extended_kalman_filter
.. autofunction:: dynamax.nonlinear_gaussian_ssm.extended_kalman_smoother

.. autofunction:: dynamax.nonlinear_gaussian_ssm.unscented_kalman_filter
.. autofunction:: dynamax.nonlinear_gaussian_ssm.unscented_kalman_smoother

Types
-----

.. autoclass:: dynamax.nonlinear_gaussian_ssm.ParamsNLGSSM


Generalized Gaussian GSSM
==========================

High-level class
----------------

.. autoclass:: dynamax.generalized_gaussian_ssm.GeneralizedGaussianSSM
  :members:

Low-level inference
-------------------

.. autofunction:: dynamax.generalized_gaussian_ssm.conditional_moments_gaussian_filter
.. autofunction:: dynamax.generalized_gaussian_ssm.conditional_moments_gaussian_smoother

Types
-----

.. autoclass:: dynamax.generalized_gaussian_ssm.ParamsGGSSM

Utilities
=========

.. autofunction:: dynamax.utils.utils.find_permutation

