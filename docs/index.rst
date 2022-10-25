.. dynamax documentation master file, created by
   sphinx-quickstart on Tue Oct 18 10:21:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to DYNAMAX!
===================

Dynamax is a library for probabiliistc state space models in JAX_.
It has code for simulating, performing inference in, and learning the
parameters of state space models like:

- Hidden Markov Models (HMMs)
- Autoregressive Hidden Markov Models (AR-HMMs)
- Linear Gaussian State Space Models (aka Linear Dynamical Systems)
- Nonlinear Gaussian State Space Models

Using the Inference Algorithms
--------------------------------

The core inference algorithms, like the forward-backward algorithm for HMMs,
the Kalman filter and smoother for LGSSMs, and the extended and unscented
Kalman filters for nonlinear SSMs, all have a simple, functional interface.
For example, the following code generates some noisy data and then smooths it
with an LGSSM smoother (aka Kalman smoother).

.. code-block:: python

   from jax import jit, grad
   import jax.numpy as jnp
   import jax.random as jr
   import matplotlib.pyplot as plt
   from dynamax.linear_gaussian_ssm.inference import LGSSMParams, lgssm_smoother

   key = jr.PRNGKey(0)
   state_dim = 1
   emission_dim = 1
   num_timesteps = 100

   # Make some noisy data
   times = jnp.arange(num_timesteps)
   emissions = jnp.cos(2 * jnp.pi * times / 20)[:, None]
   emissions += jr.normal(key, (num_timesteps, emission_dim))

   # Specify the model parameters
   params = LGSSMParams(
      initial_mean=jnp.zeros(state_dim),
      initial_covariance=jnp.eye(state_dim),
      dynamics_matrix=jnp.eye(state_dim),
      dynamics_covariance=0.5**2 * jnp.eye(state_dim),
      emission_matrix=jnp.ones((emission_dim, state_dim)),
      emission_covariance=jnp.eye(emission_dim),
   )

   # Run the LGSSM smoother (aka Kalman smoother)
   lgssm_posterior = lgssm_smoother(params, emissions)

The posterior is a dataclass with a number of fields, including the posterior mean and posterior marginal covariances.

.. code-block:: python

   print(lgssm_posterior.smoothed_means.shape)        # (100, 1)
   print(lgssm_posterior.smoothed_covariances.shape)  # (100, 1, 1)
   print(lgssm_posterior.marginal_loglik)             # -160.31303

The inference algorithms are all written in JAX, so they support automatic
differentiation and just-in-time compilation,

.. code-block:: python

   loss = lambda params: -lgssm_smoother(params, emissions).marginal_loglik
   jit(grad(loss))(params) # Returns an LGSSMParams dataclass with the gradients in it

Fitting Models
--------------

Dynamax also includes a host of model classes for various HMMs and linear Gaussian SSMs.
You can use these models to simulate data, and you can fit the models using standard
learning algorithms like expectation-maximization (EM) and stochastic gradient descent.

.. code-block:: python

   import jax.numpy as jnp
   import jax.random as jr
   import matplotlib.pyplot as plt
   from dynamax.hmm.models import GaussianHMM

   key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
   num_states = 3
   emission_dim = 2
   num_timesteps = 100

   # Make a Gaussian HMM and sample data from it
   true_hmm = GaussianHMM(num_states, emission_dim)
   true_params, _ = true_hmm.random_initialization(key1)
   true_states, emissions = true_hmm.sample(true_params, key2, num_timesteps)

   # Make a new Gaussian HMM and fit it with EM
   test_hmm = GaussianHMM(num_states, emission_dim)
   test_params, props = test_hmm.random_initialization(key3)
   test_params, lls = test_hmm.fit_em(test_params, props, emissions)

   # Plot the marginal log probs across EM iterations
   plt.plot(lls)
   plt.xlabel("EM iterations")
   plt.ylabel("marginal log prob.")

The models also play nicely with other libraries, like Blackjax_, for Bayesian inference
with Hamiltonian Monte Carlo (HMC) and sequential Monte Carlo (SMC).

Installation and Testing
------------------------

To install locally,

.. code-block:: console

   git clone git@github.com:probml/dynamax.git
   cd dynamax
   pip install -e .

To install in Colab, do this

.. code-block:: pycon

   %pip install git+https://github.com/probml/dynamax.git


To run the tests,

.. code-block:: console

   pytest dynamax                         # Run all tests
   pytest dynamax/hmm/inference_test.py   # Run a specific test
   pytest -k lgssm                        # Run tests with lgssm in the name


Related Libraries
-----------------

- murphy-lab/pgm-jax_: Factor graph library
- murphy-lab/JSL_: Deprecated library for SSMs
- linderman-lab/ssm-jax_:  Deprecated library for SSMs
- linderman-lab/ssm_:  Old numpy, autograd, and numba library for SSMs
- mattjj/pyhsmm_:  Numpy and cython library library for HMMs
- mattjj/pylds_:  Numpy and cython library library for linear dynamical systems
- sarkka-lab/parallel-non-linear-gaussian-smoothers_: Code for nonlinear smoothers using parallel scan

.. _JAX: https://github.com/google/jax
.. _pgm-jax: https://github.com/probml/pgm-jax
.. _JSL: https://github.com/probml/JSL
.. _ssm-jax: https://github.com/lindermanlab/ssm-jax
.. _ssm: https://github.com/lindermanlab/ssm
.. _pyhsmm: https://github.com/mattjj/pyhsmm
.. _pylds: https://github.com/mattjj/pylds
.. _parallel-non-linear-gaussian-smoothers: https://github.com/EEA-sensors/parallel-non-linear-gaussian-smoothers
.. _Blackjax: https://github.com/blackjax-devs/blackjax

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   notebooks/gaussian_hmm_2d.ipynb


.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   ssm

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

