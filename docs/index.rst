.. dynamax documentation master file, created by
   sphinx-quickstart on Tue Oct 18 10:21:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to DYNAMAX!
===================

Dynamax is a library for probabilistic state space models (SSMs) written in JAX_.
It has code for inference (state estimation) and learning (parameter estimation)
in a variety of SSMs, including: 

- Hidden Markov Models (HMMs)
- Autoregressive Hidden Markov Models (AR-HMMs)
- Linear Gaussian State Space Models (aka Linear Dynamical Systems)
- Nonlinear Gaussian State Space Models

The library consists of a set of core, functionally pure, low-level inference algorithms,
as well as a set of model classes which provide a more user-friendly, object-oriented interface.
It is compatible with other libraries in the JAX ecosystem,
such as optax_ (useful for estimating parameters using stochastic gradient descent),
and Blackjax_ (useful for computing the posterior using Hamiltonian Monte Carlo (HMC) 
or sequential Monte Carlo (SMC)).

Installation and Testing
------------------------

To install locally, do this:

.. code-block:: console

   git clone git@github.com:probml/dynamax.git
   cd dynamax
   pip install -e .

To install in Colab, do this:

.. code-block:: pycon

   %pip install git+https://github.com/probml/dynamax.git


To run the tests, do this:

.. code-block:: console

   pytest dynamax                         # Run all tests
   pytest dynamax/hmm/inference_test.py   # Run a specific test
   pytest -k lgssm                        # Run tests with lgssm in the name



What are state space models?
-----------------------------

..
    https://sphinx-rtd-trial.readthedocs.io/en/latest/ext/math.html

A state space model or SSM is a partially observed Markov model, in which the hidden state,
:math:`z_t`, evolves over time according to a Markov process,
possibly conditional on external inputs / controls / covariates,
:math:`u_t`,
and generates an observation, 
:math:`y_t`.
This is illustrated in the graphical model below.

.. figure:: figures/LDS-UZY.png
   :scale: 100 %
   :alt: SSM as a graphical model.


The corresponding joint distribution has the following form
(in dynamax, we restrict attention to discrete time systems):

.. math::

      p(y_{1:T}, z_{1:T} | u_{1:T}) = p(z_1 | u_1) p(y_1 | z_1, u_1) \prod_{t=1}^T p(z_t | z_{t-1}, u_t) p(y_t | z_t, u_t)
   

Here :math:`p(z_t | z_{t-1}, u_t)` is called the transition or dynamics model,
and :math:`p(y_t | z_{t}, u_t)` is called the observation or emission model.
(In both cases, the inputs :math:`u_t` are optional;
furthermore, the observation model may have auto-regressive dependencies,
in which case we write  :math:`p(y_t | z_{t}, u_t, y_{1:t-1})`.)

We assume that we see the observations :math:`y_{1:T}`,
and want to infer the hidden states, either
using online filtering (i.e., computing  :math:`p(z_t|y_{1:t})`)
or offline smoothing (i.e., computing  :math:`p(z_t|y_{1:T})`).
We may also be interested in predicting future states, 
:math:`p(z_{t+h}|y_{1:t})`,
or future observations,
:math:`p(y_{t+h}|y_{1:t})`,
where h is the forecast horizon.
(Note that by using a hidden state  to represent the past observations, 
the model can have "infinite" memory, unlike a standard auto-regressive model.)
All of these computations can be done efficiently using our library,
as we discuss below.
In addition, we can estimate the parameters of the transition and emission models,
as we discuss below.

More information can be found in these books:
- *Machine Learning: Advanced Topics*, K. Murphy, MIT Press 2023. 
Available at https://probml.github.io/pml-book/book2.html.
-  *Bayesian Filtering and Smoothing*, S. Särkkä, Cambridge University Press, 2013.
Available at https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf

Inference (state estimation)
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


Learning (parameter estimation) 
---------------------------------

Dynamax also includes a host of model classes for various HMMs and linear Gaussian SSMs.
You can use these models to simulate data, and you can fit the models using standard
learning algorithms like expectation-maximization (EM) and stochastic gradient descent (SGD).

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


Related Libraries
-----------------

- distrax_hmm_: JAX functions for HMM inference (replaced by dynamax)
- filterpy_: Numpy library for (extended) Kalman filtering
- hmmlearn_: Numpy / C++ library for HMMs 
- linderman-ssm-jax_:  JAX library for SSMs (replaced by dynamax)
- linderman-ssm-numpy_:  Numpy / numba / autograd library for SSMs  (replaced by dynamax)
- mattjj-pyhsmm_:  Numpy / Cython library library for HMMs
- mattjj-pylds_:  Numpy / Cython library library for linear dynamical systems
- pgm-jax_: JAX library for factor graphs
- JSL_: JAX library for SSMs (replaced by dynamax)
- pykalman_: Numpy library for (extended) Kalman filtering
- sarkka-parallel-non-linear-gaussian-smoothers_: JAX code for nonlinear smoothers using parallel scan
- tfp_hmm_: TF2 code for HMM inference
- tfp_lgssm_: TF2 code for Kalman filtering

.. _Blackjax: https://github.com/blackjax-devs/blackjax
.. _distrax_hmm: https://github.com/deepmind/distrax/blob/master/distrax/_src/utils/hmm.py
.. _filterpy: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
.. _hmmlearn: https://github.com/hmmlearn/hmmlearn
.. _JAX: https://github.com/google/jax
.. _JSL: https://github.com/probml/JSL
.. _linderman-ssm-jax: https://github.com/lindermanlab/ssm-jax
.. _linderman-ssm-numpy: https://github.com/lindermanlab/ssm
.. _mattjj-pyhsmm: https://github.com/mattjj/pyhsmm
.. _mattjj-pylds: https://github.com/mattjj/pylds
.. _optax: https://github.com/deepmind/optax
.. _pgm-jax: https://github.com/probml/pgm-jax
.. _pykalman: https://pykalman.github.io/
.. _sarkka-parallel-non-linear-gaussian-smoothers: https://github.com/EEA-sensors/parallel-non-linear-gaussian-smoothers
.. _tfp_hmm: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel
.. _tfp_lgssm: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LinearGaussianStateSpaceModel




Notebooks
==================

.. toctree::
   :maxdepth: 1
   :caption: Contents:


   notebooks/hmm/casino_hmm_inference.ipynb
   notebooks/hmm/casino_hmm_training.ipynb
   notebooks/hmm/gaussian_hmm_2d.ipynb
   notebooks/linear_gaussian_ssm/kf_tracking.ipynb
   notebooks/linear_gaussian_ssm/kf_linreg.ipynb
   notebooks/linear_gaussian_ssm/lgssm_learning.ipynb
   notebooks/linear_gaussian_ssm/lgssm_parallel_inference.ipynb
   notebooks/nonlinear_gaussian_ssm/ekf_ukf_spiral.ipynb
   notebooks/nonlinear_gaussian_ssm/ekf_ukf_pendulum.ipynb
   notebooks/nonlinear_gaussian_ssm/ekf_mlp.ipynb


API documentation
==================

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   ssm

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

