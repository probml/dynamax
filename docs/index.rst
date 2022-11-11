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
- Linear Gaussian State Space Models (aka Linear Dynamical Systems)
- Nonlinear Gaussian State Space Models
- Generalized Gaussian State Space Models (with non-Gaussian emission models)

The library consists of a set of core, functionally pure, low-level inference algorithms,
as well as a set of model classes which provide a more user-friendly, object-oriented interface.
It is compatible with other libraries in the JAX ecosystem,
such as optax_ (used for estimating parameters using stochastic gradient descent),
and Blackjax_ (used for computing the parameter posterior using Hamiltonian Monte Carlo (HMC)
or sequential Monte Carlo (SMC)).

Installation and Testing
------------------------

To install locally:

.. code-block:: console

   git clone git@github.com:probml/dynamax.git
   cd dynamax
   pip install -e .

To install in Colab:

.. code-block:: pycon

   %pip install git+https://github.com/probml/dynamax.git


To run the tests:

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

   * "Machine Learning: Advanced Topics", K. Murphy, MIT Press 2023. Available at https://probml.github.io/pml-book/book2.html.
   * "Bayesian Filtering and Smoothing", S. Särkkä, Cambridge University Press, 2013. Available at https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf



Example usage
---------------

Dynamax includes classes for many kinds of SSM.
You can use these models to simulate data, and you can fit the models using standard
learning algorithms like expectation-maximization (EM) and stochastic gradient descent (SGD).
Below we illustrate the high level (object-oriented) API for the case of an HMM
with Gaussian emissions. (See `this notebook <https://github.com/probml/dynamax/blob/main/docs/notebooks/hmm/gaussian_hmm.ipynb>`_ for
a runnable version of this code.)

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
   true_params, _ = true_hmm.initialize(key1)
   true_states, emissions = true_hmm.sample(true_params, key2, num_timesteps)

   # Make a new Gaussian HMM and fit it with EM
   test_hmm = GaussianHMM(num_states, emission_dim)
   params, props = test_hmm.initialize(key3)
   params, lls = test_hmm.fit_em(params, props, emissions)

   # Plot the marginal log probs across EM iterations
   plt.plot(lls)
   plt.xlabel("EM iterations")
   plt.ylabel("marginal log prob.")

We can also call the low-level inference code directly.



Notebooks
==================

.. toctree::
   :maxdepth: 1
   :caption: HMMs


   notebooks/hmm/casino_hmm_inference.ipynb
   notebooks/hmm/casino_hmm_learning.ipynb
   notebooks/hmm/gaussian_hmm.ipynb
   notebooks/hmm/autoregressive_hmm.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Linear Gaussian SSMs

   notebooks/linear_gaussian_ssm/kf_tracking.ipynb
   notebooks/linear_gaussian_ssm/kf_linreg.ipynb
   notebooks/linear_gaussian_ssm/lgssm_parallel_inference.ipynb
   notebooks/linear_gaussian_ssm/lgssm_learning.ipynb
   notebooks/linear_gaussian_ssm/lgssm_hmc.ipynb



.. toctree::
   :maxdepth: 1
   :caption: Nonlinear Gaussian SSMs

   notebooks/nonlinear_gaussian_ssm/ekf_ukf_spiral.ipynb
   notebooks/nonlinear_gaussian_ssm/ekf_ukf_pendulum.ipynb
   notebooks/nonlinear_gaussian_ssm/ekf_mlp.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Generalized Gaussian SSMs

   notebooks/generalized_gaussian_ssm/cmgf_logistic_regression_demo.ipynb
   notebooks/generalized_gaussian_ssm/cmgf_mlp_classification_demo.ipynb
   notebooks/generalized_gaussian_ssm/cmgf_poisson_demo.ipynb


API documentation
==================

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api


Related Libraries
==================

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



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

