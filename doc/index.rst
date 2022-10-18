.. dynamax documentation master file, created by
   sphinx-quickstart on Tue Oct 18 10:21:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to DYNAMAX!
===================================

Dynamax is a library for probabiliistc state space models in [JAX](https://github.com/google/jax).




Core team: Peter Chang, Giles Harper-Donnelly, Aleyna Kara, Xinglong Li, Scott Linderman, Kevin Murphy.

Other contributors: Adrien Corenflos, Gerardo Duran-Martin, Colin Schlager.

[Full list of contributors](https://github.com/probml/dynamax/graphs/contributors)

MIT License. 2022

There are a bunch of demos in the form of python scripts and jupyter notebooks.
```
python dynamax/hmm/demos/gaussian_hmm_2d.py
```

To run all the tests, do this
```
pytest dynamax
```
To run a specific test, do something like this
```
pytest dynamax/hmm/inference_test.py
pytest dynamax/hmm/demos/demos_test.py
```

To install in colab, do this
```
%pip install git+https://github.com/probml/dynamax.git
```

To install [black](https://black.readthedocs.io/en/stable/), do this (quotes are mandatory for `zsh`)
```
pip install -U 'black[jupyter]'
```

Related libraries:

- [murphy-lab/pgm-jax](https://github.com/probml/pgm-jax): Factor graph library
- [murphy-lab/JSL](https://github.com/probml/JSL) : Deprecated library for SSMs
- [linderman-lab/ssm-jax](https://github.com/lindermanlab/ssm-jax):  Deprecated library for SSMs
- [linderman-lab/ssm](https://github.com/lindermanlab/ssm):  Old numpy, autograd, and numba library for SSMs
- [mattjj/pyhsmm](https://github.com/mattjj/pyhsmm):  Numpy and cython library library for HMMs
- [mattjj/pylds](https://github.com/mattjj/pylds):  Numpy and cython library library for linear dynamical systems
- [sarkka-lab/parallel nonlinear smoothers](https://github.com/EEA-sensors/parallel-non-linear-gaussian-smoothers) : Code for nonlinear smoothers using parallel scan


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   notebooks/bernoulli_hmm_example
   notebooks/casino_hmm
   notebooks/casino_hmm_training
   notebooks/gaussian_hmm_2d
   notebooks/bach_chorales_hmm
   notebooks/multinomial_hmm
   notebooks/poisson_hmm_changepoint
   notebooks/poisson_hmm_neurons
   notebooks/autoregressive_hmm_demo.ipynb
   notebooks/switching_linear_regression
   notebooks/mvn_dplr_demo
   notebooks/parallel_message_passing

   notebooks/kf_linreg
   notebooks/kf_tracking
   notebooks/kf_spiral
   notebooks/kf_parallel

   notebooks/lgssm_blocked_gibbs
   notebooks/lgssm_hmc
   notebooks/lgssm_hmc_param_frozen

   notebooks/pendulum
   notebooks/ekf_pendulum
   notebooks/ekf_spiral
   notebooks/ekf_mlp
   notebooks/ukf_pendulum
   notebooks/ukf_spiral

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

