# dynamax

![Logo](https://github.com/probml/dynamax/blob/main/logo/logo.gif)

![Test Status](https://github.com/probml/dynamax/actions/workflows/workflow.yml/badge.svg?branch=main)

Dynamic State Space Models in JAX.

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

