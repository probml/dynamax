# ssm-jax
![Test Status](https://github.com/probml/ssm-jax/actions/workflows/workflow.yml/badge.svg?branch=main)

State Space Models in JAX.


[List of contributors](https://github.com/probml/ssm-jax/graphs/contributors)

MIT License. 2022

To run a specific demo, do something like this
```
python ssm_jax/hmm/demos/gaussian_hmm_2d.py 
```

To run all the tests, do this
```
pytest ssm_jax
```
To run a specific test, do something like this
```
pytest ssm_jax/hmm/inference_test.py
pytest ssm_jax/hmm/demos/demos_test.py 
```

To install in colab, do this
```
%pip install git+https://github.com/probml/ssm-jax.git
```

To install [black](https://black.readthedocs.io/en/stable/), do this (quotes are mandatory for `zsh`)
```
pip install -U 'black[jupyter]'
```

Related JAX libraries:

- [murphy-lab/pgm-jax](https://github.com/probml/pgm-jax): Factor graph library
- [murphy-lab/JSL](https://github.com/probml/JSL) : Deprecated library for SSMs
- [linderman-lab/ssm-jax](https://github.com/lindermanlab/ssm-jax):  Deprecated library for SSMs
- [linderman-lab/ssm](https://github.com/lindermanlab/ssm) (numpy):  Deprecated library for SSMs
- [sarkka-lab/parallel nonlinear smoothers](https://github.com/EEA-sensors/parallel-non-linear-gaussian-smoothers) : Code for nonlinear smoothers usign parallel scan

