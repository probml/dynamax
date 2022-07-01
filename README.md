# ssm-jax

State Space Models in JAX.


Authors: Peter Chang, Adrien Corenflos, Gerardo Duran-Martin,  Giles Harper-Donnelly, Aleyna Kara, Scott Linderman,  Kevin Murphy, Colin Schlager, et al.

MIT License. 2022

To run a specific demo, do something like this
```
python ssm_jax/hmm/demos/gaussian_hmm.py 
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
!git clone https://github.com/probml/ssm-jax.git
%cd ssm-jax
!pip install -e .
```

Related libraries:

- [murphy-lab/JSL](https://github.com/probml/JSL) 
- [linderman-lab/ssm-jax](https://github.com/lindermanlab/ssm-jax)
- [linderman-lab/ssm](https://github.com/lindermanlab/ssm) (numpy)
- [sarkka-lab/parallel nonlinear smoothers](https://github.com/EEA-sensors/parallel-non-linear-gaussian-smoothers) 
- [Zheng Zhao's chirpgp](https://github.com/spdes/chirpgp)
- 
