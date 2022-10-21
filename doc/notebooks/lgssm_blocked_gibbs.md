---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3.8.13 64-bit
  language: python
  name: python3
---

```{code-cell} ipython3
from jax import random as  jr
from jax import numpy as jnp
from jax import jit
from itertools import count

from dynamax.linear_gaussian_ssm.inference import lgssm_smoother
from dynamax.linear_gaussian_ssm.models.linear_gaussian_ssm_conjugate import LinearGaussianConjugateSSM

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [16, 9]
```

### Generate synthetic data and initialize parameters by running MAP

```{code-cell} ipython3
from itertools import count
state_dim = 2
emission_dim = 10
num_timesteps = 100

keys = map(jr.PRNGKey, count())

true_model = LinearGaussianConjugateSSM.random_initialization(next(keys), state_dim, emission_dim)

true_states, emissions = true_model.sample(next(keys), num_timesteps)

num_iters = 100
test_model = LinearGaussianConjugateSSM.random_initialization(next(keys), state_dim, emission_dim)
marginal_lls = test_model.fit_em(jnp.array([emissions]), num_iters=num_iters, method='MAP')

# Compute predicted emissions
posterior = test_model.smoother(emissions)
smoothed_emissions = posterior.smoothed_means @ test_model.emission_matrix.value.T \
    + test_model.emission_bias.value
smoothed_emissions_cov = (
    test_model.emission_matrix.value @ posterior.smoothed_covariances @ test_model.emission_matrix.value.T
    + test_model.emission_covariance.value)
smoothed_emissions_std = jnp.sqrt(
    jnp.array([smoothed_emissions_cov[:, i, i] for i in range(emission_dim)]))
```

```{code-cell} ipython3

spc = 3
plt.figure(figsize=(10, 4))
for i in range(emission_dim):
    plt.plot(emissions[:, i] + spc * i, "--k", label="observed" if i == 0 else None)
    ln = plt.plot(smoothed_emissions[:, i] + spc * i,
                  label="smoothed" if i == 0 else None)[0]
    plt.fill_between(
        jnp.arange(num_timesteps),
        spc * i + smoothed_emissions[:, i] - 2 * jnp.sqrt(smoothed_emissions_std[i]),
        spc * i + smoothed_emissions[:, i] + 2 * jnp.sqrt(smoothed_emissions_std[i]),
        color=ln.get_color(),
        alpha=0.25,
    )
plt.xlabel("time")
plt.xlim(0, num_timesteps - 1)
plt.ylabel("true and predicted emissions")
plt.legend()
```

### Blocked Gibbs for LiearGaussianConjugateSSM

```{code-cell} ipython3
lls, param_samples = test_model.fit_blocked_gibbs(next(keys), sample_size=500, emissions=emissions)

plt.plot(lls)
```

```{code-cell} ipython3
@jit
def smooth_emission(params):
    posterior = lgssm_smoother(params, emissions)
    return posterior.smoothed_means @ params.emission_matrix.T + params.emission_bias

smoothed_emissions = jnp.array([smooth_emission(params) for params in param_samples])
smoothed_emissions_means = smoothed_emissions.mean(axis=0)
smoothed_emissions_stds = jnp.std(smoothed_emissions, axis=0)
```

```{code-cell} ipython3
spc = 3
plt.figure(figsize=(10, 4))
for i in range(emission_dim):
    plt.plot(emissions[:, i] + spc * i, "--k", label="observed" if i == 0 else None)
    ln = plt.plot(smoothed_emissions_means[:, i] + spc * i,
                  label="smoothed" if i == 0 else None)[0]
    plt.fill_between(
        jnp.arange(num_timesteps),
        spc * i + smoothed_emissions_means[:, i] - 2 * jnp.sqrt(smoothed_emissions_stds[:, i]),
        spc * i + smoothed_emissions_means[:, i] + 2 * jnp.sqrt(smoothed_emissions_stds[:, i]),
        color=ln.get_color(),
        alpha=0.25,
    )
plt.xlabel("time")
plt.xlim(0, num_timesteps - 1)
plt.ylabel("true and predicted emissions")
plt.legend()
```
