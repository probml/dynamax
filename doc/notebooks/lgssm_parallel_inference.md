---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: VmddSCQsVn37
outputId: c9aefbe5-c83b-4446-d522-21d8b986e7f2
---
try:
    import dynamax
except:
    %pip install -U -q git+https://github.com/probml/dynamax.git
```

```{code-cell} ipython3
:id: dc7de09a-aa24-4702-816f-4e1b21fcdaa3

import numpy as np
import jax
from jax import numpy as jnp
from jax import scipy as jsc
from jax import random as jr
from jax import vmap, tree_map, config, block_until_ready
from jax.lax import associative_scan
from matplotlib import pyplot as plt
```

+++ {"id": "zDGSCr7Oc9hk"}

# Parallel filtering and smoothing in a linear Gaussian state-space model

+++ {"id": "YmYQat2xdEZl"}

This code borrows heavily from this [example notebook](https://github.com/EEA-sensors/sequential-parallelization-examples/blob/main/python/temporal-parallelization-bayes-smoothers/parallel_kalman_jax.ipynb) from [Adrien Correnflos](https://github.com/AdrienCorenflos). Some small changes have been made here and it has been updated to work with dynamax.

The parallel inference functions implemented below are available in the dynamax library in `dynamax.linear_gaussian_ssm.parallel_inference`.

```{code-cell} ipython3
:id: 815775f5-a63a-4187-ad74-f74d22a29ed3

from dynamax.linear_gaussian_ssm.inference import LGSSMParams, lgssm_filter, lgssm_smoother, lgssm_sample
from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
```

+++ {"id": "xLMBEd3Td_qI"}

## Model
The model is a simple tracking model (see Example 3.6 in *Bayesian Filtering and Smoothing* (S. Särkkä, 2013).

```{code-cell} ipython3
:id: 6357a126-28b7-4e08-9814-9ff1f7a9ce7d

dt = 0.1
F = jnp.eye(4) + dt * jnp.eye(4, k=2)
Q = 1. * jnp.kron(jnp.array([[dt**3/3, dt**2/2],
                      [dt**2/2, dt]]), 
                 jnp.eye(2))
H = jnp.eye(2, 4)
R = 0.5 ** 2 * jnp.eye(2)
μ0 = jnp.array([0.,0.,1.,-1.])
Σ0 = jnp.eye(4)
```

```{code-cell} ipython3
:id: 25027aea-e80f-49a8-a271-142458fa03ae

latent_dim = 4
observation_dim = 2
input_dim = 1

lgssm_params = LGSSMParams(
    initial_mean = μ0,
    initial_covariance = Σ0,
    dynamics_matrix = F,
    dynamics_input_weights = jnp.zeros((latent_dim,input_dim)),
    dynamics_bias = jnp.zeros(latent_dim),
    dynamics_covariance = Q,
    emission_matrix = H,
    emission_input_weights = jnp.zeros((observation_dim, input_dim)),
    emission_bias = jnp.zeros(observation_dim),
    emission_covariance = R
)
```

```{code-cell} ipython3
:id: f46ac96e-5068-4136-a3ba-b952d86d47b9

num_timesteps = 100
key = jr.PRNGKey(0)
inputs = jnp.zeros((num_timesteps,input_dim))

key, subkey = jr.split(key)
z,emissions = lgssm_sample(subkey,lgssm_params,num_timesteps, inputs)
ssm_posterior = lgssm_smoother(lgssm_params, emissions, inputs)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 265
id: 9d548207-2976-492c-b95a-88d5d08138db
outputId: f79cf911-296f-43b2-f8c8-f091c4ea1ba8
---
plt.plot(*emissions.T,'.');
plt.plot(*ssm_posterior.filtered_means[:,:2].T, '--');
plt.plot(*ssm_posterior.smoothed_means[:,:2].T);
```

+++ {"id": "6c5f9126-e9c1-4384-91a1-50b72c1fa790"}

## Parallel Inference

+++ {"id": "rXhPImwOMQnG"}

### Filtering

```{code-cell} ipython3
:id: 55a391b9-f00b-49c5-b17b-c4ebb1735437

def first_filtering_element(params, y):
    F = params.dynamics_matrix
    H = params.emission_matrix
    Q = params.dynamics_covariance
    R = params.emission_covariance
    
    S = H @ Q @ H.T + R
    CF, low = jsc.linalg.cho_factor(S)

    m1 = F @ params.initial_mean
    P1 = F @ params.initial_covariance @ F.T + Q
    S1 = H @ P1 @ H.T + R
    K1 = jsc.linalg.solve(S1, H @ P1, assume_a="pos").T  

    A = jnp.zeros_like(F)
    b = m1 + K1 @ (y - H @ m1)
    C = P1 - K1 @ S1 @ K1.T

    eta = F.T @ H.T @ jsc.linalg.cho_solve((CF, low), y)
    J = F.T @ H.T @ jsc.linalg.cho_solve((CF, low), H @ F)
    return A, b, C, J, eta


def generic_filtering_element(params, y):
    F = params.dynamics_matrix
    H = params.emission_matrix
    Q = params.dynamics_covariance
    R = params.emission_covariance
    
    S = H @ Q @ H.T + R
    CF, low = jsc.linalg.cho_factor(S)  
    K = jsc.linalg.cho_solve((CF, low), H @ Q).T  
    A = F - K @ H @ F
    b = K @ y
    C = Q - K @ H @ Q

    eta = F.T @ H.T @ jsc.linalg.cho_solve((CF, low), y)
    J = F.T @ H.T @ jsc.linalg.cho_solve((CF, low), H @ F)
    return A, b, C, J, eta
```

```{code-cell} ipython3
:id: c7de7080-7686-4791-b013-1a45af2bc066

def make_associative_filtering_elements(params, emissions):
    first_elems = first_filtering_element(params, emissions[0])
    generic_elems = vmap(lambda ems: generic_filtering_element(params, ems))(emissions[1:])
    comb_elems = tree_map(lambda f,g: jnp.concatenate((f[None,...], g)),first_elems, generic_elems)
    return comb_elems
```

```{code-cell} ipython3
:id: fefe365e-0261-42cb-bd16-930881452904

@vmap
def filtering_operator(elem1, elem2):
    A1, b1, C1, J1, eta1 = elem1
    A2, b2, C2, J2, eta2 = elem2
    dim = A1.shape[0]
    I = jnp.eye(dim)  

    I_C1J2 = I + C1 @ J2
    temp = jsc.linalg.solve(I_C1J2.T, A2.T).T
    A = temp @ A1
    b = temp @ (b1 + C1 @ eta2) + b2
    C = temp @ C1 @ A2.T + C2

    I_J2C1 = I + J2 @ C1
    temp = jsc.linalg.solve(I_J2C1.T, A1).T

    eta = temp @ (eta2 - J2 @ b1) + eta1
    J = temp @ J2 @ A1 + J1

    return A, b, C, J, eta
```

```{code-cell} ipython3
:id: d692a7e2-45f3-4350-a3b7-45d33c2de5ab

def parallel_kalman_filter(params, emissions):
    initial_elements = make_associative_filtering_elements(params, emissions)
    final_elements = associative_scan(filtering_operator, initial_elements)
    return final_elements[1], final_elements[2]
```

```{code-cell} ipython3
:id: c7bee192-36ca-4fe0-953c-b39a5868565c

pfms, pfPs = parallel_kalman_filter(lgssm_params, emissions)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 374
id: a160e709-6f81-47f2-8404-94c2346ccd4b
outputId: b22bd04c-9995-4734-a30a-1a1581a157f9
---
plt.figure(figsize=(6,6))
plt.plot(*emissions.T,'.', label="observations")
plt.plot(*ssm_posterior.filtered_means[:,:2].T, color="C2", label="serial filtering")
plt.plot(*pfms[:,:2].T, "--", color="C3",label="parallel filtering");
plt.legend();
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 702dcd7b-9229-4d00-8107-4be74eb7a018
outputId: e05ccdc2-8f49-4b13-ce1a-54a1d96c3f8a
---
np.abs(ssm_posterior.filtered_means - pfms).max()
```

+++ {"id": "Pp55tx2jMU_9"}

### Smoothing

```{code-cell} ipython3
:id: 9ccadedd-0de9-4f1d-9e76-0e403bc33670

def last_smoothing_element(m, P):
    return jnp.zeros_like(P), m, P

def generic_smoothing_element(params, m, P):
    F = params.dynamics_matrix
    H = params.emission_matrix
    Q = params.dynamics_covariance
    R = params.emission_covariance
    Pp = F @ P @ F.T + Q

    E  = jsc.linalg.solve(Pp, F @ P, assume_a='pos').T
    g  = m - E @ F @ m
    L  = P - E @ Pp @ E.T
    return E, g, L

def make_associative_smoothing_elements(params, filtered_means, filtered_covariances):
    last_elems = last_smoothing_element(filtered_means[-1], filtered_covariances[-1])
    generic_elems = vmap(lambda m, P: generic_smoothing_element(params, m, P))(filtered_means[:-1], filtered_covariances[:-1])
    combined_elems = tree_map(lambda g,l: jnp.append(g,l[None,:],axis=0), generic_elems, last_elems)
    return combined_elems


@vmap
def smoothing_operator(elem1, elem2):
    E1, g1, L1 = elem1
    E2, g2, L2 = elem2

    E = E2 @ E1
    g = E2 @ g1 + g2
    L = E2 @ L1 @ E2.T + L2

    return E, g, L

def parallel_kalman_smoother(params, emissions):
    filtered_means, filtered_covariances = parallel_kalman_filter(params, emissions)
    initial_elements = make_associative_smoothing_elements(params, filtered_means, filtered_covariances)
    final_elements = associative_scan(smoothing_operator, initial_elements, reverse=True)
    return final_elements[1], final_elements[2]
```

```{code-cell} ipython3
:id: 86ad0108-9915-449d-934d-e51309072e79

psms, psPs = parallel_kalman_smoother(lgssm_params, emissions)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 374
id: 5ececd5c-7439-4f9e-9b56-5e99543f19cc
outputId: 6748ee00-de9e-4041-d581-333d68653f9c
---
plt.figure(figsize=(6,6))
plt.plot(*emissions.T,'.', label="observations")
plt.plot(*ssm_posterior.smoothed_means[:,:2].T, color="C2", label="serial smoothing")
plt.plot(*psms[:,:2].T, "--", color="C3",label="parallel smoothing")
plt.legend();
```

+++ {"id": "Ju0dlre3Bf9K"}

## Timing

```{code-cell} ipython3
:id: mHz92DIeJlGT

import time
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: pMkWU1emEMFG
outputId: 7f9f1b96-22b1-4666-e61e-d9e8b7df0c48
---
key = jr.PRNGKey(0)
Ts = [100, 1_000, 10_000, 100_000]
serial_filtering_durations = []
parallel_filtering_durations = []
num_repeats = 5
compiled = False

for T in Ts:
    inputs = jnp.zeros((T,input_dim))
    
    key, subkey = jr.split(key)
    z,emissions = lgssm_sample(subkey,lgssm_params,T, inputs)

    if not compiled:
        ssm_posterior = block_until_ready(lgssm_filter(lgssm_params, emissions, inputs))
        pfilt_means, pfilt_covs = block_until_ready(parallel_kalman_filter(lgssm_params, emissions))
    
    start = time.time()
    for _ in range(num_repeats):
        ssm_posterior = block_until_ready(lgssm_filter(lgssm_params, emissions, inputs))
    end = time.time()
    mean_time = (end-start)/num_repeats
    serial_filtering_durations.append(mean_time)
    print(f"Num timesteps={T}, \t time serial = {mean_time}")
    
    start = time.time()
    for _ in range(num_repeats):
        pfilt_means, pfilt_covs = block_until_ready(parallel_kalman_filter(lgssm_params, emissions))
    end = time.time()
    mean_time = (end-start)/num_repeats
    parallel_filtering_durations.append(mean_time)
    print(f"Num timesteps={T}, \t time parallel = {mean_time}")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 369
id: xpKwVlDzE7T-
outputId: f3fcd575-ef23-4de2-f886-02883a2b7c07
---
plt.figure(figsize=(5, 5))
plt.loglog(Ts, serial_filtering_durations, '-o', label='serial')
plt.loglog(Ts, parallel_filtering_durations, '-o', label='parallel')
plt.xticks(Ts)
plt.xlabel("seq. length")
plt.ylabel("time per forward pass (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 0mSknJxzLCbt
outputId: 0f4db370-d0b0-4d2f-bab3-bbe6d3128316
---
key = jr.PRNGKey(0)
Ts = [100, 1_000, 10_000, 100_000]
serial_smoothing_durations = []
parallel_smoothing_durations = []
num_repeats = 5
compiled = False

for T in Ts:
    inputs = jnp.zeros((T,input_dim))
    
    key, subkey = jr.split(key)
    z,emissions = lgssm_sample(subkey,lgssm_params,T, inputs)

    if not compiled:
        ssm_posterior = block_until_ready(lgssm_smoother(lgssm_params, emissions, inputs))
        pfilt_means, pfilt_covs = block_until_ready(parallel_kalman_smoother(lgssm_params, emissions))
    
    start = time.time()
    for _ in range(num_repeats):
        ssm_posterior = block_until_ready(lgssm_smoother(lgssm_params, emissions, inputs))
    end = time.time()
    mean_time = (end-start)/num_repeats
    serial_smoothing_durations.append(mean_time)
    print(f"Num timesteps={T}, \t time serial = {mean_time}")
    
    start = time.time()
    for _ in range(num_repeats):
        pfilt_means, pfilt_covs = block_until_ready(parallel_kalman_smoother(lgssm_params, emissions))
    end = time.time()
    mean_time = (end-start)/num_repeats
    parallel_smoothing_durations.append(mean_time)
    print(f"Num timesteps={T}, \t time parallel = {mean_time}")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 369
id: eCyil2PbNdEG
outputId: ec99476b-7bc4-48f3-86fa-7112417596de
---
plt.figure(figsize=(5, 5))
plt.loglog(Ts, serial_smoothing_durations, '-o', label='serial')
plt.loglog(Ts, parallel_smoothing_durations, '-o', label='parallel')
plt.xticks(Ts)
plt.xlabel("seq. length")
plt.ylabel("time per forward pass (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
```

```{code-cell} ipython3
:id: IpBHMUtyS2-G


```
